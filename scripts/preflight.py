#!/usr/bin/env python3
"""
ARCA preflight doctor.

Validates host/runtime prerequisites before launching the stack:
- Docker + Compose
- GPU + container GPU access
- RAM/disk/ports
- .env and model files
- Compose configuration parse

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --json
    python scripts/preflight.py --json --json-out preflight.json

Exit codes:
    0 = ready (or warnings only, unless --strict-warnings)
    1 = one or more failures (or warnings with --strict-warnings)
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ARCA_ROOT = Path(__file__).resolve().parent.parent
SMOKE_GPU_IMAGE = "alpine:3.20"
SMOKE_CUDA_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


def _run(cmd: list[str], timeout: int = 20, cwd: Path | None = None) -> tuple[int, str]:
    """Run command and return (code, combined stdout/stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except FileNotFoundError:
        return -1, ""
    except (subprocess.TimeoutExpired, Exception) as exc:
        return -2, str(exc)


def _parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _bytes_to_gb(raw_bytes: int) -> float:
    return round(raw_bytes / (1024 ** 3), 1)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.75)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _host_fix_command(
    *,
    windows: str | None = None,
    linux: str | None = None,
    mac: str | None = None,
) -> list[str]:
    system = platform.system()
    if system == "Windows" and windows:
        return [windows]
    if system == "Linux" and linux:
        return [linux]
    if system == "Darwin" and mac:
        return [mac]
    return []


def _detect_system_ram_bytes() -> int:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return 0
    if system == "Darwin":
        rc, out = _run(["sysctl", "-n", "hw.memsize"])
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
        return 0
    if system == "Windows":
        try:
            import ctypes  # pylint: disable=import-outside-toplevel

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "total_phys",
                        "avail_phys",
                        "total_page_file",
                        "avail_page_file",
                        "total_virtual",
                        "avail_virtual",
                        "avail_extended_virtual",
                    )
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
        except Exception:
            return 0
    return 0


def _json_safe_command(command: list[str]) -> str:
    return " ".join(command)


@dataclass
class CheckResult:
    check_id: str
    status: str
    message: str
    fix: str = ""
    commands: list[str] | None = None
    details: str = ""


class Reporter:
    def __init__(self, *, no_color: bool, text_output: bool):
        self.no_color = no_color
        self.text_output = text_output
        self.results: list[CheckResult] = []

    def _color(self, code: str, text: str) -> str:
        if self.no_color:
            return text
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return text
        return f"\033[{code}m{text}\033[0m"

    def _green(self, text: str) -> str:
        return self._color("32", text)

    def _yellow(self, text: str) -> str:
        return self._color("33", text)

    def _red(self, text: str) -> str:
        return self._color("31", text)

    def _bold(self, text: str) -> str:
        return self._color("1", text)

    def _emit(self, result: CheckResult) -> None:
        if not self.text_output:
            return
        if result.status == "pass":
            tag = self._green("[PASS]")
        elif result.status == "warn":
            tag = self._yellow("[WARN]")
        else:
            tag = self._red("[FAIL]")
        print(f"  {tag} {result.check_id}: {result.message}")
        if result.fix:
            print(f"         Fix: {result.fix}")
        for command in result.commands or []:
            print(f"         Cmd: {command}")
        if result.details:
            print(f"         Details: {result.details}")

    def _record(
        self,
        check_id: str,
        status: str,
        message: str,
        *,
        fix: str = "",
        commands: list[str] | None = None,
        details: str = "",
    ) -> None:
        result = CheckResult(
            check_id=check_id,
            status=status,
            message=message,
            fix=fix,
            commands=commands,
            details=details,
        )
        self.results.append(result)
        self._emit(result)

    def pass_check(self, check_id: str, message: str, *, details: str = "") -> None:
        self._record(check_id, "pass", message, details=details)

    def warn_check(
        self,
        check_id: str,
        message: str,
        *,
        fix: str = "",
        commands: list[str] | None = None,
        details: str = "",
    ) -> None:
        self._record(check_id, "warn", message, fix=fix, commands=commands, details=details)

    def fail_check(
        self,
        check_id: str,
        message: str,
        *,
        fix: str = "",
        commands: list[str] | None = None,
        details: str = "",
    ) -> None:
        self._record(check_id, "fail", message, fix=fix, commands=commands, details=details)

    def counts(self) -> dict[str, int]:
        pass_count = sum(1 for item in self.results if item.status == "pass")
        warn_count = sum(1 for item in self.results if item.status == "warn")
        fail_count = sum(1 for item in self.results if item.status == "fail")
        return {"pass": pass_count, "warn": warn_count, "fail": fail_count}

    def header(self, title: str) -> None:
        if not self.text_output:
            return
        print()
        print(f"  {self._bold(title)}")
        print(f"  {'=' * 56}")
        print()

    def footer(self, *, strict_warnings: bool) -> bool:
        counts = self.counts()
        ok = counts["fail"] == 0 and (counts["warn"] == 0 if strict_warnings else True)
        if self.text_output:
            print()
            print(f"  {'=' * 56}")
            print(
                "  Result: "
                f"{counts['pass']} passed, {counts['warn']} warning"
                f"{'' if counts['warn'] == 1 else 's'}, "
                f"{counts['fail']} failure{'' if counts['fail'] == 1 else 's'}"
            )
            if strict_warnings and counts["warn"] > 0:
                print("  Mode: strict warnings (warnings treated as failures)")
            print()
            if ok:
                print(f"  {self._green('Ready for bootstrap.')}")
                print("  Next: python scripts/arca.py bootstrap")
            else:
                print(f"  {self._red('Fix the failures above before starting ARCA.')}")
            print()
        return ok

    def to_json(self, *, strict_warnings: bool) -> dict[str, Any]:
        counts = self.counts()
        ok = counts["fail"] == 0 and (counts["warn"] == 0 if strict_warnings else True)
        return {
            "ok": ok,
            "strict_warnings": strict_warnings,
            "summary": counts,
            "system": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cwd": str(ARCA_ROOT),
            },
            "checks": [asdict(item) for item in self.results],
        }


def check_docker(reporter: Reporter) -> bool:
    rc, out = _run(["docker", "--version"])
    if rc != 0:
        reporter.fail_check(
            "docker.binary",
            "Docker CLI not found.",
            fix="Install Docker Desktop/Engine and retry.",
            commands=_host_fix_command(
                windows="winget install -e --id Docker.DockerDesktop",
                linux="curl -fsSL https://get.docker.com | sh",
                mac="brew install --cask docker",
            ),
        )
        return False

    docker_version = out.splitlines()[0].replace("Docker version ", "Docker v")
    reporter.pass_check("docker.binary", docker_version)

    rc, out = _run(["docker", "info"])
    if rc != 0:
        reporter.fail_check(
            "docker.daemon",
            "Docker daemon is not reachable.",
            fix="Start Docker and wait for the engine to become healthy.",
            commands=_host_fix_command(
                windows='Start-Process "$Env:ProgramFiles\\Docker\\Docker\\Docker Desktop.exe"',
                linux="sudo systemctl start docker",
                mac="open -a Docker",
            ),
            details=out.splitlines()[0] if out else "",
        )
        return False
    reporter.pass_check("docker.daemon", "Docker daemon reachable")

    rc, out = _run(["docker", "compose", "version"])
    if rc != 0:
        reporter.fail_check(
            "docker.compose",
            "Docker Compose v2 plugin not found.",
            fix="Install or update Docker Compose v2.",
            commands=_host_fix_command(
                windows="winget upgrade -e --id Docker.DockerDesktop",
                linux="sudo apt-get install -y docker-compose-plugin",
                mac="brew install docker-compose",
            ),
        )
        return False
    compose_version = out.splitlines()[0].replace("Docker Compose version ", "Docker Compose ")
    reporter.pass_check("docker.compose", compose_version)
    return True


def _classify_container_gpu_error(output: str) -> str:
    text = output.lower()
    network_tokens = (
        "dial tcp",
        "i/o timeout",
        "tls handshake timeout",
        "context canceled",
        "temporary failure in name resolution",
        "toomanyrequests",
        "pull access denied",
        "name or service not known",
    )
    runtime_tokens = (
        "could not select device driver",
        "no gpu devices available",
        "nvidia-container-cli",
        "unknown runtime",
        "cuda driver version is insufficient",
        "could not load nvml",
    )
    if any(token in text for token in network_tokens):
        return "network"
    if any(token in text for token in runtime_tokens):
        return "runtime"
    return "other"


def check_gpu(reporter: Reporter, *, docker_ready: bool, deep_gpu_check: bool) -> bool:
    rc, out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        reporter.warn_check(
            "gpu.host",
            "No NVIDIA GPU detected on host (or nvidia-smi unavailable).",
            fix="ARCA can run CPU-only, but GPU is strongly recommended.",
            commands=_host_fix_command(
                windows="nvidia-smi",
                linux="nvidia-smi",
                mac="(NVIDIA not typically supported on macOS)",
            ),
            details="Install/update NVIDIA driver if this machine should expose a GPU.",
        )
        return False

    rows = [line.strip() for line in out.splitlines() if line.strip()]
    gpu_count = len(rows)
    vram_values: list[int] = []
    driver = ""
    first_name = "NVIDIA GPU"
    for row in rows:
        parts = [item.strip() for item in row.split(",")]
        if not parts:
            continue
        first_name = parts[0] or first_name
        if len(parts) >= 2:
            try:
                vram_values.append(int(parts[1]))
            except ValueError:
                pass
        if len(parts) >= 3 and not driver:
            driver = parts[2]
    total_vram_gb = round(sum(vram_values) / 1024, 1) if vram_values else 0.0
    max_vram_gb = round(max(vram_values) / 1024, 1) if vram_values else 0.0
    summary = f"{gpu_count} GPU(s), total VRAM ~{total_vram_gb} GB, max GPU ~{max_vram_gb} GB"
    if driver:
        summary += f", driver {driver}"
    reporter.pass_check("gpu.host", f"{first_name} detected ({summary})")

    if not docker_ready:
        reporter.warn_check(
            "gpu.container",
            "Skipped Docker GPU checks because Docker is not ready.",
        )
        return True

    rc, out = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    if rc == 0 and "nvidia" in out.lower():
        reporter.pass_check("gpu.runtime", "NVIDIA runtime visible to Docker")
    else:
        reporter.warn_check(
            "gpu.runtime",
            "NVIDIA runtime not reported by Docker info.",
            fix="GPU containers may fail until runtime/toolkit integration is fixed.",
            commands=_host_fix_command(
                windows="wsl --status",
                linux="sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
                mac="(GPU passthrough is not generally available for Docker on macOS)",
            ),
            details=out.splitlines()[0] if out else "",
        )

    smoke_cmd = ["docker", "run", "--rm", "--pull=missing", "--gpus", "all", SMOKE_GPU_IMAGE, "true"]
    rc, out = _run(smoke_cmd, timeout=45)
    if rc == 0:
        reporter.pass_check("gpu.container", "Container GPU request (--gpus all) works")
    else:
        classification = _classify_container_gpu_error(out)
        commands = _host_fix_command(
            windows="docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L",
            linux="docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L",
            mac="(GPU container smoke tests are not supported on macOS Docker)",
        )
        if classification == "runtime":
            reporter.fail_check(
                "gpu.container",
                "Docker cannot allocate GPU devices to containers.",
                fix="Fix NVIDIA container runtime integration before bootstrap.",
                commands=commands,
                details=out.splitlines()[0] if out else _json_safe_command(smoke_cmd),
            )
        elif classification == "network":
            reporter.warn_check(
                "gpu.container",
                "Could not complete GPU smoke test because the smoke image could not be pulled.",
                fix="Network/registry issue during GPU test image pull. Retry once network is stable.",
                commands=commands,
                details=out.splitlines()[0] if out else _json_safe_command(smoke_cmd),
            )
        else:
            reporter.warn_check(
                "gpu.container",
                "GPU smoke test did not succeed.",
                fix="Run the command below and inspect the error before bootstrap.",
                commands=commands,
                details=out.splitlines()[0] if out else _json_safe_command(smoke_cmd),
            )

    if deep_gpu_check:
        inspect_cmd = ["docker", "image", "inspect", SMOKE_CUDA_IMAGE]
        rc, _ = _run(inspect_cmd)
        if rc != 0:
            reporter.warn_check(
                "gpu.deep",
                f"Skipped deep CUDA check: image not present ({SMOKE_CUDA_IMAGE}).",
                fix="Optional: run one deep CUDA check once to validate in-container nvidia-smi.",
                commands=[
                    "docker run --rm --gpus all "
                    "nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"
                ],
            )
        else:
            deep_cmd = [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--gpus",
                "all",
                SMOKE_CUDA_IMAGE,
                "nvidia-smi",
                "-L",
            ]
            rc, out = _run(deep_cmd, timeout=45)
            if rc == 0:
                lines = [line for line in out.splitlines() if line.strip()]
                details = lines[0] if lines else ""
                reporter.pass_check("gpu.deep", "Deep CUDA in-container check passed", details=details)
            else:
                reporter.fail_check(
                    "gpu.deep",
                    "Deep CUDA in-container check failed.",
                    fix="Docker can see GPUs, but CUDA tools failed in-container. Check driver/runtime alignment.",
                    commands=[
                        "docker run --rm --gpus all "
                        "nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"
                    ],
                    details=out.splitlines()[0] if out else _json_safe_command(deep_cmd),
                )

    return True


def check_ram(reporter: Reporter) -> None:
    total_bytes = _detect_system_ram_bytes()
    if total_bytes <= 0:
        reporter.warn_check("system.ram", "Could not detect system RAM.")
        return
    ram_gb = _bytes_to_gb(total_bytes)
    if ram_gb < 16:
        reporter.fail_check(
            "system.ram",
            f"System RAM is {ram_gb} GB (minimum 16 GB).",
            fix="Increase system memory for reliable local inference and indexing.",
        )
    elif ram_gb < 32:
        reporter.warn_check(
            "system.ram",
            f"System RAM is {ram_gb} GB (32 GB+ recommended).",
        )
    else:
        reporter.pass_check("system.ram", f"System RAM: {ram_gb} GB")


def check_disk(reporter: Reporter) -> None:
    try:
        usage = shutil.disk_usage(ARCA_ROOT)
    except OSError:
        reporter.warn_check("system.disk", "Could not inspect free disk space.")
        return
    free_gb = _bytes_to_gb(usage.free)
    if free_gb < 20:
        reporter.fail_check(
            "system.disk",
            f"Disk free space is {free_gb} GB (minimum 20 GB).",
            fix="Free disk space before model/image pull.",
        )
    elif free_gb < 50:
        reporter.warn_check(
            "system.disk",
            f"Disk free space is {free_gb} GB (50+ GB recommended).",
        )
    else:
        reporter.pass_check("system.disk", f"Disk free space: {free_gb} GB")


def check_ports(reporter: Reporter) -> None:
    required_ports = {
        3000: "Frontend",
        5432: "PostgreSQL",
        6333: "Qdrant",
        6379: "Redis",
        7474: "Neo4j Browser",
        7687: "Neo4j Bolt",
        8000: "Backend API",
        8080: "SearXNG",
    }
    busy = [f"{port} ({name})" for port, name in required_ports.items() if _port_open(port)]
    if not busy:
        reporter.pass_check("ports", f"All required ports available ({len(required_ports)}/{len(required_ports)})")
        return

    available = len(required_ports) - len(busy)
    commands = _host_fix_command(
        windows="docker compose down",
        linux="docker compose down",
        mac="docker compose down",
    )
    rc, out = _run(["docker", "ps", "--format", "{{.Names}}"])
    if rc == 0:
        arca_containers = sorted(
            {
                line.strip()
                for line in out.splitlines()
                if line.strip().startswith("arca-")
            }
        )
        if arca_containers:
            commands.append(f"docker rm -f {' '.join(arca_containers)}")

    reporter.warn_check(
        "ports",
        f"{available}/{len(required_ports)} required ports available; busy: {', '.join(busy)}",
        fix="Stop conflicting services or existing ARCA containers before bootstrap.",
        commands=commands,
    )


def check_env(reporter: Reporter) -> dict[str, str]:
    env_path = ARCA_ROOT / ".env"
    example_path = ARCA_ROOT / ".env.example"
    if not env_path.is_file():
        fix = "Initialize .env first."
        commands = ["python scripts/arca.py init-env"]
        if not example_path.is_file():
            fix = "Missing both .env and .env.example."
        reporter.fail_check(
            "config.env",
            ".env file not found.",
            fix=fix,
            commands=commands,
        )
        return {}

    reporter.pass_check("config.env", ".env file found")
    env = _parse_env(env_path)
    default_keys = [
        key
        for key in ("POSTGRES_PASSWORD", "NEO4J_PASSWORD")
        if env.get(key, "") == "change-me-in-production"
    ]
    if default_keys:
        reporter.warn_check(
            "config.secrets",
            f"Default passwords detected: {', '.join(default_keys)}",
            fix="Rotate generated secrets before exposing ARCA outside localhost.",
            commands=["python scripts/arca.py init-env --force"],
        )
    else:
        reporter.pass_check("config.secrets", "Database secrets are not at defaults")
    return env


def check_models(reporter: Reporter, env: dict[str, str]) -> None:
    if env.get("MCP_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        reporter.pass_check("models", "MCP_MODE=true: model download/check skipped by design")
        return

    models_dir = ARCA_ROOT / "models"
    if not models_dir.is_dir():
        reporter.warn_check(
            "models.dir",
            "models/ directory not found yet.",
            fix="Bootstrap can auto-create and auto-download configured slot models.",
            commands=["python scripts/arca.py models --yes"],
        )
        return

    ggufs = sorted(models_dir.glob("*.gguf"))
    if not ggufs:
        reporter.warn_check(
            "models.files",
            "No .gguf files found in models/.",
            fix="Bootstrap will auto-download selected tier models on first run.",
            commands=["python scripts/arca.py models --yes"],
        )
        return

    total_size_gb = round(sum(item.stat().st_size for item in ggufs) / (1024 ** 3), 1)
    reporter.pass_check("models.files", f"{len(ggufs)} GGUF file(s) detected (~{total_size_gb} GB)")

    slot_keys = (
        ("chat", "LLM_CHAT_MODEL"),
        ("code", "LLM_CODE_MODEL"),
        ("expert", "LLM_EXPERT_MODEL"),
        ("vision", "LLM_VISION_MODEL"),
        ("vision_structured", "LLM_VISION_STRUCTURED_MODEL"),
    )
    missing_slots: list[str] = []
    for slot, env_key in slot_keys:
        model_name = env.get(env_key, "").strip()
        if not model_name:
            continue
        model_path = models_dir / model_name
        if model_path.is_file():
            reporter.pass_check(f"models.slot.{slot}", f"{slot} -> {model_name}")
        else:
            missing_slots.append(f"{slot}:{model_name}")
    if missing_slots:
        reporter.warn_check(
            "models.slot.missing",
            f"Configured slot models missing: {', '.join(missing_slots)}",
            fix="Use model bootstrap to pull missing files.",
            commands=["python scripts/arca.py models --yes"],
        )


def check_compose(reporter: Reporter, *, has_gpu: bool, docker_ready: bool) -> None:
    compose_file = ARCA_ROOT / "docker-compose.yml"
    if not compose_file.is_file():
        reporter.fail_check(
            "compose.file",
            "docker-compose.yml not found.",
            fix="Ensure this is a full ARCA repository checkout.",
        )
        return
    reporter.pass_check("compose.file", "docker-compose.yml present")

    if docker_ready:
        rc, out = _run(["docker", "compose", "--compatibility", "config", "-q"], cwd=ARCA_ROOT, timeout=30)
        if rc == 0:
            reporter.pass_check("compose.config", "docker compose config parse passed")
        else:
            reporter.fail_check(
                "compose.config",
                "docker compose config parse failed.",
                fix="Resolve compose/env syntax issues before bootstrap.",
                commands=["docker compose --compatibility config"],
                details=out.splitlines()[0] if out else "",
            )
    else:
        reporter.warn_check("compose.config", "Skipped compose parse because Docker is not ready")

    if not has_gpu:
        override_path = ARCA_ROOT / "docker-compose.override.yml"
        if override_path.is_file():
            override_text = override_path.read_text(encoding="utf-8", errors="replace")
            if "nvidia" in override_text.lower() or "gpus:" in override_text.lower():
                reporter.warn_check(
                    "compose.gpu_override",
                    "GPU override exists but no GPU was detected on host.",
                    fix="Delete docker-compose.override.yml for CPU-only runs.",
                    commands=_host_fix_command(
                        windows="Remove-Item docker-compose.override.yml",
                        linux="rm docker-compose.override.yml",
                        mac="rm docker-compose.override.yml",
                    ),
                )


def run_checks(reporter: Reporter, *, deep_gpu_check: bool) -> None:
    docker_ready = check_docker(reporter)
    has_gpu = check_gpu(reporter, docker_ready=docker_ready, deep_gpu_check=deep_gpu_check)
    check_ram(reporter)
    check_disk(reporter)
    check_ports(reporter)
    env = check_env(reporter)
    check_models(reporter, env)
    check_compose(reporter, has_gpu=has_gpu, docker_ready=docker_ready)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARCA environment doctor")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    parser.add_argument("--json-out", help="Optional path to write JSON report")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as failures (CI/automation mode)",
    )
    parser.add_argument(
        "--deep-gpu-check",
        action="store_true",
        help="Run optional deep in-container CUDA check if image is locally available",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter(no_color=args.no_color, text_output=not args.json)
    if not args.json:
        reporter.header("ARCA Preflight Doctor")

    run_checks(reporter, deep_gpu_check=args.deep_gpu_check)
    ok = reporter.footer(strict_warnings=args.strict_warnings)
    report_payload = reporter.to_json(strict_warnings=args.strict_warnings)

    if args.json:
        print(json.dumps(report_payload, indent=2))
    if args.json_out:
        output_path = Path(args.json_out)
        if not output_path.is_absolute():
            output_path = (ARCA_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
