"""
System Metrics Module — CPU, RAM, Disk, GPU metrics for admin dashboard.

Uses psutil for system metrics and nvidia-smi for GPU metrics.
Caches static specs (CPU model, GPU name, total RAM) on first call.
"""

import asyncio
import logging
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_cpu_model() -> str:
    """Get CPU model name. platform.processor() is empty in Docker, so read /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or "Unknown"


@lru_cache(maxsize=1)
def _get_static_specs() -> Dict[str, Any]:
    """Get static system specs (cached, only computed once)."""
    specs: Dict[str, Any] = {
        "cpu_model": "Unknown",
        "cpu_cores": 0,
        "ram_total_gb": 0,
        "gpu_name": "Unknown",
        "gpu_vram_total_mb": 0,
    }

    if PSUTIL_AVAILABLE:
        specs["cpu_model"] = _get_cpu_model()
        specs["cpu_cores"] = psutil.cpu_count(logical=True) or 0
        specs["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    # GPU specs from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                specs["gpu_name"] = parts[0].strip()
                specs["gpu_vram_total_mb"] = int(parts[1].strip())
    except Exception:
        pass

    return specs


def _get_cpu_temperature() -> Optional[int]:
    """Get CPU temperature. Try psutil sensors, then hwmon sysfs."""
    if PSUTIL_AVAILABLE:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # AMD: k10temp/zenpower, Intel: coretemp
                for key in ("k10temp", "zenpower", "coretemp", "cpu_thermal"):
                    if key in temps and temps[key]:
                        return round(temps[key][0].current)
                # Fallback: first available sensor
                for entries in temps.values():
                    if entries:
                        return round(entries[0].current)
        except Exception:
            pass
    # Try sysfs directly (works if host mounts /sys/class/hwmon)
    try:
        import glob
        for hwmon in glob.glob("/sys/class/hwmon/hwmon*/"):
            name_path = hwmon + "name"
            temp_path = hwmon + "temp1_input"
            try:
                with open(name_path) as f:
                    name = f.read().strip()
                if name in ("k10temp", "zenpower", "coretemp"):
                    with open(temp_path) as f:
                        return round(int(f.read().strip()) / 1000)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_drive_temperature() -> Optional[int]:
    """Get NVMe/SATA drive temperature via sysfs hwmon or nvme device."""
    try:
        import glob
        # NVMe drives expose temp via /sys/class/nvme/nvme*/hwmon*/temp1_input
        for path in glob.glob("/sys/class/nvme/nvme*/hwmon*/temp1_input"):
            try:
                with open(path) as f:
                    return round(int(f.read().strip()) / 1000)
            except Exception:
                continue
        # Fallback: hwmon devices named "nvme"
        for hwmon in glob.glob("/sys/class/hwmon/hwmon*/"):
            try:
                with open(hwmon + "name") as f:
                    name = f.read().strip()
                if name == "nvme":
                    with open(hwmon + "temp1_input") as f:
                        return round(int(f.read().strip()) / 1000)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_all_gpu_specs() -> list:
    """Get specs for all NVIDIA GPUs.

    Returns list of dicts: [{index, name, vram_total_mb}, ...]
    Single-GPU systems return a one-element list. No GPU returns [].
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0].strip()),
                        "name": parts[1].strip(),
                        "vram_total_mb": int(parts[2].strip()),
                    })
            return gpus
    except Exception:
        pass
    return []


def _get_gpu_metrics_for_device(device_id: int) -> Dict[str, Any]:
    """Get current metrics for a specific GPU by device index."""
    metrics: Dict[str, Any] = {
        "gpu_utilization_pct": None,
        "gpu_vram_used_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_w": None,
        "gpu_power_limit_w": None,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(device_id),
             "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                metrics["gpu_utilization_pct"] = int(parts[0].strip())
                metrics["gpu_vram_used_mb"] = int(parts[1].strip())
                metrics["gpu_temperature_c"] = int(parts[2].strip())
            if len(parts) >= 5:
                metrics["gpu_power_w"] = round(float(parts[3].strip()))
                metrics["gpu_power_limit_w"] = round(float(parts[4].strip()))
    except Exception:
        pass
    return metrics


def _get_gpu_metrics() -> Dict[str, Any]:
    """Get current GPU utilization, VRAM, temperature, and power from nvidia-smi."""
    metrics: Dict[str, Any] = {
        "gpu_utilization_pct": None,
        "gpu_vram_used_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_w": None,
        "gpu_power_limit_w": None,
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                metrics["gpu_utilization_pct"] = int(parts[0].strip())
                metrics["gpu_vram_used_mb"] = int(parts[1].strip())
                metrics["gpu_temperature_c"] = int(parts[2].strip())
            if len(parts) >= 5:
                metrics["gpu_power_w"] = round(float(parts[3].strip()))
                metrics["gpu_power_limit_w"] = round(float(parts[4].strip()))
    except Exception:
        pass

    return metrics


async def get_system_metrics() -> Dict[str, Any]:
    """Get full system metrics (CPU, RAM, Disk, GPU).

    CPU percent is run in a thread executor to avoid blocking the event loop
    (psutil.cpu_percent blocks for `interval` seconds).
    """
    metrics: Dict[str, Any] = {
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_percent": None,
        "disk_used_gb": None,
        "disk_total_gb": None,
        "disk_percent": None,
        "swap_used_gb": None,
        "swap_total_gb": None,
        "swap_percent": None,
    }

    if PSUTIL_AVAILABLE:
        loop = asyncio.get_event_loop()

        # cpu_percent with interval=0.1 blocks briefly — run in executor
        cpu_pct = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=0.1))
        metrics["cpu_percent"] = cpu_pct

        mem = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(mem.used / (1024 ** 3), 1)
        metrics["ram_percent"] = mem.percent

        swap = psutil.swap_memory()
        metrics["swap_used_gb"] = round(swap.used / (1024 ** 3), 1)
        metrics["swap_total_gb"] = round(swap.total / (1024 ** 3), 1)
        metrics["swap_percent"] = round(swap.percent, 1) if swap.total > 0 else 0

        disk = psutil.disk_usage("/")
        metrics["disk_used_gb"] = round(disk.used / (1024 ** 3), 1)
        metrics["disk_total_gb"] = round(disk.total / (1024 ** 3), 1)
        metrics["disk_percent"] = round(disk.percent, 1)

    # Temperatures
    metrics["cpu_temperature_c"] = _get_cpu_temperature()
    metrics["drive_temperature_c"] = _get_drive_temperature()

    # GPU metrics (subprocess is fast, ~5ms)
    gpu = _get_gpu_metrics()
    metrics.update(gpu)

    # Static specs (cached)
    specs = _get_static_specs()
    metrics["specs"] = specs

    return metrics
