"""
LLM Server Manager - Process lifecycle for llama-server instances.

Manages starting, stopping, and health-checking llama-server processes
running on different ports for different model slots.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

import httpx

from services.llm_config import (
    SLOTS, EXPERT_GGUF, MODELS_DIR, LLAMA_SERVER_BIN,
    ModelSlot, get_model_path, list_available_models,
)

logger = logging.getLogger(__name__)

# Health check timeout
HEALTH_CHECK_TIMEOUT = 5.0
# Startup wait timeout (large models like GLM-4.7-Flash 17GB can take 2-3 min)
STARTUP_TIMEOUT = float(os.environ.get("LLM_STARTUP_TIMEOUT", "300"))
# Poll interval when waiting for server startup
STARTUP_POLL_INTERVAL = 1.0


class LLMServerManager:
    """
    Manages llama-server process lifecycle.

    Each slot runs a separate llama-server process on its own port.
    The manager handles starting, stopping, health checking, and
    model swapping (e.g., chat -> expert mode).
    """

    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._current_models: Dict[str, str] = {}  # slot_name -> gguf_filename

    def _build_cmd(self, slot: ModelSlot, gguf_filename: str) -> list:
        """Build llama-server command line arguments."""
        model_path = get_model_path(gguf_filename)
        cmd = [
            LLAMA_SERVER_BIN,
            "--model", str(model_path),
            "--port", str(slot.port),
            "--ctx-size", str(slot.ctx_size),
            "--n-gpu-layers", str(slot.n_gpu_layers),
            "--host", "0.0.0.0",
            "--cache-type-k", slot.cache_type_k,
            "--cache-type-v", slot.cache_type_v,
        ]
        # Multi-GPU routing and optional tensor split for a single slot.
        # Default behavior remains single-GPU with component device map pinning.
        try:
            from services.hardware import get_gpu_for_component, get_gpu_count
            gpu_id = get_gpu_for_component(slot.name)
            gpu_count = get_gpu_count()
            if slot.tensor_split:
                if gpu_count < 2:
                    logger.warning(
                        f"Slot {slot.name} configured with tensor split "
                        f"('{slot.tensor_split}') but only {gpu_count} GPU detected. "
                        "Falling back to single-GPU mode."
                    )
                    cmd.extend(["--split-mode", "none", "--main-gpu", str(gpu_id)])
                else:
                    cmd.extend(
                        [
                            "--split-mode",
                            slot.split_mode,
                            "--tensor-split",
                            slot.tensor_split,
                            "--main-gpu",
                            str(gpu_id),
                        ]
                    )
                    logger.info(
                        f"Slot {slot.name} using tensor split on {gpu_count} GPUs: "
                        f"split_mode={slot.split_mode}, tensor_split={slot.tensor_split}, "
                        f"main_gpu={gpu_id}"
                    )
            elif gpu_id != 0 or gpu_count > 1:
                cmd.extend(["--split-mode", "none", "--main-gpu", str(gpu_id)])
                logger.info(f"Slot {slot.name} targeting GPU {gpu_id}")
        except Exception:
            pass
        if slot.mlock:
            cmd.append("--mlock")
        if slot.n_cpu_moe is not None:
            cmd.extend(["--n-cpu-moe", str(slot.n_cpu_moe)])
        cmd.extend(slot.extra_args)
        return cmd

    def start(self, slot_name: str, gguf_override: str = None) -> bool:
        """Start a llama-server instance for a slot.

        Args:
            slot_name: Name of the slot (e.g., "chat", "vision")
            gguf_override: Override GGUF filename (for model swapping)

        Returns:
            True if started successfully
        """
        slot = SLOTS.get(slot_name)
        if not slot:
            logger.error(f"Unknown slot: {slot_name}")
            return False

        # Stop existing process if running
        if slot_name in self._processes:
            self.stop(slot_name)

        gguf = gguf_override or slot.gguf_filename
        model_path = get_model_path(gguf)

        if not model_path.exists():
            available = list_available_models()
            logger.error(
                f"Model file not found: {model_path}. "
                f"Available models: {available}. "
                f"Fix: update LLM_{slot_name.upper()}_MODEL in .env "
                f"to match an available file, or download the missing model."
            )
            return False

        # VRAM budget check (single-GPU pinning or aggregate multi-GPU tensor split)
        try:
            from services.hardware import (
                check_vram_budget,
                estimate_gguf_vram_mb,
                get_gpu_for_component,
                get_gpu_count,
                get_vram_available_mb,
            )
            from config import runtime_config
            gpu_id = get_gpu_for_component(slot_name)
            estimated = estimate_gguf_vram_mb(model_path, slot.ctx_size)
            if slot.tensor_split and get_gpu_count() > 1:
                # MVP behavior: budget against aggregate available VRAM across visible GPUs.
                # This avoids false blocking when a large model is intentionally sharded.
                gpu_count = get_gpu_count()
                total_available = sum(get_vram_available_mb(i) for i in range(gpu_count))
                if total_available < estimated:
                    shortfall = estimated - total_available
                    msg = (
                        f"LLM {slot_name} tensor-split budget exceeded. "
                        f"Model {gguf} needs ~{estimated} MB, total available across "
                        f"{gpu_count} GPUs is {total_available} MB (short by {shortfall} MB)."
                    )
                    if runtime_config.allow_vram_spillover:
                        logger.warning(
                            f"{msg} Proceeding anyway (ALLOW_VRAM_SPILLOVER=true)."
                        )
                    else:
                        logger.error(
                            f"{msg} Set ALLOW_VRAM_SPILLOVER=true to force load, "
                            "or use a smaller quant/model."
                        )
                        return False
            else:
                if not check_vram_budget(
                    f"LLM {slot_name} slot ({gguf})",
                    estimated,
                    allow_spillover=runtime_config.allow_vram_spillover,
                    device_id=gpu_id,
                ):
                    logger.error(
                        f"LLM {slot_name} blocked by VRAM budget. "
                        f"Model {gguf} needs ~{estimated} MB. "
                        f"Try a smaller model or set ALLOW_VRAM_SPILLOVER=true."
                    )
                    return False
        except Exception as e:
            logger.debug(f"VRAM check skipped for {slot_name}: {e}")

        cmd = self._build_cmd(slot, gguf)
        logger.info(f"Starting llama-server for {slot_name} on port {slot.port}: {gguf}")

        try:
            # Log to files to avoid pipe buffer deadlocks while preserving debug output
            log_dir = Path("/tmp/llama-server-logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = open(log_dir / f"{slot_name}-stdout.log", "w")
            stderr_log = open(log_dir / f"{slot_name}-stderr.log", "w")
            process = subprocess.Popen(
                cmd,
                stdout=stdout_log,
                stderr=stderr_log,
                preexec_fn=os.setsid,
            )
            self._processes[slot_name] = process
            self._current_models[slot_name] = gguf
            logger.info(f"llama-server started for {slot_name} (PID {process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start llama-server for {slot_name}: {e}")
            return False

    def stop(self, slot_name: str) -> bool:
        """Stop a llama-server instance.

        Args:
            slot_name: Name of the slot to stop

        Returns:
            True if stopped successfully
        """
        process = self._processes.get(slot_name)
        if not process:
            return True

        logger.info(f"Stopping llama-server for {slot_name} (PID {process.pid})")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Force killing llama-server for {slot_name}")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass  # Already dead

        del self._processes[slot_name]
        self._current_models.pop(slot_name, None)
        logger.info(f"llama-server stopped for {slot_name}")
        return True

    async def swap_model(self, slot_name: str, gguf_filename: str) -> bool:
        """Swap the model on a slot (stop -> start with new model).

        Args:
            slot_name: Slot to swap
            gguf_filename: New GGUF to load

        Returns:
            True if swap succeeded
        """
        current = self._current_models.get(slot_name)
        if current == gguf_filename:
            logger.debug(f"Slot {slot_name} already running {gguf_filename}")
            return True

        logger.info(f"Swapping {slot_name}: {current} -> {gguf_filename}")
        self.stop(slot_name)
        await asyncio.sleep(2)  # Brief pause for GPU memory release
        started = self.start(slot_name, gguf_override=gguf_filename)

        if started:
            # Wait for health check
            healthy = await self._wait_for_healthy(slot_name)
            if not healthy:
                logger.error(f"Slot {slot_name} failed health check after swap")
                return False

        return started

    async def health_check(self, slot_name: str) -> Dict[str, Any]:
        """Check health of a slot's llama-server instance.

        Args:
            slot_name: Slot to check

        Returns:
            Dict with status, model, port info
        """
        slot = SLOTS.get(slot_name)
        if not slot:
            return {"status": "unknown", "error": f"Unknown slot: {slot_name}"}

        process = self._processes.get(slot_name)
        if not process or process.poll() is not None:
            return {"status": "stopped", "slot": slot_name, "port": slot.port}

        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                resp = await client.get(f"http://localhost:{slot.port}/health")
                if resp.status_code == 200:
                    return {
                        "status": "healthy",
                        "slot": slot_name,
                        "port": slot.port,
                        "model": self._current_models.get(slot_name, "unknown"),
                        "pid": process.pid,
                    }
                else:
                    return {"status": "unhealthy", "slot": slot_name, "http_status": resp.status_code}
        except Exception as e:
            return {"status": "error", "slot": slot_name, "error": str(e)}

    async def _wait_for_healthy(self, slot_name: str, timeout: float = None) -> bool:
        """Wait for a slot to become healthy after startup.

        Uses self-calibrating timeout when available, falls back to
        STARTUP_TIMEOUT env var or 300s hard cap.
        """
        if timeout is None:
            # Self-calibrating timeout (per-slot baselines)
            try:
                from services.hardware import get_calibrated_timeout
                model = self._current_models.get(slot_name, "unknown")
                timeout = get_calibrated_timeout(slot_name, model)
            except Exception:
                timeout = STARTUP_TIMEOUT

        start = time.time()
        while time.time() - start < timeout:
            check = await self.health_check(slot_name)
            if check.get("status") == "healthy":
                elapsed = time.time() - start
                logger.info(f"Slot {slot_name} healthy after {elapsed:.1f}s")

                # Record load time for self-calibrating timeout (per-slot)
                try:
                    from services.hardware import record_load_time
                    model = self._current_models.get(slot_name, "unknown")
                    record_load_time(slot_name, model, elapsed)
                except Exception:
                    pass

                # Warm up CUDA kernels with a tiny completion so first real call is fast
                await self._warmup_inference(slot_name)
                return True
            await asyncio.sleep(STARTUP_POLL_INTERVAL)

        elapsed = time.time() - start
        logger.error(
            f"Slot {slot_name} did not become healthy within {timeout:.0f}s. "
            f"Possible causes: model too large for available VRAM, "
            f"GPU thermal throttling, or slow disk I/O. "
            f"Try: increase LLM_STARTUP_TIMEOUT, use a smaller model, "
            f"or check GPU utilization with nvidia-smi."
        )
        return False

    async def _warmup_inference(self, slot_name: str) -> None:
        """Send a tiny completion to force CUDA kernel compilation and KV cache init."""
        slot = SLOTS.get(slot_name)
        if not slot:
            return
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"http://localhost:{slot.port}/v1/chat/completions",
                    json={
                        "model": "warmup",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    },
                )
                if resp.status_code == 200:
                    logger.info(f"Slot {slot_name} inference warmup complete")
                else:
                    logger.warning(f"Slot {slot_name} warmup returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Slot {slot_name} warmup failed: {e}")

    def list_running(self) -> Dict[str, Dict[str, Any]]:
        """List all running slots with their current models."""
        result = {}
        for name, process in self._processes.items():
            alive = process.poll() is None
            slot = SLOTS.get(name)
            # Extract --parallel count from extra_args
            parallel = 1
            if slot and "--parallel" in slot.extra_args:
                try:
                    idx = slot.extra_args.index("--parallel")
                    parallel = int(slot.extra_args[idx + 1])
                except (IndexError, ValueError):
                    pass
            result[name] = {
                "alive": alive,
                "pid": process.pid if alive else None,
                "model": self._current_models.get(name),
                "port": slot.port if slot else None,
                "parallel": parallel,
            }
        return result

    def get_available_models(self) -> list:
        """List available GGUF models in the models directory."""
        return list_available_models()

    async def startup_sequence(self) -> Dict[str, bool]:
        """Start all always-running slots and wait for health.

        Starts all servers first (in parallel), then waits for health
        concurrently so slow-loading models don't block faster ones.

        Returns:
            Dict of slot_name -> success
        """
        # Phase 1: Start all processes
        to_check = []
        for name, slot in SLOTS.items():
            if slot.always_running:
                started = self.start(name)
                if started:
                    to_check.append(name)

        if not to_check:
            return {}

        # Phase 2: Wait for health on all slots concurrently
        async def check_one(slot_name: str) -> tuple:
            healthy = await self._wait_for_healthy(slot_name)
            return (slot_name, healthy)

        checks = await asyncio.gather(*[check_one(n) for n in to_check])
        return dict(checks)

    def shutdown_sequence(self) -> None:
        """Stop all running slots."""
        for name in list(self._processes.keys()):
            self.stop(name)
        logger.info("All llama-server instances stopped")

    def stop_all(self) -> None:
        """Alias for shutdown_sequence."""
        self.shutdown_sequence()


# Singleton
_server_manager: Optional[LLMServerManager] = None


def get_server_manager() -> LLMServerManager:
    """Get or create the singleton server manager."""
    global _server_manager
    if _server_manager is None:
        _server_manager = LLMServerManager()
    return _server_manager
