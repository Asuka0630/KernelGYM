"""
KernelGym compile service -- standalone OS process.

CUDA Backend Two-stage execution model:
    stage 1: backend.compile() runs in a CPU-only ProcessPoolExecutor
            (TORCH_CUDA_ARCH_LIST is auto-detected via nvidia-smi at
            pool startup, no user configuration needed).
    stage 2: backend.load() + correctness + performance run in the GPU
            subprocess (which skips its own backend.compile() because
            an artifact is already attached to the task payload).

This service is the stage-1 producer:
    - BRPOP cuda+kernel_code tasks from
      ``<prefix>:queue:compile:priority:{high,normal,low}``.
    - Run ``backend.compile()`` in a CPU-only ``ProcessPoolExecutor``
      (auto-detected ``TORCH_CUDA_ARCH_LIST`` so workers don't need a
      visible GPU).
    - On success: persist a slim artifact dict to
      ``<prefix>:artifact:<task_id>`` and LPUSH the task into the
      regular GPU priority queue (``<prefix>:queue:priority:*``).
    - On failure: skip the GPU dispatch and write a ``compiled=False``
      result directly to ``<prefix>:result:<task_id>`` so the client
      gets the nvcc error verbatim.

Concurrency model
-----------------
A single asyncio loop drives:

* a ``Semaphore(max_workers)`` that bounds in-flight compiles;
* one coroutine per pulled task that awaits ``run_in_executor`` over
  ``CompileOffloadPool.compile`` and finalises the result.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from kernelgym.common import Priority
from kernelgym.config import settings, setup_logging
from kernelgym.server.task_manager import TaskManager
from kernelgym.utils.error_classifier import classify_error
from kernelgym.worker.compile_offload import CompileOffloadPool

KEY_PREFIX = settings.redis_key_prefix
logger = logging.getLogger("kernelgym.compile_service")


def _task_build_dir(task_data: Dict[str, Any]) -> str:
    """Stable per-task build directory.

    Uses ``task_id`` + a short hash of the kernel source so retries with
    edited code don't reuse a stale ``.so``. The directory is shared
    between this service (stage 1) and the GPU subprocess (stage 2) over
    the local filesystem -- single-node deployment only.
    """
    root = (
        getattr(settings, "compile_offload_build_root", "")
        or os.environ.get("COMPILE_OFFLOAD_BUILD_ROOT")
        or tempfile.gettempdir()
    )
    task_id = str(task_data.get("task_id") or "unknown")
    kernel_code = task_data.get("kernel_code") or ""
    digest = hashlib.sha1(
        (task_id + "\0" + kernel_code).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    safe_task = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_id)
    return os.path.join(root, f"kernelgym_cuda_{safe_task}_{digest}")


def _build_compile_failed_result(
    task_data: Dict[str, Any],
    error_message: str,
    *,
    elapsed_total_sec: float,
    elapsed_nvcc_sec: float = 0.0,
    queue_wait_sec: float = 0.0,
) -> Dict[str, Any]:
    """Construct a ``compiled=False`` result without involving the GPU.

    Aligns the produced metadata / error_message shape with the
    in-subprocess (``ENABLE_COMPILE_OFFLOAD=false``) path.
    """
    from kernelgym.schema import (
        EvaluationResult,
        KernelEvaluationResult,
    )
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

    task_id = task_data.get("task_id", "unknown")
    base_task_id = task_data.get("base_task_id", task_id)
    task_type = task_data.get("task_type", "evaluation")

    # Best-effort recovery of the original exception class name from the
    # offload worker's "<ClassName>: <message>" prefix
    # (compile_offload._compile_in_worker writes this format), to align with
    # the in-subprocess path that records ``get_error_name(e)``.
    err_name = "compile_error"
    prefix, sep, _ = error_message.partition(": ")
    if sep and prefix and " " not in prefix and prefix.isidentifier():
        err_name = prefix

    metadata: Dict[str, Any] = {
        "compilation_error": error_message,
        "compilation_error_name": err_name,
        "phase_timings_ms": {
            "compile_offload": float(elapsed_nvcc_sec) * 1000.0,
            "compile_offload_queue_wait": float(queue_wait_sec) * 1000.0,
            # Mirror the legacy phase name so existing bench summaries
            # surface the bucket users are used to seeing.
            "load": float(elapsed_total_sec) * 1000.0,
        },
    }

    exec_result = KernelExecResult(
        compiled=False,
        correctness=False,
        decoy_kernel=False,
        runtime=-1.0,
        metadata=metadata,
    )

    kernel_view = KernelEvaluationResult.from_kernel_exec_result(
        task_id=task_id,
        base_task_id=base_task_id,
        result=exec_result,
    )

    if task_type == "kernel_evaluation":
        return kernel_view.to_dict()

    result = EvaluationResult(
        task_id=task_id,
        compiled=False,
        correctness=False,
        decoy_kernel=False,
        reference_runtime=-1.0,
        kernel_runtime=-1.0,
        speedup=0.0,
        metadata=kernel_view.metadata,
        status="failed",
        error_message=kernel_view.error_message,
        error_code=kernel_view.error_code,
    )
    return result.to_dict()


class CompileService:
    """Async compile dispatcher driving a shared :class:`CompileOffloadPool`."""

    def __init__(self) -> None:
        self.running = False
        self.redis: Optional[redis.Redis] = None
        self.task_manager: Optional[TaskManager] = None
        self.pool: Optional[CompileOffloadPool] = None
        self.max_workers = max(1, int(getattr(settings, "compile_offload_workers", 8)))
        self.timeout_sec = float(
            getattr(settings, "compile_offload_timeout_sec", 180.0) or 180.0
        )
        self.artifact_ttl_sec = max(
            300,
            int(getattr(settings, "compile_offload_timeout_sec", 180.0) or 180.0) * 4,
        )
        self._sem = asyncio.Semaphore(self.max_workers)
        self._in_flight: set[asyncio.Task] = set()
        self._stats = {
            "submitted": 0,
            "succeeded": 0,
            "failed": 0,
        }

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.running = True
        self.redis = redis.from_url(settings.redis_url)
        await self.redis.ping()
        logger.info("Compile service connected to Redis")

        self.task_manager = TaskManager(self.redis)

        try:
            self.pool = CompileOffloadPool(
                max_workers=self.max_workers,
                device_ids=list(settings.gpu_devices) or None,
                nvcc_threads=settings.compile_offload_nvcc_threads or None,
            )
        except Exception as exc:
            logger.exception(f"Failed to start CompileOffloadPool: {exc}")
            raise

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Background heartbeat so the operator can tell the service is alive.
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._dispatch_loop()
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    def _signal_handler(self, signum, _frame) -> None:
        logger.info(f"Compile service received signal {signum}; shutting down")
        self.running = False

    async def _shutdown(self) -> None:
        # Wait briefly for in-flight compiles to drain so their results
        # land in Redis before we tear down.
        if self._in_flight:
            logger.info(
                f"Draining {len(self._in_flight)} in-flight compiles " f"(up to 30s)..."
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._in_flight, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Compile drain timed out; some artifacts may be missing")

        if self.pool is not None:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.warning(f"Pool shutdown error: {exc}")
            self.pool = None

        if self.redis is not None:
            try:
                # Mark service offline.
                await self.redis.hset(
                    f"{KEY_PREFIX}:compile_service:status",
                    mapping={
                        "online": "false",
                        "stopped_at": datetime.now().isoformat(),
                        **{k: str(v) for k, v in self._stats.items()},
                    },
                )
                await self.redis.aclose()
            except Exception as exc:
                logger.warning(f"Redis shutdown error: {exc}")
            self.redis = None

        logger.info(f"Compile service stopped (stats={self._stats})")

    # ----------------------------------------------------------- main loop

    async def _dispatch_loop(self) -> None:
        assert self.task_manager is not None
        logger.info(
            f"Compile service dispatch loop started "
            f"(max_workers={self.max_workers}, timeout={self.timeout_sec}s)"
        )

        while self.running:
            await self._sem.acquire()
            if not self.running:
                self._sem.release()
                break

            # BRPOP one task. The semaphore guarantees we never have more
            # than max_workers compiles in flight.
            try:
                task_data = await self.task_manager.get_next_compile_task(timeout=1)
            except Exception as exc:
                logger.exception(f"BRPOP on compile queue failed: {exc}")
                self._sem.release()
                await asyncio.sleep(1.0)
                continue

            if task_data is None:
                # Queue idle; release the slot and try again.
                self._sem.release()
                continue

            task = asyncio.create_task(self._compile_one(task_data))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    async def _compile_one(self, task_data: Dict[str, Any]) -> None:
        try:
            await self._compile_one_inner(task_data)
        finally:
            self._sem.release()

    async def _compile_one_inner(self, task_data: Dict[str, Any]) -> None:
        assert self.task_manager is not None
        assert self.pool is not None
        task_id = task_data.get("task_id", "unknown")
        kernel_code = task_data.get("kernel_code") or ""
        if not kernel_code:
            logger.warning(
                f"Compile queue gave us task {task_id} without kernel_code; "
                f"forwarding to GPU queue unchanged"
            )
            await self._forward_to_gpu(task_data)
            return

        backend_name = (task_data.get("backend") or "cuda").strip().lower()
        if backend_name != "cuda":
            logger.warning(
                f"Compile queue gave us non-cuda task {task_id} "
                f"(backend={backend_name}); forwarding to GPU queue"
            )
            await self._forward_to_gpu(task_data)
            return

        backend_adapter = task_data.get("backend_adapter") or "kernelbench"
        entry_point = task_data.get("entry_point") or "Model"
        kernel_entry_point = f"{entry_point}New"
        build_dir = _task_build_dir(task_data)
        try:
            os.makedirs(build_dir, exist_ok=True)
        except Exception as exc:
            logger.warning(
                f"Failed to create build_dir {build_dir} for {task_id}: {exc}"
            )

        device_str = task_data.get("device") or "cuda:0"
        loop = asyncio.get_event_loop()

        # Wall-clock from the moment the task was popped off the compile
        # queue. ``elapsed_total`` includes:
        #   * waiting for a free pool slot (when WORKERS < concurrency)
        #   * the spawn/IPC hop into the ProcessPoolExecutor
        #   * the actual nvcc compile inside the pool worker
        # ``artifact["elapsed_sec"]`` from the pool worker is the *inner*
        # nvcc-only number; the difference is queue / IPC overhead and is
        # surfaced separately as ``compile_offload_queue_wait`` so the
        # bench summary can attribute slowdowns correctly.
        started = loop.time()
        self._stats["submitted"] += 1

        # Run the (synchronous) pool.compile in a thread so the asyncio
        # loop keeps draining other coroutines.
        pool = self.pool

        def _do_compile() -> Dict[str, Any]:
            return pool.compile(
                kernel_code,
                entry_point=kernel_entry_point,
                build_dir=build_dir,
                backend_name=backend_name,
                backend_adapter=backend_adapter,
                device_str=device_str,
                timeout=self.timeout_sec,
            )

        try:
            artifact = await loop.run_in_executor(None, _do_compile)
        except Exception as exc:
            logger.exception(f"Compile crashed for {task_id}: {exc}")
            artifact = {
                "compiled": False,
                "error": f"compile dispatch crashed: {type(exc).__name__}: {exc}",
                "build_dir": build_dir,
            }
        elapsed_total = loop.time() - started
        # nvcc-only (inner) elapsed reported by the pool worker; ``queue_wait``
        # is the slack between the two -- time spent waiting for a free pool
        # slot or in IPC. May be near-zero when WORKERS >= concurrency.
        elapsed_nvcc = (
            float(artifact.get("elapsed_sec") or 0.0)
            if isinstance(artifact, dict)
            else 0.0
        )
        queue_wait = max(0.0, elapsed_total - elapsed_nvcc)

        if not isinstance(artifact, dict) or not artifact.get("compiled"):
            error_message = (
                str(artifact.get("error"))
                if isinstance(artifact, dict)
                else "compile off-load failed"
            )
            logger.info(
                f"[CompileService] task {task_id} compile failed "
                f"in {elapsed_total:.2f}s "
                f"(nvcc={elapsed_nvcc:.2f}s, queue_wait={queue_wait:.2f}s): "
                f"{error_message[:300]}"
            )
            self._stats["failed"] += 1
            await self._write_compile_failure(
                task_data, error_message, elapsed_total, elapsed_nvcc, queue_wait
            )
            return

        # Stage 1 succeeded. Persist a slim artifact (drop the kernel
        # source -- already in task_data["kernel_code"] -- and the
        # verbose nvcc stdout/stderr to keep the IPC payload small).
        slim_artifact = dict(artifact)
        if slim_artifact.get("code"):
            slim_artifact["code"] = ""  # GPU side restores from custom_model_src
        slim_artifact.pop("stdout", None)
        slim_artifact.pop("stderr", None)
        slim_artifact["build_dir"] = build_dir
        # Phase timings for the toolkit pipeline to surface in the bench
        # summary. ``elapsed_sec`` (nvcc-only) is what was already there;
        # we add the outer wall-clock and the implied queue wait.
        slim_artifact["elapsed_total_sec"] = elapsed_total
        slim_artifact["queue_wait_sec"] = queue_wait

        try:
            await self.task_manager.store_artifact(
                task_id, slim_artifact, ttl_sec=self.artifact_ttl_sec
            )
        except Exception as exc:
            logger.exception(
                f"Failed to persist artifact for {task_id}: {exc}; "
                f"GPU subprocess will compile from scratch"
            )
            # We still hand the task to the GPU queue; worst case it
            # recompiles in-subprocess as before.

        self._stats["succeeded"] += 1
        logger.info(
            f"[CompileService] task {task_id} compiled in {elapsed_total:.2f}s "
            f"(nvcc={elapsed_nvcc:.2f}s, queue_wait={queue_wait:.2f}s) "
            f"at {build_dir}"
        )
        await self._forward_to_gpu(task_data)

    async def _forward_to_gpu(self, task_data: Dict[str, Any]) -> None:
        """Hand a task off to the regular GPU priority queue."""
        assert self.task_manager is not None
        try:
            priority = Priority(task_data.get("priority", Priority.NORMAL))
        except Exception:
            priority = Priority.NORMAL
        try:
            await self.task_manager.enqueue_to_gpu_queue(task_data["task_id"], priority)
        except Exception as exc:
            logger.exception(
                f"Failed to enqueue {task_data.get('task_id')} to GPU queue: {exc}"
            )

    async def _write_compile_failure(
        self,
        task_data: Dict[str, Any],
        error_message: str,
        elapsed_total_sec: float,
        elapsed_nvcc_sec: float,
        queue_wait_sec: float,
    ) -> None:
        """Persist a compiled=False result so the client gets the nvcc error."""
        assert self.task_manager is not None
        result = _build_compile_failed_result(
            task_data,
            error_message,
            elapsed_total_sec=elapsed_total_sec,
            elapsed_nvcc_sec=elapsed_nvcc_sec,
            queue_wait_sec=queue_wait_sec,
        )
        try:
            await self.task_manager.complete_task(task_data["task_id"], result)
        except Exception as exc:
            logger.exception(
                f"Failed to write compile-failure result for "
                f"{task_data.get('task_id')}: {exc}"
            )

    # --------------------------------------------------------- heartbeat

    async def _heartbeat_loop(self) -> None:
        assert self.redis is not None
        key = f"{KEY_PREFIX}:compile_service:status"
        while self.running:
            try:
                await self.redis.hset(
                    key,
                    mapping={
                        "online": "true",
                        "last_heartbeat": datetime.now().isoformat(),
                        "max_workers": str(self.max_workers),
                        "in_flight": str(len(self._in_flight)),
                        **{k: str(v) for k, v in self._stats.items()},
                    },
                )
                await self.redis.expire(key, 120)
            except Exception as exc:
                logger.warning(f"Compile service heartbeat failed: {exc}")
            await asyncio.sleep(10)


async def amain() -> None:
    setup_logging("compile_service")
    parser = argparse.ArgumentParser(description="KernelGym compile service")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="(Reserved) Record process info for the worker monitor.",
    )
    args = parser.parse_args()
    del args  # currently unused; kept for parity with single_worker.py

    service = CompileService()
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Compile service interrupted")
    except Exception as exc:
        logger.exception(f"Compile service failed: {exc}")
        sys.exit(1)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
