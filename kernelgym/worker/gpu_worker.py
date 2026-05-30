"""
GPU Worker for KernelGym - with Worker Pool Architecture.

Modified: 2025-10-30
Version: v0.3.3-rc - Worker Pool for performance optimization with CUDA error isolation
"""
import asyncio
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import redis.asyncio as redis
from contextlib import asynccontextmanager
import aiohttp
import torch

from kernelgym.config import settings
KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from kernelgym.common import ErrorCode
from kernelgym.server.task_manager import TaskManager
from kernelgym.server.code_retry_manager import CodeRetryManager
from kernelgym.utils.error_classifier import classify_error
from kernelgym.worker.subprocess_pool import (
    CompilationTimeoutError,
    RuntimeTimeoutError,
)
from aiohttp import ClientConnectorError, ClientResponseError

# Import Worker Pool for persistent subprocess workers
from kernelgym.worker.subprocess_pool import SubprocessWorkerPool

logger = logging.getLogger("kernelgym.worker")


class GPUWorker:
    """GPU worker for processing evaluation tasks."""
    
    def __init__(self, worker_id: str, device: str, redis_client: redis.Redis):
        self.worker_id = worker_id
        self.device = device
        self.redis = redis_client
        self.task_manager = TaskManager(redis_client)
        self.running = False
        self.current_task: Optional[str] = None
        self.tasks_processed = 0
        self.last_heartbeat = None
        
        # Worker statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_task_time": 0.0
        }
        
        # CUDA error tracking (for monitoring, worker pool handles auto-restart)
        self.cuda_error_count = 0
        self.max_cuda_errors_for_alert = 50  # Alert threshold (worker pool auto-restarts on CUDA errors)
        self.cuda_errors_window = []  # Track recent CUDA errors with timestamps
        self.last_cuda_error_time = None
        self.shutdown_due_to_error = False

        # Main process health tracking
        self.main_process_error_count = 0
        self.max_main_process_errors = 3  # If main process itself has errors, we need restart
        # Per-task wall-clock ceiling (seconds).
        self.per_task_timeout_sec = float(settings.default_timeout)

        # Worker Pool (NEW!)
        # Each GPU worker maintains a pool of subprocess workers
        # Pool size and per-worker task limit are configurable to enforce isolation.
        self.worker_pool: Optional[SubprocessWorkerPool] = None
        self.pool_size = getattr(settings, "worker_pool_size", 1)
        self.max_tasks_per_worker = getattr(settings, "max_tasks_per_worker", 1)
        
        # GPU device setup (主进程不使用CUDA，只存储device_id)
        # 从"cuda:N"提取device_id
        if device.startswith("cuda:"):
            self.device_id = int(device.split(":")[1])
        else:
            raise ValueError(f"Invalid device format: {device}, expected 'cuda:N'")
        
        # GPU信息缓存（用于_get_worker_info）
        self.gpu_info = {
            'name': 'Unknown',
            'total_memory': 0
        }
        
        # API server URL - handle IPv6 addresses properly
        if ':' in settings.api_host and not settings.api_host.startswith('['):
            # IPv6 address needs brackets in URL
            self.api_url = f"http://[{settings.api_host}]:{settings.api_port}"
        else:
            self.api_url = f"http://{settings.api_host}:{settings.api_port}"
        
        # HTTP session for API calls
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.node_id: Optional[str] = None
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}")
        # Stop consuming new tasks ASAP and begin shutdown
        self.running = False
        asyncio.create_task(self.stop())
    
    async def start(self):
        """Start the worker."""
        try:
            self.running = True
            logger.info(f"Starting GPU worker {self.worker_id} on device {self.device}")

            # Write initial heartbeat immediately to prevent monitor from detecting missing key
            # This happens before any potentially slow operations (API registration, GPU init)
            try:
                worker_key = f"{KEY_PREFIX}:worker:{self.worker_id}"
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "initializing",
                        "last_heartbeat": datetime.now().isoformat(),
                        "device": self.device,
                        "current_task": "",
                        "tasks_processed": "0",
                    }
                )
                await self.redis.expire(worker_key, 120)
                logger.info(f"Worker {self.worker_id} wrote initial heartbeat during startup")
            except Exception as e:
                logger.warning(f"Failed to write initial heartbeat: {e}")

            # Create HTTP session
            self.http_session = aiohttp.ClientSession()
            
            # Obtain/allocate node_id from server if not configured
            import socket
            hostname = socket.gethostname()
            if not settings.node_id:
                try:
                    url = f"{self.api_url}/node/allocate"
                    async with self.http_session.post(url, params={"hostname": hostname}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.node_id = data.get("node_id")
                            logger.info(f"Obtained server-assigned node_id={self.node_id} for hostname={hostname}")
                        else:
                            logger.warning(f"Failed to allocate node_id from server: {resp.status}")
                except Exception as e:
                    logger.warning(f"Allocate node_id error: {e}")
            else:
                self.node_id = settings.node_id

            # Register with API server
            registered = await self._register_with_api()
            if not registered:
                logger.error(f"Failed to register worker {self.worker_id}")
                raise RuntimeError("Worker registration failed")
            
            # Initialize GPU device
            await self._initialize_gpu()

            # ============================================================
            # Initialize Worker Pool (NEW!)
            # ============================================================
            logger.info(
                f"Initializing worker pool for {self.worker_id} "
                f"(device={self.device}, pool_size={self.pool_size}, "
                f"max_tasks_per_worker={self.max_tasks_per_worker})"
            )
            try:
                self.worker_pool = SubprocessWorkerPool(
                    device_id=self.device_id,
                    pool_size=self.pool_size,
                    worker_prefix=f"{self.worker_id}_pool",
                    max_tasks_per_worker=self.max_tasks_per_worker
                )
                logger.info(
                    f"Worker pool initialized successfully for {self.worker_id} "
                    f"with {self.pool_size} subprocess workers "
                    f"(max {self.max_tasks_per_worker} tasks per worker)"
                )
            except Exception as e:
                logger.error(f"Failed to initialize worker pool for {self.worker_id}: {e}")
                raise

            # Send initial heartbeat immediately after registration
            await self._update_worker_status(online=True)
            logger.info(f"Worker {self.worker_id} sent initial heartbeat")

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Start main processing loop
            processing_task = asyncio.create_task(self._processing_loop())

            # Wait for either task to complete
            await asyncio.gather(
                heartbeat_task, processing_task, return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error in worker {self.worker_id}: {e}")
            raise
        finally:
            try:
                await self.stop()
            finally:
                if self.http_session and not self.http_session.closed:
                    await self.http_session.close()
                self.http_session = None
    
    async def stop(self):
        """Stop the worker."""
        # Make stop idempotent and ensure cleanup even if running already False
        if getattr(self, "_stopping", False):
            return
        self._stopping = True
        
        # Ensure loops observe shutdown
        self.running = False
            
        logger.info(f"Stopping GPU worker {self.worker_id}")
        
        # Cancel current task if any
        if self.current_task:
            try:
                await self.task_manager.fail_task(
                    self.current_task, 
                    "Worker shutdown"
                )
            except Exception:
                pass
        
        # Unregister from API server
        if not self.shutdown_due_to_error:
            await self._unregister_from_api()
        
        # Update worker status
        await self._update_worker_status(online=False)

        # Shutdown worker pool
        if self.worker_pool:
            try:
                logger.info(f"Shutting down worker pool for {self.worker_id}...")
                await self.worker_pool.shutdown(timeout=30)
                logger.info(f"Worker pool shut down successfully for {self.worker_id}")
            except Exception as e:
                logger.error(f"Error shutting down worker pool: {e}")
            finally:
                self.worker_pool = None

        # Close HTTP session
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.http_session = None

        # GPU cleanup不再需要（主进程不使用CUDA，worker pool已清理）
        logger.info("GPU cleanup handled by worker pool shutdown")

        # Log final statistics
        logger.info(f"Worker {self.worker_id} processed {self.tasks_processed} tasks")
    
    async def _initialize_gpu(self):
        """
        验证GPU可用性（不在主进程中初始化CUDA）
        
        使用nvidia-smi验证GPU，不会触发CUDA初始化。
        GPU信息缓存用于后续的worker info查询。
        """
        try:
            from kernelgym.utils.gpu_diagnostics import GPUDiagnostics
            
            logger.info(f"Verifying GPU {self.device_id} availability (no CUDA init in main process)")
            
            # 使用nvidia-smi验证GPU（不初始化CUDA）
            health = GPUDiagnostics.test_gpu_health_nvidia_smi(self.device_id)
            
            if not health.healthy:
                raise RuntimeError(
                    f"GPU {self.device_id} not healthy: {health.error_message}"
                )
            
            # 缓存GPU信息
            self.gpu_info = {
                'name': health.device_name or 'Unknown',
                'total_memory': int(health.total_memory_gb * 1024**3) if health.total_memory_gb else 0
            }
            
            logger.info(f"GPU {self.device_id} verified successfully")
            logger.info(f"GPU Name: {health.device_name}")
            logger.info(f"GPU Memory: {health.total_memory_gb:.1f}GB")
            logger.info("Main process will NOT use CUDA (subprocess isolation enabled)")
            
        except Exception as e:
            logger.error(f"Failed to verify GPU {self.device_id}: {e}")
            raise
    
    async def _processing_loop(self):
        """Main processing loop."""
        logger.info(f"Worker {self.worker_id} processing loop started")

        while self.running:
            try:
                task_data = await self.task_manager.get_next_task(self.worker_id)

                if task_data:
                    await self._maybe_attach_artifact(task_data)
                    await self._process_task(task_data)
                else:
                    # No tasks available. get_next_task already BRPOP'd for
                    # 1s; a brief sleep avoids busy-looping when the queue
                    # is genuinely empty.
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop for worker {self.worker_id}: {e}")

                # Distinguish between subprocess errors and main process errors
                from kernelgym.server.code_retry_manager import CodeRetryManager
                if CodeRetryManager(self.redis)._is_memory_error(str(e)):
                    logger.info(f"[SUBPROCESS-ISOLATION] CUDA error detected in loop for worker {self.worker_id}, but isolated in subprocess")
                else:
                    self.main_process_error_count += 1
                    logger.warning(f"Main process error in worker {self.worker_id}: {self.main_process_error_count}/{self.max_main_process_errors}")

                    if self.main_process_error_count >= self.max_main_process_errors:
                        logger.error(f"Worker {self.worker_id} main process has too many errors. Shutting down for restart.")
                        await self.redis.hset(
                            f"{KEY_PREFIX}:worker:{self.worker_id}",
                            mapping={
                                "cuda_error_shutdown": "true",
                                "shutdown_reason": "main_process_errors",
                                "shutdown_time": datetime.now().isoformat()
                            }
                        )
                        self.running = False
                        break

                await asyncio.sleep(5)

    async def _maybe_attach_artifact(self, task_data: Dict[str, Any]) -> None:
        """If the compile_service produced an artifact for this task, attach it.

        The compile_service writes the slim artifact dict to
        ``<prefix>:artifact:<task_id>`` (HSET ``data`` as JSON).
        We pop-and-read it here and attach it to ``task_data`` so the
        toolkit pipeline skips ``backend.compile()``.

        For tasks that bypass the compile_service (triton, reference_timing,
        no kernel_code) the key simply doesn't exist and we no-op.
        """
        task_id = task_data.get("task_id")
        if not task_id:
            return
        artifact_key = f"{KEY_PREFIX}:artifact:{task_id}"
        try:
            raw = await self.redis.hget(artifact_key, "data")
        except Exception as exc:  # noqa: BLE001 - never break dispatch.
            logger.warning(
                f"Failed to read artifact for task {task_id}: {exc}; "
                f"GPU subprocess will compile from scratch."
            )
            return
        if not raw:
            return
        try:
            import json
            artifact = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as exc:  # noqa: BLE001 - corrupt payload.
            logger.warning(
                f"Failed to decode artifact for task {task_id}: {exc}"
            )
            return
        if isinstance(artifact, dict) and artifact.get("compiled"):
            task_data["precompiled_artifact"] = artifact
            build_dir = artifact.get("build_dir")
            if build_dir:
                task_data["build_dir"] = build_dir
        # Best-effort cleanup; if delete fails (e.g. eviction race) we
        # don't care -- the artifact has TTL set by compile_service.
        try:
            await self.redis.delete(artifact_key)
        except Exception:
            pass

    async def _process_task(self, task_data: Dict[str, Any]):
        """Process a single task on the GPU subprocess."""
        task_id = task_data["task_id"]
        self.current_task = task_id
        start_time = datetime.now()

        try:
            logger.info(f"Worker {self.worker_id} processing task {task_id}")

            await self._process_toolkit_task(task_data, start_time)

            self.cuda_error_count = 0

            if "_retry" in task_id:
                original_task_id = task_id.rsplit("_retry", 1)[0]
                await self.task_manager.retry_manager.clear_retry_history(original_task_id)

        except Exception as e:
            error_message = f"Task processing failed: {str(e)}"
            logger.error(f"Worker {self.worker_id} failed task {task_id}: {error_message}")

            from kernelgym.server.code_retry_manager import CodeRetryManager
            if CodeRetryManager(self.redis)._is_memory_error(str(e)):
                try:
                    if task_data.get("reference_code"):
                        logger.error(f"[MEMORY-ERROR] Task {task_id} reference_code below:\n{task_data['reference_code']}")
                    if task_data.get("kernel_code"):
                        logger.error(f"[MEMORY-ERROR] Task {task_id} kernel_code below:\n{task_data['kernel_code']}")
                except Exception:
                    pass

                self._track_cuda_error()
                logger.info(f"[SUBPROCESS-ISOLATION] CUDA error contained in subprocess for task {task_id}, worker continues normally")

            # Branch on timeout sub-type so we can emit the precise contract:
            #   * compilation timeout -> compiled=False, RUNTIME marker absent
            #   * runtime timeout     -> compiled=True, correctness=None
            #   * any other failure   -> classify by error_message
            if isinstance(e, CompilationTimeoutError):
                error_code = ErrorCode.COMPILATION_TIMEOUT
                failed_result = self._build_timeout_result(
                    task_data, error_message, error_code, kind="compile"
                )
            elif isinstance(e, RuntimeTimeoutError):
                error_code = ErrorCode.RUNTIME_TIMEOUT
                failed_result = self._build_timeout_result(
                    task_data, error_message, error_code, kind="runtime"
                )
            else:
                error_code = classify_error(str(e), "runtime")
                failed_result = self._build_failed_result(task_data, error_message, error_code)
            await self.task_manager.complete_task(task_id, failed_result)

            self.stats["tasks_failed"] += 1

        finally:
            self.current_task = None
            self.tasks_processed += 1

    async def _process_toolkit_task(
        self, task_data: Dict[str, Any], start_time: datetime
    ):
        """Process task via toolkit/backend abstractions."""
        task_id = task_data["task_id"]

        task_data["device"] = self.device
        if "toolkit" not in task_data:
            raise ValueError("Task payload missing required 'toolkit'")
        if "backend_adapter" not in task_data:
            raise ValueError("Task payload missing required 'backend_adapter'")

        import time as _time
        _dispatch_start = _time.perf_counter()
        result_dict = await self._run_toolkit_task(task_data)
        _dispatch_elapsed = _time.perf_counter() - _dispatch_start

        # Annotate the result with how long this worker layer spent dispatching
        # the task (await + thread-pool + subprocess IPC).
        try:
            md = result_dict.get("metadata") if isinstance(result_dict, dict) else None
            if isinstance(md, dict):
                bucket = md.setdefault("phase_timings_ms", {})
                for key in list(bucket.keys()):
                    bucket[f"worker_dispatch.{key}"] = bucket.pop(key)
                dispatch_ms = float(_dispatch_elapsed) * 1000.0
                bucket["worker_dispatch"] = dispatch_ms
                total_ms = bucket.pop("worker_dispatch.total", None)
                if isinstance(total_ms, (int, float)) and total_ms > 0:
                    overhead_ms = dispatch_ms - float(total_ms)
                    if overhead_ms > 0:
                        bucket["worker_dispatch.overhead"] = overhead_ms
        except Exception:
            pass

        status = result_dict.get("status")
        error_message = result_dict.get("error_message") or "Task failed"
        error_code = result_dict.get("error_code")

        if status != "completed":
            if error_code is None:
                error_code = classify_error(error_message, "runtime")
                result_dict["error_code"] = error_code
            result_dict["status"] = "failed"
            result_dict["error_message"] = error_message

        await self.task_manager.complete_task(task_id, result_dict)

        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_task_stats(processing_time, status == "completed")

        logger.info(
            f"Worker {self.worker_id} completed task {task_id} in {processing_time:.2f}s"
        )

    def _build_failed_result(
        self,
        task_data: Dict[str, Any],
        error_message: str,
        error_code: Any,
    ) -> Dict[str, Any]:
        from kernelgym.schema import (
            EvaluationResult,
            KernelEvaluationResult,
            ReferenceTimingResult,
        )

        task_id = task_data.get("task_id", "unknown")
        base_task_id = task_data.get("base_task_id", task_id)
        task_type = task_data.get("task_type", "evaluation")

        metadata = {"error": error_message}

        if task_type == "reference_timing":
            result = ReferenceTimingResult(
                task_id=task_id,
                base_task_id=base_task_id,
                reference_runtime=-1.0,
                metadata=metadata,
                status="failed",
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        if task_type == "kernel_evaluation":
            result = KernelEvaluationResult(
                task_id=task_id,
                base_task_id=base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata=metadata,
                status="failed",
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        result = EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata=metadata,
            status="failed",
            error_message=error_message,
            error_code=error_code,
        )
        return result.to_dict()

    def _build_timeout_result(
        self,
        task_data: Dict[str, Any],
        error_message: str,
        error_code: ErrorCode,
        kind: str,
    ) -> Dict[str, Any]:
        """Build a result dict for a wall-clock timeout, encoding which phase
        the subprocess was in when killed.

        Contract:
          * kind="compile" -> compiled=False, correctness=None
              (toolkit never produced a usable kernel binary).
          * kind="runtime" -> compiled=True,  correctness=None
              (kernel built fine; correctness/perf hung -- this is a runtime
              perf bug, NOT a compile bug. Downstream debug agents must not
              be told to look for syntax/include issues.)
        """
        from kernelgym.schema import (
            EvaluationResult,
            KernelEvaluationResult,
            ReferenceTimingResult,
        )

        task_id = task_data.get("task_id", "unknown")
        base_task_id = task_data.get("base_task_id", task_id)
        task_type = task_data.get("task_type", "evaluation")

        compiled_flag: Optional[bool] = False if kind == "compile" else True

        metadata = {
            "error": error_message,
            "timeout_kind": kind,           # "compile" | "runtime"
            "timeout_phase": kind,          # alias: same info, easier to grep
        }

        if task_type == "reference_timing":
            # Reference timing has no compile step; any timeout there is by
            # definition a runtime timeout of the baseline pipeline.
            result = ReferenceTimingResult(
                task_id=task_id,
                base_task_id=base_task_id,
                reference_runtime=-1.0,
                metadata=metadata,
                status="failed",
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        if task_type == "kernel_evaluation":
            result = KernelEvaluationResult(
                task_id=task_id,
                base_task_id=base_task_id,
                compiled=compiled_flag,
                correctness=None,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata=metadata,
                status="failed",
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        result = EvaluationResult(
            task_id=task_id,
            compiled=compiled_flag,
            correctness=None,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata=metadata,
            status="failed",
            error_message=error_message,
            error_code=error_code,
        )
        return result.to_dict()

    async def _run_toolkit_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run task payload through worker pool."""
        # Effective budget = min(operator ceiling, payload.timeout).
        # ``payload.timeout`` is pydantic-validated server-side
        # (EvaluationRequest: ge=10, le=3600) so it's always a positive
        # int by the time we see it.
        per_task_timeout_sec = min(
            self.per_task_timeout_sec, float(task_data["timeout"])
        )
        logger.info(
            f"[Worker] payload.timeout={task_data['timeout']}s, "
            f"ceiling={self.per_task_timeout_sec}s, "
            f"effective={per_task_timeout_sec}s"
        )

        result_data = await self.worker_pool.execute_task(
            task_data,
            timeout=per_task_timeout_sec,
            max_retries=2,
        )

        if not result_data.get("success", False):
            error_type = result_data.get("error_type", "Unknown")
            error_message = result_data.get("error_message", "Unknown error")
            raise RuntimeError(f"{error_type}: {error_message}")

        return result_data["result"]

    
    def _update_task_stats(self, processing_time: float, success: bool):
        """Update task statistics."""
        if success:
            self.stats["tasks_completed"] += 1
        else:
            self.stats["tasks_failed"] += 1
        
        self.stats["total_processing_time"] += processing_time
        completed_tasks = self.stats["tasks_completed"]
        if completed_tasks > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / completed_tasks
            )
        self.stats["last_task_time"] = processing_time
    
    def _track_cuda_error(self):
        """
        Track CUDA errors for monitoring purposes.
        
        In subprocess isolation architecture, CUDA errors don't require worker restart,
        but we still track them to detect anomalies and potential issues.
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        self.cuda_error_count += 1
        self.last_cuda_error_time = now
        self.cuda_errors_window.append(now)
        
        # Keep only errors from last 5 minutes
        cutoff = now - timedelta(minutes=5)
        self.cuda_errors_window = [t for t in self.cuda_errors_window if t > cutoff]
        
        # Log warning if too many errors in short time
        if len(self.cuda_errors_window) >= self.max_cuda_errors_for_alert:
            logger.warning(
                f"[MONITORING] Worker {self.worker_id} has {len(self.cuda_errors_window)} CUDA errors in last 5 minutes. "
                f"Total: {self.cuda_error_count}. This is high but subprocess isolation is handling them."
            )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to indicate worker is alive."""
        while self.running:
            try:
                # 先发 API 心跳，只有服务端接受后才更新 Redis 状态，避免幽灵条目
                ok = await self._send_heartbeat_to_api()
                if not ok:
                    # _send_heartbeat_to_api 内已处理停机/剔除
                    break
                # Update Redis status（仅当 API 接受心跳时）
                await self._update_worker_status(online=True)
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop for worker {self.worker_id}: {e}")
                await asyncio.sleep(20)  # Sleep on error, then retry
    
    async def _update_worker_status(self, online: bool):
        """Update worker status in Redis."""
        try:
            worker_key = f"{KEY_PREFIX}:worker:{self.worker_id}"
            
            if online:
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "true",
                        "last_heartbeat": datetime.now().isoformat(),
                        "current_task": self.current_task or "",
                        "tasks_processed": str(self.tasks_processed),
                        "device": self.device,
                        "stats": str(self.stats)
                    }
                )
                # Set expiration for heartbeat (120s). Monitor handles persistence for expected workers.
                await self.redis.expire(worker_key, 120)
            else:
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "false",
                        "last_heartbeat": datetime.now().isoformat(),
                        "current_task": "",
                        "tasks_processed": str(self.tasks_processed),
                        "device": self.device,
                        "stats": str(self.stats)
                    }
                )
                # Ensure offline records expire to avoid long-term residue
                await self.redis.expire(worker_key, 120)
                
        except Exception as e:
            logger.error(f"Failed to update worker status for {self.worker_id}: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "running": self.running,
            "current_task": self.current_task,
            "tasks_processed": self.tasks_processed,
            "stats": self.stats,
            "gpu_info": {
                "name": self.gpu_info.get('name', 'Unknown'),
                "memory_total": self.gpu_info.get('total_memory', 0),
                # 主进程不使用CUDA，无法获取实时内存使用
                "memory_allocated": 0,
                "memory_reserved": 0
            }
        }
    
    async def _register_with_api(self) -> bool:
        """Register worker with the API server."""
        try:
            if not self.http_session:
                logger.error("HTTP session not initialized")
                return False
                
            url = f"{self.api_url}/worker/register"
            print(f"[DEBUG]: url: {url}")
            import socket
            hostname = socket.gethostname()
            node_id = self.node_id or settings.node_id or hostname
            params = {"worker_id": self.worker_id, "device": self.device, "node_id": node_id, "hostname": hostname}
            
            # Retry register until API ready (e.g., when server just started)
            retry_deadline = asyncio.get_event_loop().time() + 60.0
            last_err = None
            while True:
                try:
                    async with self.http_session.post(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully registered with API server: {data}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to register with API server: {response.status} - {error_text}")
                            last_err = RuntimeError(f"HTTP {response.status}")
                except (ClientConnectorError, ClientResponseError) as e:
                    last_err = e
                    logger.warning(f"API not ready for worker register: {e}. Retrying...")
                except Exception as e:
                    last_err = e
                    logger.warning(f"Register error: {e}. Retrying...")
                if asyncio.get_event_loop().time() > retry_deadline:
                    logger.error(f"Worker register timeout: {last_err}")
                    return False
                await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error registering with API server: {e}")
            return False
    
    async def _unregister_from_api(self) -> bool:
        """Unregister worker from the API server."""
        try:
            if not self.http_session:
                return True
                
            url = f"{self.api_url}/worker/unregister"
            params = {"worker_id": self.worker_id}
            
            async with self.http_session.post(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully unregistered from API server: {data}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to unregister from API server: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error unregistering from API server: {e}")
            return False
    
    async def _send_heartbeat_to_api(self) -> bool:
        """Send heartbeat to API server."""
        try:
            if not self.http_session:
                return False
                
            url = f"{self.api_url}/worker/heartbeat"
            import socket
            hostname = socket.gethostname()
            node_id = self.node_id or settings.node_id or hostname
            params = {"worker_id": self.worker_id, "device": self.device, "node_id": node_id, "hostname": hostname}
            
            async with self.http_session.post(url, params=params) as response:
                if response.status == 200:
                    return True
                # 如果被拒绝（如409/410），主动停机，避免“幽灵心跳”
                logger.warning(f"Failed to send heartbeat: HTTP {response.status}; shutting down worker {self.worker_id}")
                # 标记，避免监控误判
                self.shutdown_due_to_error = True
                # 尝试从LB剔除，防止残留
                try:
                    evict_url = f"{self.api_url}/worker/evict_from_lb"
                    await self.http_session.post(evict_url, params={"worker_id": self.worker_id})
                except Exception:
                    pass
                # 主动停止
                self.running = False
                # Clear current_task to avoid duplicate fail_task in stop()
                self.current_task = None
                await self.stop()
                return False
                    
        except Exception as e:
            logger.error(f"Error sending heartbeat to API server: {e}")
            return False


class WorkerManager:
    """Manages multiple GPU workers."""
    
    def __init__(self):
        self.workers: Dict[str, GPUWorker] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
    
    async def start(self):
        """Start all workers."""
        try:
            self.running = True
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established for worker manager")
            
            # Create workers for each GPU device
            worker_tasks = []
            import socket
            node_id = settings.node_id or socket.gethostname()
            for device in settings.gpu_devices:
                device_name = f"cuda:{device}"
                worker_id = f"{node_id}_gpu_{device}"
                worker = GPUWorker(worker_id, device_name, self.redis_client)
                self.workers[worker_id] = worker
                
                # Start worker in background
                worker_task = asyncio.create_task(worker.start())
                worker_tasks.append(worker_task)
            
            logger.info(f"Started {len(self.workers)} GPU workers")
            
            # Wait for all workers to complete
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in worker manager: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop all workers."""
        if not self.running:
            return
            
        logger.info("Stopping worker manager")
        self.running = False
        
        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Worker manager stopped")
    
    async def get_workers_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        status = {}
        for device, worker in self.workers.items():
            status[device] = await worker.get_stats()
        return status


async def main():
    """Main entry point for GPU workers."""
    # Configure logging with file support
    logger = setup_logging("worker")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    available_devices = torch.cuda.device_count()
    required_devices = max(settings.gpu_devices) + 1 if settings.gpu_devices else 1
    
    if available_devices < required_devices:
        logger.error(f"Not enough GPUs available. Required: {required_devices}, Available: {available_devices}")
        sys.exit(1)
    
    # Start worker manager
    worker_manager = WorkerManager()
    
    try:
        await worker_manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker manager error: {e}")
        sys.exit(1)
    finally:
        await worker_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
