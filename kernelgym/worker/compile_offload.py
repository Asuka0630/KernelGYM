"""
Off-GPU compile service for the KernelBench *cuda* backend.

This module off-loads phase-1 (compile) onto a process-wide CPU pool so
multiple kernels can ``nvcc`` in parallel. The GPU subprocess receives a
ready-to-load artifact and proceeds straight to ``backend.load()`` (a
sub-second ``dlopen``) followed by correctness + performance.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kernelgym.compile_offload")


def detect_cuda_arch_list(device_ids: Optional[List[int]] = None) -> str:
    """Return a ``TORCH_CUDA_ARCH_LIST`` value for the locally visible GPUs

    runs once in the main process at pool startup.
    """

    cmd = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    if device_ids:
        cmd.extend(["-i", ",".join(str(i) for i in device_ids)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi unavailable for arch detection: %s", exc)
        return ""

    if result.returncode != 0:
        logger.warning(
            "nvidia-smi compute_cap query failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return ""

    # Output is one ``M.m`` per line, possibly with duplicates across
    # identical GPUs. Preserve insertion order so the dominant arch
    # appears first, but de-duplicate.
    seen: List[str] = []
    for line in result.stdout.splitlines():
        cap = line.strip()
        if cap and cap not in seen:
            seen.append(cap)

    return ";".join(seen)


def _compile_worker_initializer(
    cuda_arch_list: str,
    nvcc_threads: Optional[str],
) -> None:
    """Initializer for each CPU compile worker.

    Sets the env vars that govern ``torch.utils.cpp_extension`` *before*
    ``torch`` is imported, so the very first ``load_inline`` call picks
    them up. We deliberately do **not** import torch here; the worker
    imports it lazily on the first task to keep idle pool workers cheap.

    ``CUDA_VISIBLE_DEVICES=""`` hides every GPU. Without that, PyTorch's
    cpp_extension would try to bind to the GPU during the build and we'd
    lose the entire point of off-loading. Cross-compilation works as
    long as ``TORCH_CUDA_ARCH_LIST`` is set, which is why we forward the
    auto-detected list from the main process.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
    if nvcc_threads:
        os.environ["NVCC_THREADS"] = nvcc_threads
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


def _compile_in_worker(
    code: str,
    *,
    entry_point: str,
    backend_name: str,
    backend_adapter: str,
    build_dir: str,
    device_str: str,
) -> Dict[str, Any]:
    """Run ``backend.compile`` for a kernel source in a CPU-only worker.

    Returns the artifact dict as produced by ``KernelBenchCudaBackend.compile``.
    On unexpected failures we still return a well-formed artifact
    (``compiled=False`` + ``error``) so the main process never has to
    handle exceptions raised across the process boundary.
    """

    started = time.perf_counter()
    try:
        from kernelgym.backend import get_backend

        # ----------------------------------------------------------------
        # Reset PyTorch cpp_extension's per-process version cache before
        # every compile. The cache (``JIT_EXTENSION_VERSIONER.entries``,
        # keyed by extension ``name``) bumps the version number each time
        # the same name is loaded with different
        # (sources, build_arguments, build_directory) -- and since this
        # pool worker handles many tasks with different per-task
        # build_dirs, the version monotonically grows here. Stage-2 (a
        # fresh GPU subprocess) always starts at version 0, so the .so
        # filename it expects (``<name>.so``) doesn't match what stage-1
        # wrote (``<name>_v{N}.so``) -- causing stage-2 to redo nvcc.
        #
        # Clearing the cache forces stage-1 to also start at version 0
        # for every task, so both stages converge on ``<name>.so`` and
        # stage-2 hits the build_dir cache (sub-second dlopen).
        try:
            from torch.utils.cpp_extension import JIT_EXTENSION_VERSIONER
            JIT_EXTENSION_VERSIONER.entries.clear()
        except Exception:
            pass

        backend = get_backend(backend_adapter)
        artifact = backend.compile(
            code,
            device=device_str,
            backend=backend_name,
            entry_point=entry_point,
            build_dir=build_dir,
        )
        if not isinstance(artifact, dict):
            artifact = {
                "compiled": False,
                "error": f"backend.compile() returned non-dict: {type(artifact)}",
            }
        artifact.setdefault("build_dir", build_dir)
        artifact.setdefault("entry_point", entry_point)
        artifact.setdefault("backend", backend_name)
        artifact.setdefault("code", code)
        artifact["elapsed_sec"] = time.perf_counter() - started
        return artifact
    except BaseException as exc:  # noqa: BLE001 - boundary-crossing safeguard.
        return {
            "compiled": False,
            "error": f"{type(exc).__name__}: {exc}",
            "build_dir": build_dir,
            "entry_point": entry_point,
            "backend": backend_name,
            "code": code,
            "elapsed_sec": time.perf_counter() - started,
        }


class CompileOffloadPool:
    """CPU process pool that synchronously off-loads CUDA-extension compilation.

    Public API is a single blocking method, :meth:`compile`. Tune
    ``max_workers`` to match available CPU cores / RAM headroom; ``2 *
    num_gpus`` is a good starting point.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        device_ids: Optional[List[int]] = None,
        cuda_arch_list: Optional[str] = None,
        nvcc_threads: Optional[str] = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")

        # Auto-detect arch list from nvidia-smi when the caller didn't
        # provide one. Falls back to empty string (and the worker will
        # surface a clear error) if no GPUs are visible.
        if not cuda_arch_list:
            cuda_arch_list = detect_cuda_arch_list(device_ids)
        if not cuda_arch_list:
            raise RuntimeError(
                "Could not auto-detect TORCH_CUDA_ARCH_LIST via nvidia-smi. "
                "Pass cuda_arch_list= explicitly (e.g. '7.0' for V100, "
                "'8.0' for A100, '9.0' for H100)."
            )

        self._max_workers = max_workers
        self._cuda_arch_list = cuda_arch_list
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_compile_worker_initializer,
            initargs=(cuda_arch_list, nvcc_threads),
        )
        self._lock = threading.Lock()
        self._closed = False

        self._submitted = 0
        self._succeeded = 0
        self._failed = 0
        # build_dir -> in-flight Future. New callers targeting an
        # in-flight build_dir share the same future instead of submitting
        # a duplicate compile.
        self._dedup: Dict[str, Any] = {}

        logger.info(
            "CompileOffloadPool started (max_workers=%d, TORCH_CUDA_ARCH_LIST=%s)",
            max_workers,
            cuda_arch_list,
        )

    # ------------------------------------------------------------------ API

    def compile(
        self,
        code: str,
        *,
        entry_point: str,
        build_dir: str,
        backend_name: str = "cuda",
        backend_adapter: str = "kernelbench",
        device_str: str = "cuda:0",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synchronously off-load compilation to a CPU worker.

        Always returns a well-formed artifact dict; check
        ``artifact["compiled"]`` for success.

        Parameters
        ----------
        code : str
            Kernel source code (must define ``entry_point``).
        entry_point : str
            Class name to look up in the compiled module (typically
            ``"ModelNew"``).
        build_dir : str
            Filesystem path where ``torch.utils.cpp_extension`` will write
            its build artifacts. **Must match the path used by the GPU
            subprocess** so the cached ``.so`` is reused.
        backend_name : str
            Always ``"cuda"`` here -- triton tasks should not reach this
            method.
        backend_adapter : str
            Toolkit-level backend adapter, currently ``"kernelbench"``.
        device_str : str
            Carried through into the artifact for downstream
            ``backend.load()`` -- not used during compile itself.
        timeout : float, optional
            Hard wall-clock cap; on timeout returns ``compiled=False``.
        """

        if not code:
            return self._fail(build_dir, entry_point, backend_name, code, "empty kernel source")

        with self._lock:
            executor = self._executor
            if self._closed or executor is None:
                return self._fail(
                    build_dir, entry_point, backend_name, code,
                    "CompileOffloadPool is closed",
                )

            future = self._dedup.get(build_dir)
            if future is None:
                try:
                    future = executor.submit(
                        _compile_in_worker,
                        code,
                        entry_point=entry_point,
                        backend_name=backend_name,
                        backend_adapter=backend_adapter,
                        build_dir=build_dir,
                        device_str=device_str,
                    )
                except (RuntimeError, BrokenPipeError) as exc:
                    return self._fail(
                        build_dir, entry_point, backend_name, code,
                        f"submit rejected: {exc}",
                    )
                self._submitted += 1
                self._dedup[build_dir] = future

        try:
            artifact = future.result(timeout=timeout)
        except TimeoutError:
            artifact = self._fail(
                build_dir, entry_point, backend_name, code,
                f"compile timeout after {timeout}s",
                elapsed=float(timeout or 0.0),
            )
        except BaseException as exc:  # noqa: BLE001 - cross-process safety.
            artifact = self._fail(
                build_dir, entry_point, backend_name, code,
                f"compile worker crashed: {type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                self._dedup.pop(build_dir, None)
                if isinstance(artifact, dict) and artifact.get("compiled"):
                    self._succeeded += 1
                else:
                    self._failed += 1

        return artifact

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None

        if executor is None:
            return

        try:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise.
            logger.warning("CompileOffloadPool shutdown error: %s", exc)
        logger.info(
            "CompileOffloadPool stopped (submitted=%d, succeeded=%d, failed=%d)",
            self._submitted, self._succeeded, self._failed,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "cuda_arch_list": self._cuda_arch_list,
                "submitted": self._submitted,
                "succeeded": self._succeeded,
                "failed": self._failed,
                "in_flight": len(self._dedup),
                "closed": self._closed,
            }

    # ------------------------------------------------------------ internals

    @staticmethod
    def _fail(
        build_dir: str,
        entry_point: str,
        backend_name: str,
        code: str,
        error: str,
        *,
        elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "compiled": False,
            "error": error,
            "build_dir": build_dir,
            "entry_point": entry_point,
            "backend": backend_name,
            "code": code,
            "elapsed_sec": elapsed,
        }
