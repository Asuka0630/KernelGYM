"""KernelBench timing helpers (toolkit layer)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    profiling_context,
)


def _record_phase_ms(
    metadata: Optional[Dict[str, Any]], phase: str, elapsed_sec: float
) -> None:
    """Record/aggregate ``phase`` wall-time (ms) under
    ``metadata['phase_timings_ms']``. No-op when metadata is None.
    """
    if metadata is None:
        return
    bucket = metadata.setdefault("phase_timings_ms", {})
    elapsed_ms = float(elapsed_sec) * 1000.0
    if phase in bucket:
        try:
            bucket[phase] = float(bucket[phase]) + elapsed_ms
        except (TypeError, ValueError):
            bucket[phase] = elapsed_ms
    else:
        bucket[phase] = elapsed_ms


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    _warmup_start = time.perf_counter()
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)
    _record_phase_ms(metadata, "performance.warmup", time.perf_counter() - _warmup_start)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    _trials_start = time.perf_counter()
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        torch.cuda.synchronize(device=device)

        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)
    _record_phase_ms(metadata, "performance.timing_trials", time.perf_counter() - _trials_start)

    profiling_metrics: Dict[str, Any] = {}
    if enable_profiling:
        _prof_start = time.perf_counter()
        try:
            torch.cuda.synchronize(device=device)

            num_profiling_trials = min(10, num_trials)
            print(
                f"[Profiling] Running {num_profiling_trials} additional iterations for profiling..."
            )

            with profiling_context(True) as prof:
                for _ in range(num_profiling_trials):
                    kernel_fn(*args)
                torch.cuda.synchronize(device=device)

            profiling_metrics = extract_profiling_metrics(prof)
            if profiling_metrics:
                print(
                    f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
                )
                print(
                    f"[Profiling] Total CUDA time: {profiling_metrics.get('total_cuda_time_us', 0):.2f} us"
                )

        except Exception as e:
            print(f"[Profiling] Warning: Profiling failed: {e}")
            profiling_metrics = {"profiling_error": str(e)}
        finally:
            _record_phase_ms(
                metadata, "performance.profiling_inline", time.perf_counter() - _prof_start
            )

    return elapsed_times, profiling_metrics


def run_profiling_only(
    kernel_fn: callable,
    *args,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> Dict[str, Any]:
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    profiling_metrics: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)
        print(f"[Profiling] Running {num_trials} iterations (profiling-only)...")
        with profiling_context(True) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            print(
                f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
            )
    except Exception as e:
        print(f"[Profiling] Warning: Profiling-only failed: {e}")
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def get_timing_stats(elapsed_times: List[float], device: torch.device = None) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats
