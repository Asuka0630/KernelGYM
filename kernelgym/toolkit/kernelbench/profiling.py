"""KernelBench profiling helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

from kernelgym.config import settings

logger = logging.getLogger("kernelgym.toolkit.kernelbench.profiling")


def _safe_metric(evt: Any, names: Tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        if hasattr(evt, name):
            value = getattr(evt, name)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _safe_int_metric(evt: Any, names: Tuple[str, ...], default: int = 0) -> int:
    for name in names:
        if hasattr(evt, name):
            value = getattr(evt, name)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return default


def _is_memory_overhead_event(name: str) -> bool:
    """Return True for allocation/initialization/copy events.

    These events are real overhead and should remain visible in diagnostic
    profiling, but they should not dilute the Stage-2 anti-hack signal. The
    anti-hack check asks whether the submitted custom kernel ran meaningful
    device work, not whether output allocation or zero-fill was expensive.
    """
    lowered = name.lower()
    memory_tokens = (
        "cudamalloc",
        "cudafree",
        "cudaalloc",
        "cuda_free",
        "cudamemset",
        "cuda_memset",
        "cudamemcpy",
        "cuda_memcpy",
        "memset",
        "memcpy",
        "aten::empty",
        "aten::zeros",
        "aten::zero_",
        "aten::fill_",
        "aten::copy_",
        "allocation",
        "allocator",
    )
    return any(token in lowered for token in memory_tokens)


def _memory_stats() -> Dict[str, float]:
    """Collect CUDA memory stats without affecting profiling decisions."""
    memory_stats = {}
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_stats = {
                "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
            }
    except Exception as e:
        logger.warning(f"[Profiler] Failed to collect memory stats: {e}")
    return memory_stats


def compute_cuda_kernel_coverage(matched_cuda_kernels: List[str], profilling_result: Dict[str, Any]):
    """Compute custom-kernel coverage from profiler CUDA/device events.

    Stage-2 anti-hack uses this ratio as a device-time signal. Do not mix in
    CPU-side allocator or high-level operator aggregate time here; operations
    like ``aten::zeros`` can spend a long time allocating memory while the
    custom CUDA kernel is still the real device-side work.
    """

    def _matches_profiler_name(captured: str, profiler_name: str) -> bool:
        cap = captured.lower()
        prof = profiler_name.lower()
        if cap == prof:
            return True
        if cap in prof or prof in cap:
            return True
        return False

    kernels = matched_cuda_kernels
    num_custom_kernels = 0
    kernel_names = [kernel.split(" ")[0] for kernel in kernels]

    kernels_in_profiling = profilling_result["kernels"]
    use_device_events = any(
        bool(prof_kernel.get("is_cuda_device_event", False))
        for prof_kernel in kernels_in_profiling
    )
    use_self_cuda_time = any(
        float(prof_kernel.get("self_cuda_time_us", 0.0) or 0.0) > 0.0
        for prof_kernel in kernels_in_profiling
    )

    total_time = 0.0
    anti_hack_total_time = 0.0
    matched_cuda_time = 0.0
    num_total_kernels = 0
    anti_hack_num_total_kernels = 0
    cuda_kernels_in_profiling = []
    memory_overhead_kernels_in_profiling = []

    for prof_kernel in kernels_in_profiling:
        if use_device_events and not bool(prof_kernel.get("is_cuda_device_event", False)):
            continue

        prof_name = prof_kernel["name"]
        is_custom_kernel = any(
            _matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names
        )
        self_cuda_time = float(prof_kernel.get("self_cuda_time_us", 0.0) or 0.0)
        total_cuda_time = float(prof_kernel.get("cuda_time_us", 0.0) or 0.0)
        if use_self_cuda_time:
            cuda_time = self_cuda_time
            if cuda_time <= 0.0 and use_device_events:
                cuda_time = total_cuda_time
        else:
            cuda_time = total_cuda_time
        if cuda_time <= 0.0:
            continue

        total_time += cuda_time
        num_total_kernels += 1
        is_memory_overhead = _is_memory_overhead_event(prof_name)
        if is_custom_kernel or not is_memory_overhead:
            anti_hack_total_time += cuda_time
            anti_hack_num_total_kernels += 1
        elif is_memory_overhead:
            memory_overhead_kernels_in_profiling.append(prof_name)

        if is_custom_kernel:
            cuda_kernels_in_profiling.append(prof_name)
            num_custom_kernels += 1
            matched_cuda_time += cuda_time

    cuda_kernels_not_in_profiling = [
        kernel_name
        for kernel_name in kernel_names
        if not any(_matches_profiler_name(kernel_name, prof_name) for prof_name in cuda_kernels_in_profiling)
    ]

    return {
        "num_custom_kernels": num_custom_kernels,
        "num_total_kernels": num_total_kernels,
        "anti_hack_num_total_kernels": anti_hack_num_total_kernels,
        "total_kernel_run_time_in_profiling_us": total_time,
        "anti_hack_total_kernel_run_time_in_profiling_us": anti_hack_total_time,
        "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
        "memory_overhead_kernels_in_profiling": memory_overhead_kernels_in_profiling,
        "cuda_kernels_not_in_profiling": cuda_kernels_not_in_profiling,
        "cuda_kernels_in_profiling": cuda_kernels_in_profiling,
        "profiling_time_basis": "self_cuda_time_us" if use_self_cuda_time else "cuda_time_us",
    }


@contextmanager
def profiling_context(enabled: bool = True, light: bool = False):
    """Context manager that wraps ``torch.profiler.profile``.

    Args:
        enabled: When False, immediately yields None (no-op).
        light: When True, runs a lightweight config — CUDA activity only,
            no record_shapes / profile_memory / with_stack.  Used by the
            anti-hack Stage 2 profiler ratio check to minimise overhead.
    """
    if not enabled:
        yield None
        return

    try:
        import torch.profiler as profiler

        activities = []
        if "cpu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CPU)
        if "cuda" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CUDA)

        if light:
            # Light mode: CUDA-only, no shapes / memory / stack overhead
            activities = [profiler.ProfilerActivity.CUDA]
            _record_shapes = False
            _profile_memory = False
            _with_stack = False
        else:
            _record_shapes = settings.profiling_record_shapes
            _profile_memory = settings.profiling_profile_memory
            _with_stack = settings.profiling_with_stack

        mode_tag = "light" if light else "full"
        print(f"[Profiler] Initializing ({mode_tag}) with activities: {[str(a) for a in activities]}")

        if not activities:
            print("[Profiler] No activities configured, profiler will return no data")
            yield None
            return

        # Self-test CUDA before entering the profiler so the test's own
        # kernels (torch.ones / sum) are NOT captured and don't dilute the
        # custom-kernel time ratio used by anti-hack Stage 2.
        cuda_available = torch.cuda.is_available()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_info = "cuda:unavailable"
        if cuda_available:
            try:
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                device_info = f"cuda:{current_device} ({device_name})"
                test = torch.ones((1024,), device="cuda")
                _ = test.sum()
                torch.cuda.synchronize()
                print("[Profiler] Self-test CUDA op executed")
            except Exception as e:
                print(f"[Profiler] Self-test failed: {e}")
                device_info = f"cuda:unknown (error={e})"
        print(
            f"[Profiler] Context pid={os.getpid()} cuda_available={cuda_available}",
            f" device={device_info} CUDA_VISIBLE_DEVICES={cuda_visible}",
        )

        prof = profiler.profile(
            activities=activities,
            record_shapes=_record_shapes,
            profile_memory=_profile_memory,
            with_stack=_with_stack,
            on_trace_ready=None,
        )

        prof.__enter__()
        try:
            print("[Profiler] Profiler started successfully")
            yield prof
        finally:
            try:
                prof.__exit__(None, None, None)
                print("[Profiler] Profiler stopped successfully")
            except Exception as e:
                print(f"[Profiler] Error during profiler cleanup: {e}")

    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize profiler: {e}. Continuing without profiling.")
        yield None


def extract_profiling_metrics(prof: Optional["torch.profiler.profile"]) -> Dict[str, Any]:
    if prof is None:
        return {}

    try:
        import torch.profiler as profiler

        events = prof.key_averages()
        print(f"[Profiler] key_averages: {events}")
        total_events = len(events)
        cuda_device_event_count = 0
        cuda_time_event_count = 0
        self_cuda_time_event_count = 0

        logger.debug(f"[Profiler] Captured {total_events} total events")

        cuda_entries = []
        total_cpu_time = 0.0
        total_self_cuda_time = 0.0
        for evt in events:
            cpu_time_us = _safe_metric(evt, ("cpu_time_total", "cpu_time"), 0.0)
            total_cpu_time += cpu_time_us

            cuda_time_us = _safe_metric(
                evt,
                ("device_time_total", "device_time", "cuda_time_total", "cuda_time"),
                0.0,
            )
            self_cuda_time_us = _safe_metric(
                evt,
                ("self_device_time_total", "self_cuda_time_total", "self_cuda_time"),
                0.0,
            )
            if self_cuda_time_us > 0.0:
                self_cuda_time_event_count += 1
                total_self_cuda_time += self_cuda_time_us
            if cuda_time_us <= 0.0 and self_cuda_time_us <= 0.0:
                continue
            device_type = getattr(evt, "device_type", None)
            is_cuda_device_event = device_type == profiler.DeviceType.CUDA
            if is_cuda_device_event:
                cuda_device_event_count += 1
            cuda_time_event_count += 1

            kernel_entry = {
                "name": getattr(evt, "key", "unknown"),
                "cuda_time_us": cuda_time_us,
                "self_cuda_time_us": self_cuda_time_us,
                "cpu_time_us": cpu_time_us,
                "count": _safe_int_metric(evt, ("count",), 0),
                "is_cuda_device_event": is_cuda_device_event,
            }
            if device_type is not None:
                kernel_entry["device_type"] = str(device_type)
            memory_usage = _safe_metric(evt, ("cuda_memory_usage",), 0.0)
            if memory_usage > 0.0:
                kernel_entry["cuda_memory_usage"] = memory_usage
            cuda_entries.append(kernel_entry)

        cuda_device_entries = [
            entry for entry in cuda_entries if entry["is_cuda_device_event"]
        ]
        cuda_kernels = cuda_device_entries or cuda_entries
        cuda_kernels.sort(
            key=lambda x: max(
                float(x.get("self_cuda_time_us", 0.0)),
                float(x["cuda_time_us"]),
            ),
            reverse=True,
        )

        logger.debug(
            f"[Profiler] Filtered to {len(cuda_kernels)} CUDA kernels (from {len(events)} total)"
        )
        if len(cuda_kernels) == 0 and len(events) > 0:
            logger.warning(
                f"[Profiler] Captured events but no CUDA kernels! Event types: {[getattr(evt, 'device_type', 'unknown') for evt in list(events)[:5]]}"
            )

        profiling_metrics = {
            "kernels": cuda_kernels,
            "kernel_count": len(cuda_kernels),
            "total_cpu_time_us": total_cpu_time,
            "total_cuda_time_us": sum(k["cuda_time_us"] for k in cuda_kernels),
            "total_self_cuda_time_us": total_self_cuda_time,
            "cuda_device_event_count": cuda_device_event_count,
            "cuda_time_event_count": cuda_time_event_count,
            "self_cuda_time_event_count": self_cuda_time_event_count,
            "memory_stats": _memory_stats(),
        }

        if len(cuda_kernels) == 0:
            profiling_metrics["profiling_warning"] = (
                "Profiler captured no CUDA kernels. This may indicate a profiler failure."
            )

        return profiling_metrics

    except Exception as e:
        logger.warning(f"[Profiler] Failed to extract profiling metrics: {e}")
        return {"profiling_error": str(e)}
