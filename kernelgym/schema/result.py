"""Shared result models for KernelBench workflows."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from kernelgym.common import ErrorCode
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

from .serialization import coerce_error_code, make_json_safe, serialize_error_code


def _filter_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in valid_fields}


@dataclass
class ReferenceTimingResult:
    task_id: str
    base_task_id: str
    reference_runtime: float
    metadata: Dict[str, Any]
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get("metadata"):
            result["metadata"] = make_json_safe(result["metadata"])
        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTimingResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)


@dataclass
class KernelEvaluationResult:
    task_id: str
    base_task_id: str
    compiled: Optional[bool]
    correctness: Optional[bool]
    decoy_kernel: bool
    kernel_runtime: float
    metadata: Dict[str, Any]
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get("metadata"):
            result["metadata"] = make_json_safe(result["metadata"])
        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelEvaluationResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)

    @classmethod
    def from_kernel_exec_result(
        cls,
        task_id: str,
        base_task_id: str,
        result: KernelExecResult,
    ) -> "KernelEvaluationResult":
        metadata: Dict[str, Any] = dict(result.metadata or {})

        # All upstream call sites now pass strings into ``metadata``
        # (see kernelgym.utils.traceback_utils.capture_compile_error /
        # capture_runtime_error).  Defensive coercion only handles the
        # rare case where a non-string slipped through.
        for key in (
            "compilation_error",
            "runtime_error",
            "error",
            "correctness_issue",
            "error_during_performance",
            "triton_kernel_coverage",
            "num_custom_kernels",
            "num_total_kernels",
            "triton_profiler_matches",
            "custom_kernel_cuda_time_in_profiling_us",
            "total_kernel_run_time_in_profiling_us",
            "custom_kernel_cuda_time_coverage",
        ):
            value = metadata.get(key)
            if value is None or isinstance(value, (str, int, float, bool)):
                continue
            metadata[key] = str(value)

        error_message: Optional[str] = None
        error_code: Optional[ErrorCode] = None

        if not result.compiled:
            detail = (
                metadata.get("compilation_error")
                or metadata.get("error")
                or metadata.get("validation_error")
            )
            if detail:
                error_message = f"Kernel compilation failed: {detail}"
            else:
                error_message = "Kernel compilation failed"
            # Distinguish "nvcc/build wedged past the budget" (compile
            # offload pool returned with a TimeoutError) from "code is
            # syntactically broken" so downstream agents can react
            # appropriately.
            detail_str = str(detail).lower() if detail else ""
            if "timeout" in detail_str or "timed out" in detail_str:
                error_code = ErrorCode.COMPILATION_TIMEOUT
            else:
                error_code = ErrorCode.COMPILATION_ERROR
        elif not result.correctness:
            # Distinguish "kernel raised at runtime" from "kernel returned
            # numerically wrong output". The former carries an exception
            # detail in ``runtime_error`` / ``error``; the latter carries
            # the diff summary in ``correctness_issue`` (e.g.
            # "Output mismatch; max_difference=[...]"). Surface whichever
            # one is populated so the LLM agent gets the actionable
            # detail directly from ``error_message`` without having to
            # reach into ``metadata``.
            runtime_detail = metadata.get("runtime_error") or metadata.get("error")
            correctness_detail = metadata.get("correctness_issue")
            if runtime_detail:
                error_message = f"Kernel execution failed: {runtime_detail}"
                error_code = ErrorCode.RUNTIME_ERROR
            elif correctness_detail:
                error_message = f"Kernel produced incorrect results: {correctness_detail}"
                error_code = ErrorCode.CORRECTNESS_ERROR
            else:
                error_message = "Kernel produced incorrect results"
                error_code = ErrorCode.CORRECTNESS_ERROR

        return cls(
            task_id=task_id,
            base_task_id=base_task_id,
            compiled=result.compiled,
            decoy_kernel=result.decoy_kernel,
            # When the kernel never compiled, correctness was never even
            # attempted -- surface that as None (unknown), not False, so
            # the downstream client can distinguish "skipped" from "failed".
            correctness=result.correctness if result.compiled else None,
            kernel_runtime=result.runtime,
            metadata=metadata,
            status="completed" if result.compiled else "failed",
            error_message=error_message,
            error_code=error_code,
        )


@dataclass
class EvaluationResult:
    task_id: str
    compiled: Optional[bool]
    correctness: Optional[bool]
    decoy_kernel: bool
    reference_runtime: float
    kernel_runtime: float
    speedup: float
    metadata: Dict[str, Any]
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get("metadata"):
            result["metadata"] = make_json_safe(result["metadata"])
        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)

    @classmethod
    def from_kernel_exec_result(
        cls, task_id: str, result: KernelExecResult, reference_runtime: float
    ) -> "EvaluationResult":
        speedup = 0.0
        if result.correctness and result.runtime > 0 and reference_runtime > 0:
            speedup = reference_runtime / result.runtime

        return cls(
            task_id=task_id,
            compiled=result.compiled,
            correctness=result.correctness,
            decoy_kernel=result.decoy_kernel,
            reference_runtime=reference_runtime,
            kernel_runtime=result.runtime,
            speedup=speedup,
            metadata=result.metadata,
            status="completed" if result.compiled else "failed",
        )

    @classmethod
    def from_paired_results(
        cls, base_task_id: str, reference_result: ReferenceTimingResult, kernel_result: KernelEvaluationResult
    ) -> "EvaluationResult":
        speedup = 0.0
        if (
            kernel_result.correctness
            and kernel_result.kernel_runtime > 0
            and reference_result.reference_runtime > 0
        ):
            speedup = reference_result.reference_runtime / kernel_result.kernel_runtime

        combined_metadata: Dict[str, Any] = {}
        combined_metadata.update(reference_result.metadata or {})
        combined_metadata.update(kernel_result.metadata or {})
        combined_metadata["reference_task_id"] = reference_result.task_id
        combined_metadata["kernel_task_id"] = kernel_result.task_id

        status = "completed"
        error_message = None
        error_code = None

        if reference_result.status != "completed":
            status = "failed"
            error_message = f"Reference timing failed: {reference_result.error_message}"
            error_code = reference_result.error_code
        elif kernel_result.status != "completed":
            status = "failed"
            error_message = f"Kernel evaluation failed: {kernel_result.error_message}"
            error_code = kernel_result.error_code

        return cls(
            task_id=base_task_id,
            compiled=kernel_result.compiled,
            correctness=kernel_result.correctness,
            decoy_kernel=kernel_result.decoy_kernel,
            reference_runtime=reference_result.reference_runtime,
            kernel_runtime=kernel_result.kernel_runtime,
            speedup=speedup,
            metadata=combined_metadata,
            status=status,
            error_message=error_message,
            error_code=error_code,
        )
