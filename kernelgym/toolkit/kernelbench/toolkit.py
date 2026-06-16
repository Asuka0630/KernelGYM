"""KernelBench toolkit wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.utils.traceback_utils import capture_runtime_error
from kernelgym.schema import (
    EvaluationTask,
    EvaluationResult,
    KernelEvaluationResult,
    KernelEvaluationTask,
    ReferenceTimingResult,
    ReferenceTimingTask,
)
from kernelgym.toolkit.validation import validate_code
from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.kernelbench import pipeline as kernelbench_pipeline

from ..base import Toolkit


class KernelBenchToolkit(Toolkit):
    """Toolkit adapter around KernelBench evaluation."""

    name = "kernelbench"

    def __init__(self) -> None:
        pass

    def _resolve_eval_flags(self, task: Any) -> tuple[bool, bool, bool]:
        run_correctness = task.run_correctness
        if run_correctness is None:
            run_correctness = True

        run_triton_detection = task.run_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.enable_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.backend == "triton"

        run_performance = task.run_performance
        if run_performance is None:
            run_performance = task.measure_performance
        if run_performance is None:
            run_performance = True

        return run_correctness, run_triton_detection, run_performance

    def evaluate(self, task: Dict[str, Any], backend=None, **kwargs: Any) -> Dict[str, Any]:
        task_type = task.get("task_type", "evaluation")
        # The off-GPU compile pipeline (kernelgym.worker.compile_service)
        # attaches its stage-1 artifact + build_dir directly to the task
        # payload. Forward those into the typed helpers so the pipeline can
        # bypass backend.compile() entirely.
        precompiled_artifact = task.get("precompiled_artifact")
        attached_build_dir = task.get("build_dir")
        if task_type == "evaluation":
            result = self.evaluate_kernel(
                EvaluationTask.from_dict(task),
                backend_adapter=backend,
                precompiled_artifact=precompiled_artifact,
                attached_build_dir=attached_build_dir,
            )
        elif task_type == "reference_timing":
            result = self.evaluate_reference_timing(
                ReferenceTimingTask.from_dict(task),
                backend_adapter=backend,
            )
        elif task_type == "kernel_evaluation":
            result = self.evaluate_kernel_only(
                KernelEvaluationTask.from_dict(task),
                verbose_errors=task.get("verbose_errors", True),
                enable_profiling=task.get("enable_profiling", settings.enable_profiling),
                backend_adapter=backend,
                precompiled_artifact=precompiled_artifact,
                attached_build_dir=attached_build_dir,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return result.to_dict()

    def evaluate_kernel(
        self,
        task: EvaluationTask,
        backend_adapter=None,
        precompiled_artifact: Optional[Dict[str, Any]] = None,
        attached_build_dir: Optional[str] = None,
    ) -> EvaluationResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            run_correctness, enable_triton_detection, measure_performance = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0

            enable_profiling = task.enable_profiling
            if enable_profiling is None:
                enable_profiling = settings.enable_profiling

            result = kernelbench_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                num_warmup=task.num_warmup,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                enable_profiling=bool(enable_profiling),
                enable_triton_detection=enable_triton_detection,
                backend_adapter=backend_adapter,
                build_dir=attached_build_dir,
                precompiled_artifact=precompiled_artifact,
                enable_ncu=bool(task.enable_ncu),
                ncu_top_k_rules=int(
                    task.ncu_top_k_rules
                    if task.ncu_top_k_rules is not None
                    else settings.ncu_top_k_rules
                ),
                kernel_names=task.kernel_names,
                enable_anti_hack=(
                    bool(task.enable_anti_hack)
                    if task.enable_anti_hack is not None
                    else True
                ),
                anti_hack_ratio_min=(
                    float(task.anti_hack_ratio_min)
                    if task.anti_hack_ratio_min is not None
                    else 0.02
                ),
                anti_hack_profiling_trials=(
                    int(task.anti_hack_profiling_trials)
                    if task.anti_hack_profiling_trials is not None
                    else 3
                ),
            )

            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            reference_runtime = kernelbench_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                num_warmup=task.num_warmup,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                backend_adapter=backend_adapter,
            ).runtime

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                    "num_warmup": task.num_warmup,
                }
            )

            return EvaluationResult.from_kernel_exec_result(task.task_id, result, reference_runtime)

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            _meta_error = capture_runtime_error(e)
            _error_msg = f"Evaluation failed: {_meta_error}"
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"error": _meta_error},
                status="failed",
                error_message=_error_msg,
                error_code=error_code,
            )

    def evaluate_reference_timing(
        self, task: ReferenceTimingTask, backend_adapter=None
    ) -> ReferenceTimingResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            if task.reference_backend:
                print(
                    f"[RefTiming] task={task.task_id} reference_backend={task.reference_backend}"
                )

            ref_exec_result = kernelbench_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                num_warmup=task.num_warmup,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                backend_adapter=backend_adapter,
            )
            reference_runtime = ref_exec_result.runtime

            metadata = {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "backend": task.backend,
                "num_perf_trials": task.num_perf_trials,
                "num_warmup": task.num_warmup,
            }
            if ref_exec_result.metadata:
                metadata.update(ref_exec_result.metadata)

            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=reference_runtime,
                metadata=metadata,
                status="completed",
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Reference timing failed: {str(e)}",
                error_code=error_code,
            )

    def evaluate_kernel_only(
        self,
        task: KernelEvaluationTask,
        verbose_errors: bool = True,
        enable_profiling: bool = False,
        backend_adapter=None,
        precompiled_artifact: Optional[Dict[str, Any]] = None,
        attached_build_dir: Optional[str] = None,
    ) -> KernelEvaluationResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            run_correctness, enable_triton_detection, measure_performance = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0

            result = kernelbench_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                num_warmup=task.num_warmup,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                enable_profiling=enable_profiling,
                enable_triton_detection=enable_triton_detection,
                backend_adapter=backend_adapter,
                build_dir=attached_build_dir,
                precompiled_artifact=precompiled_artifact,
                enable_ncu=bool(task.enable_ncu),
                ncu_top_k_rules=int(
                    task.ncu_top_k_rules
                    if task.ncu_top_k_rules is not None
                    else settings.ncu_top_k_rules
                ),
                kernel_names=task.kernel_names,
                enable_anti_hack=(
                    bool(task.enable_anti_hack)
                    if task.enable_anti_hack is not None
                    else True
                ),
                anti_hack_ratio_min=(
                    float(task.anti_hack_ratio_min)
                    if task.anti_hack_ratio_min is not None
                    else 0.02
                ),
                anti_hack_profiling_trials=(
                    int(task.anti_hack_profiling_trials)
                    if task.anti_hack_profiling_trials is not None
                    else 3
                ),
            )

            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                    "num_warmup": task.num_warmup,
                }
            )

            if enable_profiling and "profiling" in result.metadata:
                profiling_metrics = result.metadata["profiling"]
                if profiling_metrics:
                    print(
                        f"[DEBUG] Profiling captured {profiling_metrics.get('kernel_count', 0)} kernels"
                    )

            return KernelEvaluationResult.from_kernel_exec_result(
                task.task_id, task.base_task_id, result
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            _meta_error = capture_runtime_error(e)
            _error_msg = f"Kernel evaluation failed: {_meta_error}"
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"error": _meta_error},
                status="failed",
                error_message=_error_msg,
                error_code=error_code,
            )
