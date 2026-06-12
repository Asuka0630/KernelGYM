# KernelBench/src/kernelbench/eval.py
"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)
from kernelgym.utils.traceback_utils import capture_runtime_error


def _record_phase_ms(metadata: dict, phase: str, elapsed_sec: float) -> None:
    """Append/aggregate ``phase`` wall-time (ms) to
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


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    """Record an exception into ``metadata`` as a *string*"""

    msg = str(exception_msg)
    if truncate and len(msg) > max_length:
        msg = msg[:max_length] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {msg} ")

    metadata[exception_type] = msg
    return metadata


class _ShapeMismatch(Exception):
    """Internal signal: ref/new output shape differs (not a runtime error)."""

    def __init__(self, ref_shape, new_shape):
        super().__init__(
            f"Output shape mismatch: Expected {ref_shape}, got {new_shape}"
        )
        self.ref_shape = ref_shape
        self.new_shape = new_shape


def _run_single_correctness_trial(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    trial_seed: int,
    device: Any,
    verbose: bool,
) -> tuple[bool, float | None, float | None]:
    """Run one correctness trial."""
    # Pre-bind names so the ``finally`` block below can ``del`` them unconditionally.
    inputs = None
    model = None
    model_new = None
    output = None
    output_new = None
    try:
        _t = time.perf_counter()
        set_seed(trial_seed)
        inputs = get_inputs_fn()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs
        ]
        _record_phase_ms(metadata, "correctness.input_setup", time.perf_counter() - _t)

        _t = time.perf_counter()
        set_seed(trial_seed)
        model = original_model_instance.cuda(device=device)
        set_seed(trial_seed)
        model_new = new_model_instance.cuda(device=device)
        _record_phase_ms(
            metadata, "correctness.model_to_device", time.perf_counter() - _t
        )

        if verbose:
            print(f"device: {device}")
            if inputs and isinstance(inputs[0], torch.Tensor):
                print(f"inputs: {inputs[0].device}")

        _t = time.perf_counter()
        output = model(*inputs)
        torch.cuda.synchronize(device=device)
        _record_phase_ms(metadata, "correctness.ref_forward", time.perf_counter() - _t)

        _t = time.perf_counter()
        output_new = model_new(*inputs)
        torch.cuda.synchronize(device=device)
        _record_phase_ms(metadata, "correctness.new_forward", time.perf_counter() - _t)

        # Free inputs and modules as early as possible; only the two outputs
        # are needed for the comparison below.
        del inputs
        inputs = None
        del model
        model = None
        del model_new
        model_new = None

        if output.shape != output_new.shape:
            raise _ShapeMismatch(output.shape, output_new.shape)

        _t = time.perf_counter()
        # Memory-efficient replacement for ``torch.allclose``.
        atol, rtol = 1e-02, 1e-02  # FP16/BF16
        output.sub_(output_new).abs_()  # output := |output - output_new|
        max_diff = output.max().item()
        avg_diff = output.mean().item()
        max_abs_b = output_new.abs_().max().item()
        is_close = max_diff <= atol + rtol * max_abs_b
        _record_phase_ms(metadata, "correctness.compare", time.perf_counter() - _t)

        return is_close, max_diff, avg_diff
    finally:
        # Drop local references to any GPU-resident objects on every exit
        # path (success / mismatch / runtime error).
        del inputs, model, model_new, output, output_new


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            try:
                is_close, max_diff, avg_diff = _run_single_correctness_trial(
                    original_model_instance=original_model_instance,
                    new_model_instance=new_model_instance,
                    get_inputs_fn=get_inputs_fn,
                    metadata=metadata,
                    trial_seed=trial_seed,
                    device=device,
                    verbose=verbose,
                )
            except _ShapeMismatch as e:
                err_msg = str(e)
                metadata = register_and_format_exception(
                    "correctness_issue", err_msg, metadata
                )
                metadata["correctness_issue_name"] = "correctness_issue"
                metadata["correctness_trials"] = (
                    f"({pass_count} / {num_correct_trials})"
                )
                if verbose:
                    print(f"[FAIL] check_correctness trial {trial}: {err_msg}")
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
            except Exception as e:
                err_msg = str(e)
                err_name = get_error_name(e)
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {err_msg}")
                metadata["runtime_error"] = capture_runtime_error(e)
                metadata["runtime_error_name"] = err_name
                metadata["correctness_trials"] = (
                    f"({pass_count} / {num_correct_trials})"
                )
                if verbose:
                    print(f"[FAIL] check_correctness trial {trial}: {err_msg}")
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

            metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
            metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
            if is_close:
                pass_count += 1
                continue
            metadata["correctness_issue"] = "Output mismatch"
            metadata["correctness_issue_name"] = "correctness_issue"
            if verbose:
                print(f"[FAIL] check_correctness trial {trial}: Output mismatch ")

    if verbose:
        print(f"[Eval] check_correctness pass: {pass_count}/{num_correct_trials}")

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    # Fold the per-trial max/avg difference lists into ``correctness_issue``
    max_diffs = metadata.pop("max_difference", None)
    avg_diffs = metadata.pop("avg_difference", None)
    if max_diffs and "correctness_issue" in metadata:
        parts = [f"max_difference=[{', '.join(max_diffs)}]"]
        if avg_diffs:
            parts.append(f"avg_difference=[{', '.join(avg_diffs)}]")
        metadata["correctness_issue"] = (
            f"{metadata['correctness_issue']}; {'; '.join(parts)}"
        )

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
