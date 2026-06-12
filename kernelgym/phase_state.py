"""Cross-stage phase reporter.

The toolkit pipeline runs several distinct stages inside the GPU subprocess
(compile, correctness, performance). When the outer worker pool kills the
subprocess on timeout, it needs to know **which stage was running** to
classify the timeout (COMPILATION_TIMEOUT vs RUNTIME_TIMEOUT).

This module provides a tiny global hook:

* The pipeline calls :func:`enter_phase` at the start of each stage.
* The subprocess wrapper calls :func:`set_reporter` once at startup,
  pointing at a function that writes the phase code to an
  ``mp.Value`` shared with the parent process.
"""

from __future__ import annotations

from typing import Callable, Optional


# Phase codes shared with subprocess_pool / gpu_worker.
PHASE_IDLE = 0
PHASE_COMPILE = 1
PHASE_CORRECTNESS = 2
PHASE_PERFORMANCE = 3

PHASE_NAMES = {
    PHASE_IDLE: "idle",
    PHASE_COMPILE: "compile",
    PHASE_CORRECTNESS: "correctness",
    PHASE_PERFORMANCE: "performance",
}

_NAME_TO_CODE = {v: k for k, v in PHASE_NAMES.items()}


_reporter: Optional[Callable[[int], None]] = None


def set_reporter(fn: Optional[Callable[[int], None]]) -> None:
    """Install a process-wide phase reporter (or clear with ``None``)."""
    global _reporter
    _reporter = fn


def enter_phase(name: str) -> None:
    """Mark the start of a new pipeline stage."""
    if _reporter is None:
        return
    code = _NAME_TO_CODE.get(name, PHASE_IDLE)
    try:
        _reporter(code)
    except Exception:
        # Phase reporting must never break evaluation.
        pass


def phase_name(code: int) -> str:
    return PHASE_NAMES.get(int(code), "unknown")
