"""Traceback formatting utilities.

Server-side error capture is intentionally narrow:

* :func:`capture_compile_error` -- compile-time exceptions (PyTorch JIT
  / nvcc / ninja).  Returns the raw PyTorch ``RuntimeError`` message
  (which already contains the ninja+nvcc stdout/stderr) **as-is**, with
  the chained-exception preamble folded out and an actionable Python
  traceback appended.  No heuristic de-duplication of ninja's command
  echo is done here -- that is left to the client (stark) so the
  service contract stays "complete, lossless, faithful".

* :func:`capture_runtime_error` -- runtime exceptions raised by the
  user's kernel (CUDA error, illegal memory access, etc.).  Returns
  ``"<ExcType>: <msg>"`` plus an actionable traceback.

Both helpers rely on :func:`format_user_traceback` to trim KernelGYM's
own orchestration frames out of the traceback while keeping the
``<user_kernel>`` frame and the surrounding ``torch`` frames a user
would see when running the kernel standalone.
"""

from __future__ import annotations

import linecache
import sys
import traceback as _traceback
from typing import List

# ---------------------------------------------------------------------------
# Public: compile with linecache pre-population
# ---------------------------------------------------------------------------


def compile_with_source(code: str, filename: str, mode: str = "exec") -> compile:
    """Wrap :func:`compile` and pre-populate :mod:`linecache` for *filename*.

    When code is compiled with a virtual filename (``"<user_kernel>"``,
    ``"<string>"``, ``"<test>"``, ``"<reference>"``, etc.), Python's
    traceback machinery cannot read the source from disk, so
    :class:`traceback.FrameSummary` objects have ``line=None`` and the
    formatted traceback omits the source line.

    This helper injects the full source into :data:`linecache.cache` so
    every frame in the resulting traceback shows its source line, just as
    it would for a real file on disk.
    """
    linecache.cache[filename] = (
        len(code),
        None,
        [line + "\n" for line in code.splitlines()],
        filename,
    )
    return compile(code, filename, mode)


# ---------------------------------------------------------------------------
# Frame classification (used by :func:`format_user_traceback`)
# ---------------------------------------------------------------------------

# Filenames we consider "user code" whose frames are worth preserving.
_USER_FILENAMES = frozenset({"<user_kernel>", "<string>"})

# Library frames that are part of the user's observable execution path.  For
# example, a runtime CUDA failure raised by ``torch.cuda.synchronize()`` should
# show the ``torch.nn.Module`` call frames above ``ModelNew.forward`` and the
# ``torch.cuda`` frame below it, just like standalone execution.
_VISIBLE_LIBRARY_PATH_PARTS = (
    "/site-packages/torch/",
    "/site-packages/torch\\",
)

# Internal frames that should not leak into client-visible logs.
_INTERNAL_PATH_PARTS = (
    "/kernelgym/",
    "/3rdparty/KernelGYM/",
    "\\kernelgym\\",
    "\\3rdparty\\KernelGYM\\",
)


def _is_user_frame(frame: _traceback.FrameSummary) -> bool:
    return frame.filename in _USER_FILENAMES


def _is_visible_library_frame(frame: _traceback.FrameSummary) -> bool:
    return any(part in frame.filename for part in _VISIBLE_LIBRARY_PATH_PARTS)


def _is_internal_frame(frame: _traceback.FrameSummary) -> bool:
    return any(part in frame.filename for part in _INTERNAL_PATH_PARTS)


# ---------------------------------------------------------------------------
# Public: traceback formatter
# ---------------------------------------------------------------------------

def format_user_traceback() -> str:
    """Format the currently-handled exception's actionable traceback.

    KernelGYM catches exceptions several stack levels above user code.  A raw
    traceback therefore starts with scheduler/toolkit frames that are useless
    to the client.  This formatter starts at the first user frame, pulls in
    adjacent PyTorch frames that are part of the call path, and keeps the
    downstream frames where the error actually surfaced.

    Chained causes are intentionally omitted.  For compile failures PyTorch's
    ``RuntimeError`` message already contains the full ninja/nvcc diagnostic;
    formatting the ``CalledProcessError`` cause would add the redundant
    "The above exception..." preamble the client does not need.
    """
    _, exc_value, exc_tb = sys.exc_info()
    if exc_value is None:
        return ""

    all_frames: List[_traceback.FrameSummary] = list(_traceback.extract_tb(exc_tb))
    if not all_frames:
        return ""

    first_user_idx = next(
        (idx for idx, frame in enumerate(all_frames) if _is_user_frame(frame)),
        None,
    )
    if first_user_idx is None:
        first_visible_idx = next(
            (
                idx
                for idx, frame in enumerate(all_frames)
                if _is_visible_library_frame(frame) and not _is_internal_frame(frame)
            ),
            None,
        )
        if first_visible_idx is None:
            return ""
        start_idx = first_visible_idx
    else:
        start_idx = first_user_idx
        while start_idx > 0 and _is_visible_library_frame(all_frames[start_idx - 1]):
            start_idx -= 1

    relevant_frames = [
        frame for frame in all_frames[start_idx:] if not _is_internal_frame(frame)
    ]
    if not relevant_frames:
        return ""

    return "Traceback (most recent call last):\n" + "".join(
        _traceback.format_list(relevant_frames)
    )


# ---------------------------------------------------------------------------
# Public: the *only* two error-capture entry points.
# ---------------------------------------------------------------------------

# PyTorch raises ``RuntimeError("Error building extension '...': <ninja
# stdout/stderr>")`` from a chain that starts with a ``CalledProcessError``.
# When the surfaced ``RuntimeError`` is rendered with full chaining, Python
# prepends the inner traceback + this fixed preamble:
_CHAINED_PREAMBLE = (
    "\nThe above exception was the direct cause of the following exception:\n"
)

# Stable markers for "this string already contains the ninja/nvcc dump".
_NINJA_MARKERS = ("FAILED:", "ninja: build stopped", "ninja: error:")


def _fold_chained_preamble(message: str) -> str:
    """If ``message`` contains the ``CalledProcessError`` -> ``RuntimeError``
    chain preamble, drop everything before (and including) the preamble.

    PyTorch's ``_run_ninja_build`` constructs::

        raise RuntimeError(message) from e

    where ``e`` is a ``CalledProcessError``.  When some upstream caller
    formats this with chaining (e.g. via :func:`traceback.format_exception`),
    the resulting text contains the inner exception's full traceback followed
    by the fixed preamble below, then the outer ``RuntimeError`` and *its*
    traceback.  Folding away the preamble + everything before it keeps only
    the actionable RuntimeError block.
    """
    if not message:
        return message
    idx = message.find(_CHAINED_PREAMBLE)
    if idx == -1:
        return message
    return message[idx + len(_CHAINED_PREAMBLE):].lstrip("\n")


def capture_compile_error(
    exc: BaseException,
    *,
    captured_output: str = "",
) -> str:
    """Build the canonical compile-error string for client consumption.

    Output structure (order is stable -- the client may rely on it)::

        <main message>

        Compile error traceback:
        <format_user_traceback() result>

    The main message is computed as follows:

    1. Start from ``str(exc)``.
    2. If it does *not* already contain ninja's stable markers (``FAILED:``,
       ``ninja: build stopped``, ``ninja: error:``), prepend
       ``captured_output`` -- this handles AOT compile paths that catch
       fd-1/fd-2 output themselves before the exception bubbles up.
    3. Fold away PyTorch's chained-exception preamble so we never emit
       both the ``CalledProcessError`` traceback and the ``RuntimeError``
       traceback.
    4. If the result is still empty (e.g. a bare ``raise CustomError()``
       with no args, or a ``BaseException`` whose ``str()`` returns ""),
       fall back to ``"<ExcType>"`` / ``"<ExcType>: <str(exc)>"`` so the
       client never sees a completely empty diagnostic.
    """
    detail = str(exc) if exc is not None else ""

    if captured_output:
        captured = captured_output.rstrip()
        if captured and not any(m in detail for m in _NINJA_MARKERS):
            detail = captured + ("\n" + detail if detail else "")

    detail = _fold_chained_preamble(detail)

    # Fallback for exceptions whose ``str()`` is empty.  Without this an
    # ``except`` block that catches e.g. ``SystemExit()`` or a
    # ``Custom()`` exception with no args would surface as an empty
    # ``error_message`` and leave the LLM agent with no signal.
    if not detail.strip() and exc is not None:
        type_name = type(exc).__name__
        raw = str(exc)
        detail = f"{type_name}: {raw}" if raw else type_name

    tb = format_user_traceback()
    if tb.strip():
        joiner = "\n\n" if detail else ""
        return detail.rstrip() + joiner + "Compile error traceback:\n" + tb.strip()
    return detail


def capture_runtime_error(exc: BaseException) -> str:
    """Build the canonical runtime-error string for client consumption.

    Output structure (order is stable)::

        <ExcType>: <str(exc)>     (omits ': ' when str(exc) is empty)

        Runtime error traceback:
        <format_user_traceback() result>
    """
    if exc is None:
        head = ""
    else:
        type_name = type(exc).__name__
        raw = str(exc)
        head = f"{type_name}: {raw}" if raw else type_name

    tb = format_user_traceback()
    if tb.strip():
        joiner = "\n\n" if head else ""
        return head + joiner + "Runtime error traceback:\n" + tb.strip()
    return head
