"""KernelBench compile helpers (CUDA cache build)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

from kernelgym.toolkit.kernelbench.loading import load_custom_model


def _drain_pipe(r_fd: int) -> str:
    """Non-blocking read of an OS pipe fd (does NOT close the fd)."""
    import fcntl
    fl = fcntl.fcntl(r_fd, fcntl.F_GETFL)
    fcntl.fcntl(r_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    parts = []
    while True:
        try:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            parts.append(chunk)
        except (BlockingIOError, OSError):
            break
    return b"".join(parts).decode("utf-8", errors="replace")


def build_compile_cache(custom_model_src: str, build_dir: str | None, verbose: bool = False) -> Dict[str, Any]:
    """Pre-compile a custom CUDA kernel and capture its nvcc output.

    Uses OS-level fd redirection so that ``load_inline(verbose=True)``,
    which calls ``subprocess.run(stdout=1)``, has its nvcc diagnostics
    captured (Python-level ``redirect_stdout`` cannot intercept fd 1 writes).
    """
    context: Dict[str, Any] = {}

    if verbose:
        print("[Compilation] Pre-compile custom CUDA binaries")

    # Redirect fd 1+2 at the OS level so nvcc output from
    # ``load_inline(verbose=True)`` is captured.
    old_1, old_2 = os.dup(1), os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 1)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    try:
        try:
            if build_dir:
                custom_model_src = (
                    "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
                ) + custom_model_src

            load_custom_model(custom_model_src, context, build_dir)

            if verbose:
                print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
            return {
                "compiled": True,
                "stdout": _drain_pipe(r_fd),
                "stderr": "",
                "error": None,
            }
        except Exception as exc:
            compile_output = _drain_pipe(r_fd)
            if verbose:
                print(
                    f"[Compilation] Failed to compile custom CUDA kernel. "
                    f"Unable to cache, Error: {exc}"
                )
            # Prepend raw nvcc output so the diagnostic is visible to clients.
            error_msg = str(exc)
            if compile_output.strip():
                error_msg = compile_output.rstrip() + "\n" + error_msg
            return {
                "compiled": False,
                "stdout": compile_output,
                "stderr": "",
                "error": error_msg,
            }
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_1, 1)
        os.dup2(old_2, 2)
        os.close(old_1)
        os.close(old_2)
        # Only close r_fd *after* fds 1+2 are restored to their original
        # targets.  Closing earlier would leave those fds pointing to a
        # pipe whose read-end is gone — any stray write then hits EPIPE.
        try:
            os.close(r_fd)
        except OSError:
            pass
