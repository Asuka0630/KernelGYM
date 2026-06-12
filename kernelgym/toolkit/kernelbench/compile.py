"""KernelBench compile helpers (CUDA cache build)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kernelgym.toolkit.kernelbench.loading import load_custom_model
from kernelgym.utils.traceback_utils import capture_compile_error


def build_compile_cache(
    custom_model_src: str,
    build_dir: str | None,
    verbose: bool = False,
    *,
    extra_cuda_cflags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Pre-compile a custom CUDA kernel.

    Returns a dict with ``compiled``/``error`` and (legacy) empty
    ``stdout``/``stderr`` slots.  On compile failure the ``error``
    field carries the canonical compile-error string produced by
    :func:`capture_compile_error` (already containing the full
    ninja+nvcc diagnostic plus a user-anchored Python traceback).
    """
    context: Dict[str, Any] = {}

    if verbose:
        print("[Compilation] Pre-compile custom CUDA binaries")

    try:
        if build_dir:
            custom_model_src = (
                "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
            ) + custom_model_src

        load_custom_model(
            custom_model_src,
            context,
            build_dir,
            extra_cuda_cflags=extra_cuda_cflags,
        )

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
        return {
            "compiled": True,
            "stdout": "",
            "stderr": "",
            "error": None,
        }
    except Exception as exc:
        if verbose:
            print(
                f"[Compilation] Failed to compile custom CUDA kernel. "
                f"Unable to cache, Error: {exc}"
            )
        return {
            "compiled": False,
            "stdout": "",
            "stderr": "",
            "error": capture_compile_error(exc),
        }
