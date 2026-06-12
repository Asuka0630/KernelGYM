# KernelBench/src/kernelbench/eval.py
"""KernelBench model loading helpers (toolkit layer)."""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from kernelgym.utils.traceback_utils import compile_with_source

logger = logging.getLogger(__name__)


def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point: str = "Model"
) -> Tuple[nn.Module, callable, callable]:
    try:
        compile_with_source(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None
    try:
        exec(model_original_src, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get(entry_point)

    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(
    model_custom_src: str, entry_point: str = "ModelNew"
):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(model_custom_src)
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    return ModelNew, temp_file


# ---------------------------------------------------------------------------
# Direct-dlopen helpers for precompiled extensions
# ---------------------------------------------------------------------------


def _find_precompiled_extensions(build_directory: str) -> Dict[str, Path]:
    """Scan ``build_directory`` for ``<name>/<name>.so`` pairs.

    PyTorch's ``_get_build_directory(name)`` creates a per-extension
    subdirectory under ``TORCH_EXTENSIONS_DIR``, and ``_get_exec_path``
    writes ``<name>.so`` (or ``<name>_vN.so``) inside it.

    Returns ``{extension_name: path_to_so}``.
    """
    root = Path(build_directory)
    if not root.is_dir():
        return {}
    found: Dict[str, Path] = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        # Versioned: ``name_v0.so``, ``name_v1.so``, ...
        versioned = sorted(
            subdir.glob(f"{name}_v[0-9]*{sysconfig_get_so_ext()}"),
            reverse=True,
        )
        if versioned:
            found[name] = versioned[0]
            continue
        # Unversioned: ``name.so``
        plain = subdir / f"{name}{sysconfig_get_so_ext()}"
        if plain.is_file():
            found[name] = plain
    return found


_sysconfig_so_ext: Optional[str] = None


def sysconfig_get_so_ext() -> str:
    """Cached ``sysconfig.get_config_var('EXT_SUFFIX')`` or ``.so``."""
    global _sysconfig_so_ext
    if _sysconfig_so_ext is None:
        import sysconfig

        _sysconfig_so_ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return _sysconfig_so_ext


def _preload_extension(so_path: Path) -> Any:
    """Import a compiled extension .so and return the module object."""
    name = so_path.parent.name  # directory name = extension name
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from {so_path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so subsequent lookups (including the
    # patched load_inline) find the same module object.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_preloaded_registry(build_directory: str) -> Dict[str, Any]:
    """Pre-load every compiled extension under ``build_directory``.

    Returns ``{extension_name: module}`` for all .so files found.
    """
    registry: Dict[str, Any] = {}
    for name, so_path in _find_precompiled_extensions(build_directory).items():
        try:
            registry[name] = _preload_extension(so_path)
            logger.info("Pre-loaded extension %s from %s", name, so_path)
        except Exception:
            logger.warning(
                "Failed to pre-load extension %s from %s", name, so_path, exc_info=True
            )
    return registry


@contextmanager
def _patch_load_inline_with_registry(registry: Dict[str, Any]):
    """Context manager: intercept ``load_inline(name=...)`` calls whose
    *name* matches a key in *registry*, returning the pre-loaded module
    immediately instead of invoking the real ``load_inline``.

    Calls with unrecognised names fall through to the original.
    """
    try:
        from torch.utils import cpp_extension as _cpp_ext
    except Exception:
        yield
        return

    _original = _cpp_ext.load_inline

    def _intercept(name="", *args, **kwargs):
        if name in registry:
            return registry[name]
        return _original(name=name, *args, **kwargs)

    _cpp_ext.load_inline = _intercept
    try:
        yield
    finally:
        _cpp_ext.load_inline = _original


@contextmanager
def _patch_load_inline_inject_cflags(extra_cflags: List[str]) -> Iterator[None]:
    """Context manager: inject *extra_cflags* into every
    ``torch.utils.cpp_extension.load_inline`` call inside the block.

    LLM-generated kernel code calls ``load_inline(...)`` directly, so we
    cannot pass ``extra_cuda_cflags`` from outside without rewriting that
    source. Patching for the duration of ``exec()`` is the only viable
    injection point under the PyTorch JIT path; the patch is local to
    this context and restored on exit.
    """
    try:
        from torch.utils import cpp_extension as _cpp_ext
    except Exception as e:  # noqa: BLE001
        logger.warning("inject-cflags patch: torch unavailable (%s); skipping", e)
        yield
        return

    original = _cpp_ext.load_inline

    def _patched(*args, **kwargs):
        flags = list(kwargs.get("extra_cuda_cflags") or [])
        for f in extra_cflags:
            if f not in flags:
                flags.append(f)
        kwargs["extra_cuda_cflags"] = flags
        return original(*args, **kwargs)

    _cpp_ext.load_inline = _patched
    try:
        yield
    finally:
        _cpp_ext.load_inline = original


@contextmanager
def _patch_load_inline_force_quiet() -> Iterator[None]:
    """Context manager: force ``verbose=False`` on every
    ``torch.utils.cpp_extension.load_inline`` call inside the block.
    """
    try:
        from torch.utils import cpp_extension as _cpp_ext
    except Exception as e:  # noqa: BLE001
        logger.warning("force-quiet patch: torch unavailable (%s); skipping", e)
        yield
        return

    original = _cpp_ext.load_inline

    def _patched(*args, **kwargs):
        # Silently coerce both the documented and the legacy spelling.
        kwargs["verbose"] = False
        return original(*args, **kwargs)

    _cpp_ext.load_inline = _patched
    try:
        yield
    finally:
        _cpp_ext.load_inline = original


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None,
    extra_cuda_cflags: Optional[List[str]] = None,
) -> nn.Module:
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    registry: Dict[str, Any] = (
        _build_preloaded_registry(build_directory) if build_directory else {}
    )

    # Compose the patches: registry interception (fast path) takes
    # precedence; cflag injection only matters when nvcc actually runs,
    # i.e. the registry is empty.  ``force_quiet`` is unconditional so
    # every nvcc invocation lands in PyTorch's ``subprocess.PIPE`` capture
    # path -- otherwise an LLM-emitted ``load_inline(verbose=True)`` would
    # bypass it and dump straight to ``fd 1``.
    patches = [_patch_load_inline_force_quiet()]
    if registry:
        patches.append(_patch_load_inline_with_registry(registry))
    elif extra_cuda_cflags:
        patches.append(_patch_load_inline_inject_cflags(list(extra_cuda_cflags)))

    try:
        compiled = compile_with_source(model_custom_src, "<user_kernel>", "exec")
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            exec(compiled, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    del curr_context
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)
