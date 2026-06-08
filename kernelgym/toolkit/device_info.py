"""Target GPU device info collection.

Returns a structured dict (matching the ``DeviceInfoResponse`` API model)
that summarises the GPU's specs for downstream LLM agents. Combines:

  * ``torch.cuda.get_device_properties()`` — covers SM count, shared
    memory, registers, total memory, L2 cache, compute capability, warp
    size, max threads per multiprocessor.
  * CUDA driver API (cuda-python preferred; ctypes + libcuda fallback)
    for the launch limits that PyTorch does NOT expose:
    ``max_threads_per_block``, ``max_block_dim``, ``max_grid_dim``,
    ``max_blocks_per_multiprocessor``.
  * Hard-coded ``GPU_LOOKUP`` table for vendor-published peak DRAM
    bandwidth and tensor-core / FMA peak TFLOPS — these are NOT exposed
    anywhere in CUDA's runtime API.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kernelgym.toolkit.device_info")

GPU_LOOKUP: Dict[str, Dict[str, Optional[float]]] = {
    # Volta
    "Tesla V100-PCIE-32GB": {
        "dram_bandwidth_gb_s": 900.0,
        "f32_cuda_core": 14.0,
        "f32_tensor_core": None,
        "f16_cuda_core": 28.0,
        "f16_tensor_core": 112.0,
    },
    # Ampere
    "NVIDIA A100-SXM4-80GB": {
        "dram_bandwidth_gb_s": 2039.0,
        "f32_cuda_core": 19.5,
        "f32_tensor_core": 156.0,
        "f16_cuda_core": 78.0,
        "f16_tensor_core": 312.0,
    },
    # Hopper
    "NVIDIA H100 PCIe": {
        "dram_bandwidth_gb_s": 2000.0,
        "f32_cuda_core": 51.0,
        "f32_tensor_core": 756.0,
        "f16_cuda_core": 102.0,
        "f16_tensor_core": 1513.0,
    },
}


def _lookup_peaks(gpu_name: str) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    """Return (peaks_dict, warning_or_None) given a torch device name.

    Match strategy:
      1. Exact match against ``GPU_LOOKUP`` keys.
      2. Substring match (helps for vendor-prefix variations like
         "NVIDIA A100-SXM4-40GB" vs "A100-SXM4-40GB").
      3. Fallback: all peaks None + warning string.
    """
    if gpu_name in GPU_LOOKUP:
        return GPU_LOOKUP[gpu_name], None

    candidates: List[Tuple[int, str]] = []
    for key in GPU_LOOKUP:
        if key in gpu_name or gpu_name in key:
            candidates.append((len(key), key))
    if candidates:
        candidates.sort(reverse=True)
        matched = candidates[0][1]
        return (
            GPU_LOOKUP[matched],
            f"GPU '{gpu_name}' not in GPU_LOOKUP table; using '{matched}' as approximate match",
        )

    return (
        {
            "dram_bandwidth_gb_s": None,
            "f32_cuda_core": None,
            "f32_tensor_core": None,
            "f16_cuda_core": None,
            "f16_tensor_core": None,
        },
        f"GPU '{gpu_name}' not in GPU_LOOKUP table; metrics are unavailable.",
    )


def collect_device_info(
    device_index: int = 0, include_raw: bool = False
) -> Dict[str, Any]:
    """Collect target-GPU specs into a JSON-friendly dict.

    Schema is identical to the final ``DeviceInfoResponse`` so the output
    can be unit-tested as a fixture.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot probe device info")

    if device_index >= torch.cuda.device_count():
        raise IndexError(
            f"device_index={device_index} out of range "
            f"(visible devices: {torch.cuda.device_count()})"
        )

    props = torch.cuda.get_device_properties(device_index)
    gpu_name = props.name
    peaks, warning = _lookup_peaks(gpu_name)

    warp_size = getattr(props, "warp_size", 32)
    max_warps_per_sm = (
        props.max_threads_per_multi_processor // warp_size if warp_size else None
    )
    # ``shared_memory_per_block_optin`` (the "opt-in" max via
    # cudaFuncSetAttribute) is what kernel authors care about — it's the
    # actual budget when `__shared__` allocations exceed the default 48KB
    # static cap. Fall back to the static value if the optin field is
    # missing.
    shared_mem_per_block = int(
        getattr(props, "shared_memory_per_block_optin", None)
        or props.shared_memory_per_block
    )

    # ``torch.cuda.get_device_properties`` does NOT expose
    # max_threads_per_block, max_block_dim, max_grid_dim, or
    # max_blocks_per_multi_processor on most builds. Probe these via the
    # CUDA driver API. Returns None for any field that can't be fetched.
    drv = _query_driver_attrs(device_index)

    info: Dict[str, Any] = {
        "name": gpu_name,
        "compute_capability": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "memory": {
            "total_gb": round(int(props.total_memory) / (1024**3), 2),
            "dram_bandwidth_gb_s": peaks["dram_bandwidth_gb_s"],
            "l2_cache_mb": round(
                int(getattr(props, "L2_cache_size", 0) or 0) / (1024**2), 2
            )
            or None,
        },
        "per_sm": {
            "shared_memory_kb": round(shared_mem_per_block / 1024, 2),
            "registers": int(props.regs_per_multiprocessor),
            "max_warps": max_warps_per_sm,
            "max_blocks": drv.get("max_blocks_per_sm"),
        },
        "peak_tflops": {
            "f32_cuda_core": peaks["f32_cuda_core"],
            "f32_tensor_core": peaks["f32_tensor_core"],
            "f16_cuda_core": peaks["f16_cuda_core"],
            "f16_tensor_core": peaks["f16_tensor_core"],
        },
        "limits": {
            "max_threads_per_block": drv.get("max_threads_per_block"),
            "max_block_dim": drv.get("max_block_dim"),
            "max_grid_dim": drv.get("max_grid_dim"),
            "warp_size": int(warp_size),
        },
        "warning": warning,
    }

    if include_raw:
        info["_raw_props"] = {
            k: getattr(props, k)
            for k in dir(props)
            if not k.startswith("_") and not callable(getattr(props, k, None))
        }
        # Make the raw props JSON-friendly.
        info["_raw_props"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in info["_raw_props"].items()
        }

    return info


@lru_cache(maxsize=4)
def get_device_info_cached(device_index: int = 0, include_raw: bool = False) -> Dict[str, Any]:
    """LRU-cached wrapper around :func:`collect_device_info`."""
    return collect_device_info(device_index=device_index, include_raw=include_raw)


def _query_driver_attrs(device_index: int) -> Dict[str, Any]:
    """Best-effort fetch of CUDA driver-level device attributes.

    Returns a dict with optional keys (each may be None on failure):
        max_threads_per_block, max_block_dim ([x,y,z]),
        max_grid_dim ([x,y,z]), max_blocks_per_sm

    torch.cuda.get_device_properties does NOT expose these on most
    builds. We try ``cuda-python`` first (cudart), then fall back to
    ``ctypes`` + libcuda via cuDeviceGetAttribute. Returns an empty dict
    if neither path works.
    """
    out: Dict[str, Any] = {
        "max_threads_per_block": None,
        "max_block_dim": None,
        "max_grid_dim": None,
        "max_blocks_per_sm": None,
    }

    # ----- Path 1: cuda-python (lazy import) -----
    try:
        from cuda import cudart  # type: ignore[import-not-found]

        def _attr(idx: int, attr: int) -> Optional[int]:
            err, val = cudart.cudaDeviceGetAttribute(attr, idx)
            if err == cudart.cudaError_t.cudaSuccess:
                return int(val)
            return None

        # cudaDevAttr enum values, see cuda_runtime_api.h
        bx = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimX)
        by = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimY)
        bz = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimZ)
        gx = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimX)
        gy = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimY)
        gz = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimZ)
        mt = _attr(device_index, cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock)
        # cudaDevAttrMaxBlocksPerMultiprocessor = 106 (CUDA >= 11.0)
        mb = _attr(
            device_index,
            getattr(
                cudart.cudaDeviceAttr,
                "cudaDevAttrMaxBlocksPerMultiprocessor",
                106,
            ),
        )

        if all(v is not None for v in (bx, by, bz)):
            out["max_block_dim"] = [bx, by, bz]
        if all(v is not None for v in (gx, gy, gz)):
            out["max_grid_dim"] = [gx, gy, gz]
        out["max_threads_per_block"] = mt
        out["max_blocks_per_sm"] = mb
        return out
    except Exception:
        pass

    # ----- Path 2: ctypes + libcuda -----
    try:
        import ctypes
        import ctypes.util

        libcuda_path = ctypes.util.find_library("cuda")
        if not libcuda_path:
            return out
        libcuda = ctypes.CDLL(libcuda_path)

        # CUdevice_attribute enum values from cuda.h:
        #   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    = 1
        #   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          = 2
        #   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          = 3
        #   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          = 4
        #   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           = 5
        #   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           = 6
        #   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           = 7
        #   CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106
        attrs = {
            "max_threads_per_block": 1,
            "block_x": 2,
            "block_y": 3,
            "block_z": 4,
            "grid_x": 5,
            "grid_y": 6,
            "grid_z": 7,
            "max_blocks_per_sm": 106,
        }

        # cuInit may already have been called by torch; calling it again
        # is a no-op per CUDA docs.
        libcuda.cuInit(0)
        cu_device = ctypes.c_int(0)
        rc = libcuda.cuDeviceGet(ctypes.byref(cu_device), ctypes.c_int(device_index))
        if rc != 0:
            return out
        vals: Dict[str, Optional[int]] = {}
        for k, v in attrs.items():
            val = ctypes.c_int(0)
            rc = libcuda.cuDeviceGetAttribute(
                ctypes.byref(val), ctypes.c_int(v), cu_device
            )
            vals[k] = int(val.value) if rc == 0 else None
        out["max_threads_per_block"] = vals.get("max_threads_per_block")
        if all(vals.get(k) is not None for k in ("block_x", "block_y", "block_z")):
            out["max_block_dim"] = [vals["block_x"], vals["block_y"], vals["block_z"]]
        if all(vals.get(k) is not None for k in ("grid_x", "grid_y", "grid_z")):
            out["max_grid_dim"] = [vals["grid_x"], vals["grid_y"], vals["grid_z"]]
        out["max_blocks_per_sm"] = vals.get("max_blocks_per_sm")
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("driver-attr fallback failed: %s", e)
        return out


