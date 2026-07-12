"""Parse a .ncu-rep file into a compact LLM-friendly summary.

Produces a dict suitable for ``metadata.ncu``:

    {
      "kernels": [
        {
          "name": "...",
          "key_metrics": {"<metric>": <value or None>, ...},
          "top_rules":  [{"est_speedup_pct": ..., "severity": ..., "message": "..."}, ...]
        },
        ...
      ],
      "report_path": "...",
      "ncu_warning": "..." | None,
    }

Two key compatibility decisions, made after probing the dev box (Nsight
Compute 2025.1, sm_90 H100 PCIe):

  * Per-action rule extraction via ``action.rule_results_as_dicts()``
    is GONE in ncu_report 2025.1+. We therefore pull rule text from the
    ``ncu --import <report> --page details`` CLI output and parse the
    ``OPT|INF|WRN  Est. Local Speedup: X%`` markers.
  * Metric names differ across sm versions (sm_70..sm_100). We use a
    curated cross-arch set + ``metric_or_none`` fallback so a missing
    metric returns None rather than crashing.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kernelgym.toolkit.kernelbench.ncu_summary")


# ---------------------------------------------------------------------------
# Curated cross-architecture metric set.
#
# Picked from ncu-report-skill's B200_KEY_METRICS, trimmed to ~30 names
# that have meaningful values on sm_70 / sm_80 / sm_90 / sm_100. For each
# logical metric we may try multiple names; ``metric_or_none`` returns
# the first that resolves.
# ---------------------------------------------------------------------------
KEY_METRICS: Dict[str, List[str]] = {
    # --- launch geometry ---
    "launch.grid_size": ["launch__grid_size"],
    "launch.block_size": ["launch__block_size"],
    "launch.registers_per_thread": ["launch__registers_per_thread"],
    "launch.shared_mem_per_block": ["launch__shared_mem_per_block"],
    "launch.waves_per_sm": ["launch__waves_per_multiprocessor"],
    # --- timing ---
    "gpu.time_us": ["gpu__time_duration.sum"],
    # --- SOL (% of peak sustained) ---
    "sol.sm_throughput_pct": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "sol.dram_throughput_pct": [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    ],
    "sol.l1_throughput_pct": [
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
    ],
    # --- occupancy ---
    "occupancy.warps_active_pct": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    "occupancy.theoretical_pct": [
        "sm__maximum_warps_per_active_cycle_pct",
    ],
    # --- IPC / scheduler ---
    "ipc.executed": ["sm__inst_executed.avg.per_cycle_active"],
    "ipc.issue_active_pct": [
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
    ],
    # --- compute pipes ---
    "pipe.fma_pct": [
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    ],
    "pipe.alu_pct": [
        "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    ],
    "pipe.lsu_pct": [
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    ],
    "pipe.tensor_pct": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    # --- DRAM ---
    "dram.bytes_read": ["dram__bytes_read.sum"],
    "dram.bytes_write": ["dram__bytes_write.sum"],
    # --- caches ---
    "cache.l1_hit_pct": ["l1tex__t_sector_hit_rate.pct"],
    "cache.l2_hit_pct": ["lts__t_sector_hit_rate.pct"],
    # --- top stall reasons (averages, not per-PC) ---
    "stall.long_scoreboard": [
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    ],
    "stall.short_scoreboard": [
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    ],
    "stall.wait": [
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    ],
    "stall.barrier": [
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    ],
    "stall.mio_throttle": [
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    ],
    "stall.lg_throttle": [
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    ],
    "stall.math_throttle": [
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    ],
    "stall.no_instruction": [
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    ],
    "stall.not_selected": [
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    ],
}


# ---------------------------------------------------------------------------
# Tiny ncu_report helpers (forked from ncu-report-skill/helpers/ncu_utils.py
# but simplified to remove dependencies and dead code).
# ---------------------------------------------------------------------------
def _safe_value(action: Any, name: str) -> Optional[float]:
    try:
        return action[name].value()
    except Exception:
        return None


def _metric_or_none(action: Any, candidates: List[str]) -> Optional[float]:
    for n in candidates:
        v = _safe_value(action, n)
        if v is not None:
            return v
    return None


def _key_metrics_for_action(action: Any) -> Dict[str, Optional[float]]:
    return {
        logical: _metric_or_none(action, names)
        for logical, names in KEY_METRICS.items()
    }


# ---------------------------------------------------------------------------
# Rule extraction via ncu --page details CLI.
# ncu_report 2025.1+ no longer exposes rule_results_as_dicts(), so we
# fall back to text parsing.
# ---------------------------------------------------------------------------
_RULE_HEADER_RE = re.compile(
    r"^\s*(OPT|INF|WRN)\b(?:\s+Est\.\s+Local\s+Speedup:\s+([\d.]+)%)?\s*(.*)$",
    re.MULTILINE,
)
_KERNEL_HEADER_RE = re.compile(
    # e.g. "  probe_add_kernel(const float *, const float *, ...) (1024, 1, 1)x(256, 1, 1), Context 1, ..."
    r"^\s*(?P<name>\w+)\s*\([^)]*\)\s*\(\s*\d+",
    re.MULTILINE,
)


def _rules_from_details_cli(
    report_path: Path,
    ncu_binary: str,
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse rules per-kernel from ``ncu --import <report> --page details``.

    Returns ``{kernel_name: [rule_dict, ...]}`` (sorted desc by est_speedup,
    truncated to ``top_k`` per kernel). Empty dict on failure.
    """
    try:
        proc = subprocess.run(
            [ncu_binary, "--import", str(report_path), "--page", "details"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ncu --page details failed: %s", e)
        return {}

    if proc.returncode != 0:
        logger.warning("ncu --page details returncode=%d", proc.returncode)
        return {}

    text = proc.stdout
    if not text:
        return {}

    # Split by kernel header occurrences. Each section runs from one
    # kernel header to the next (or end of text).
    headers: List[Tuple[int, str]] = []
    for m in _KERNEL_HEADER_RE.finditer(text):
        headers.append((m.start(), m.group("name")))

    if not headers:
        # Fallback: lump all rules into a "<unknown>" bucket so the
        # caller still gets something.
        return {"<unknown>": _parse_rules_in_segment(text, top_k)}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, (start, name) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        seg = text[start:end]
        out.setdefault(name, []).extend(_parse_rules_in_segment(seg, top_k))
    # Truncate after merging duplicates.
    for k, lst in out.items():
        lst.sort(key=lambda r: -float(r.get("est_speedup_pct") or 0.0))
        out[k] = lst[:top_k]
    return out


def _parse_rules_in_segment(text: str, top_k: int) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    rules: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = _RULE_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        severity = m.group(1)
        est_str = m.group(2)
        first_msg = m.group(3).strip()

        # Collect indented continuation lines until a blank line, new
        # section header, or another rule header.
        msg_parts: List[str] = []
        if first_msg:
            msg_parts.append(first_msg)
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            stripped = nxt.strip()
            if not stripped:
                break
            if _RULE_HEADER_RE.match(nxt):
                break
            if stripped.startswith("Section:") or stripped.startswith("---"):
                break
            msg_parts.append(stripped)
            j += 1

        try:
            est = float(est_str) if est_str else 0.0
        except ValueError:
            est = 0.0
        rules.append({
            "est_speedup_pct": est,
            "severity": severity,
            "message": " ".join(msg_parts)[:600],
        })
        i = j

    rules.sort(key=lambda r: -float(r.get("est_speedup_pct") or 0.0))
    return rules[:top_k]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def summarize_report(
    report_path: str,
    *,
    top_k_rules: int = 5,
    ncu_binary: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a .ncu-rep file into the ``metadata.ncu`` summary dict.

    On any unrecoverable failure (e.g. ncu_report not importable, file
    missing, parse error), returns a dict with ``ncu_warning`` set and
    ``kernels=[]`` so the caller can still surface the failure to the
    LLM without exception unwinding the eval pipeline.
    """
    out: Dict[str, Any] = {
        "kernels": [],
        "report_path": str(report_path),
        "ncu_warning": None,
    }

    rp = Path(report_path)
    if not rp.is_file() or rp.stat().st_size == 0:
        out["ncu_warning"] = f"report file missing or empty: {rp}"
        return out

    # Lazy import — module-level import would tie KernelGYM startup to
    # ncu_report being on PYTHONPATH.
    try:
        import ncu_report  # type: ignore[import-not-found]
    except ImportError:
        # Try to discover Nsight install on the fly.
        from kernelgym.toolkit.kernelbench.ncu_runner import ensure_ncu_report_importable
        if not ensure_ncu_report_importable():
            out["ncu_warning"] = (
                "ncu_report Python module not importable. "
                "Set NCU_PYTHON_EXTRA_PATHS or PYTHONPATH to the Nsight Compute "
                "extras/python directory."
            )
            return out
        import ncu_report  # type: ignore[import-not-found]  # noqa: F401

    try:
        rep = ncu_report.load_report(str(rp))
    except Exception as e:  # noqa: BLE001
        out["ncu_warning"] = f"ncu_report.load_report failed: {e}"
        return out

    # Collect actions.
    actions: List[Any] = []
    try:
        for ri in range(rep.num_ranges()):
            rng = rep.range_by_idx(ri)
            for ai in range(rng.num_actions()):
                actions.append(rng.action_by_idx(ai))
    except Exception as e:  # noqa: BLE001
        out["ncu_warning"] = f"failed to enumerate actions: {e}"
        return out

    if not actions:
        out["ncu_warning"] = "no actions in report (-k regex may have filtered everything)"
        return out

    # Per-kernel CLI rule extraction (one call, parses all kernels).
    if ncu_binary is None:
        ncu_binary = shutil.which("ncu")
    rules_by_kernel: Dict[str, List[Dict[str, Any]]] = {}
    if ncu_binary:
        rules_by_kernel = _rules_from_details_cli(rp, ncu_binary, top_k_rules)
    else:
        logger.info("ncu binary unavailable; skipping rule extraction")

    # Build per-action summaries.
    kernels: List[Dict[str, Any]] = []
    for a in actions:
        try:
            name = a.name()
        except Exception:
            name = "<unknown>"
        # ncu --page details prints the demangled-but-stripped name (e.g.
        # ``probe_add_kernel`` not the full signature). action.name()
        # also returns this in 2025.1, but be defensive: try exact match,
        # else fall back to the only-kernel-bucket case.
        rules = rules_by_kernel.get(name)
        if rules is None and len(rules_by_kernel) == 1:
            rules = next(iter(rules_by_kernel.values()))
        if rules is None:
            rules = []
        kernels.append({
            "name": name,
            "key_metrics": _key_metrics_for_action(a),
            "top_rules": rules,
        })

    out["kernels"] = kernels
    return out
