"""Extract ``__global__`` kernel function names from CUDA source.

Used by the NCU profiling pipeline to build a ``-k regex:^(name1|...)$``
filter so that ``ncu`` only profiles the user's custom kernels and skips
cuBLAS/cuDNN/PyTorch eager kernels (which would otherwise dominate the
report and slow ncu's replay loop).

Robust against:
  * leading qualifiers (``extern "C"``, ``static``, ``inline``, ``__launch_bounds__``)
  * template kernels (``template <int N> __global__ void foo(...)``)
  * complex return types (``__global__ void __launch_bounds__(256, 4) foo(...)``)
  * commented-out kernels (``// __global__ void foo(...)``)
  * ``__device__`` helpers (correctly excluded)

The regex is intentionally over-permissive on the function-name capture
group; we then filter against C/C++ identifier rules.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

# Strategy: don't try to write one giant regex that handles every C++
# declaration shape. Instead, do a two-pass scan:
#
#   1. Locate every ``__global__`` token.
#   2. From there, walk forward, skipping balanced ``(...)`` groups
#      (handles ``__launch_bounds__(256, 4)``) and reading identifier
#      tokens; the LAST identifier before the un-skipped ``(`` is the
#      kernel name.
#
# This is far easier to reason about than a multi-page regex.

_GLOBAL_RE = re.compile(r"\b__global__\b")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# Tokens that should NOT be treated as the kernel name even if they
# happen to be the last identifier before ``(``.
_RESERVED = {
    "void", "static", "inline", "extern", "const", "constexpr",
    "__device__", "__host__", "__forceinline__", "__noinline__",
    "__launch_bounds__", "__cluster_dims__", "__grid_constant__",
    "__restrict__", "__restrict",
    # primitive return types — kernels return void anyway, but be safe.
    "int", "float", "double", "bool", "char", "short", "long",
    "unsigned", "signed",
    "if", "while", "for", "return",
    "template", "typename", "class", "struct", "using",
}

# Strip C and C++ comments before parsing — most false-positives come
# from kernels mentioned in comments / docstrings.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_PY_TRIPLE_RE = re.compile(r'"{3}.*?"{3}|\'{3}.*?\'{3}', re.DOTALL)


def _strip_comments(src: str) -> str:
    """Remove C/C++ comments AND Python triple-quoted strings.

    Note: KernelGYM kernels are typically emitted by an LLM as a Python
    source where the CUDA code lives inside a triple-quoted string passed
    to ``load_inline``. We don't strip those — we WANT to find the
    ``__global__`` declarations inside. Only strip Python triple-quoted
    *docstrings* by relying on the fact that real kernel sources contain
    actual ``__global__`` text (the string-stripping pass would otherwise
    erase them).

    Strategy: only strip C/C++ comments; leave Python string literals
    intact. This is good enough because:
      * The CUDA code passed to load_inline starts at the line containing
        ``__global__``, never inside a Python comment.
      * Python ``# ...`` comments inside a CUDA string also don't match
        our regex (they're stripped as ``// ...`` would be).
    """
    src = _BLOCK_COMMENT_RE.sub("", src)
    src = _LINE_COMMENT_RE.sub("", src)
    return src


def _find_kernel_name_after(src: str, start: int) -> tuple:
    """Walk forward from ``start`` to find the kernel-name identifier.

    Returns ``(name_or_None, end_pos)`` where ``end_pos`` is the position
    right after the opening ``(`` of the kernel's parameter list (used
    by the caller to advance past this declaration).

    Algorithm:
      * skip whitespace.
      * if next char is ``(``: this is an attribute call like
        ``__launch_bounds__(256, 4)``. Skip the balanced parentheses.
      * if next chars form a non-reserved identifier: remember it as the
        candidate kernel name, advance past it.
      * if next chars form a reserved identifier (``void``,
        ``__launch_bounds__``, ``static``, ...): advance past it without
        updating the candidate.
      * if next char is ``<``: skip the balanced angle brackets
        (template return types).
      * if next char is ``*`` or ``&`` or whitespace: advance.
      * if next char is ``(``: we've reached the parameter list — stop.
        Return the most recent candidate (this is the kernel name).
      * any other char: bail (give up on this match).
    """
    n = len(src)
    pos = start
    candidate: Optional[str] = None  # last seen non-reserved identifier

    def _skip_balanced(open_ch: str, close_ch: str, p: int) -> int:
        """Skip a balanced ``open_ch...close_ch`` group starting at p
        (assumes ``src[p] == open_ch``). Returns position past the closer.
        """
        depth = 0
        while p < n:
            c = src[p]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return p + 1
            p += 1
        return n  # unbalanced — bail to end-of-source

    while pos < n:
        c = src[pos]
        if c.isspace() or c in "*&":
            pos += 1
            continue
        if c == "(":
            # If we have a candidate, this is the kernel parameter list.
            if candidate is not None:
                return candidate, pos + 1
            # Otherwise: an attribute-with-args macro right after
            # __global__ (rare). Skip the group and continue scanning.
            pos = _skip_balanced("(", ")", pos)
            continue
        if c == "<":
            pos = _skip_balanced("<", ">", pos)
            continue
        if c.isalpha() or c == "_":
            m = _IDENT_RE.match(src, pos)
            if not m:
                return None, pos + 1  # shouldn't happen
            ident = m.group(0)
            new_pos = m.end()
            # If the next non-space char is ``(`` AND this identifier is
            # reserved (e.g. ``__launch_bounds__``), treat it as an
            # attribute macro: skip the args.
            tmp = new_pos
            while tmp < n and src[tmp].isspace():
                tmp += 1
            if ident in _RESERVED:
                # Don't update candidate. If followed by `(`, skip args.
                if tmp < n and src[tmp] == "(":
                    pos = _skip_balanced("(", ")", tmp)
                else:
                    pos = new_pos
                continue
            # Non-reserved identifier — remember as candidate name.
            candidate = ident
            pos = new_pos
            continue
        # Unrecognised character — bail.
        return None, pos + 1

    return None, pos


def extract_global_kernel_names(custom_src: str) -> List[str]:
    """Return a deduplicated, source-ordered list of ``__global__`` names.

    Empty list if no ``__global__`` declarations are found.

    Examples
    --------
    >>> src = '''
    ... __global__ void foo(float* x) { ... }
    ... template <int N>
    ... __global__ void __launch_bounds__(256, 4) bar(float* y) { ... }
    ... __device__ int helper(int x) { return x; }
    ... '''
    >>> extract_global_kernel_names(src)
    ['foo', 'bar']
    """
    if not custom_src:
        return []
    cleaned = _strip_comments(custom_src)
    seen = set()
    out: List[str] = []
    pos = 0
    while pos < len(cleaned):
        m = _GLOBAL_RE.search(cleaned, pos)
        if not m:
            break
        name, end = _find_kernel_name_after(cleaned, m.end())
        if name and name not in seen:
            seen.add(name)
            out.append(name)
        pos = end
    return out


def build_kernel_regex(names: Iterable[str]) -> str:
    """Build an anchored alternation regex for ``ncu -k 'regex:...'``.

    Returns ``""`` if the input is empty (caller should skip ncu in that
    case rather than passing an empty regex).

    Names are escaped to handle unusual identifiers (rare but possible
    via ``extern "C"`` aliases).
    """
    names = [n for n in names if n]
    if not names:
        return ""
    pattern = "|".join(re.escape(n) for n in names)
    return f"^({pattern})$"
