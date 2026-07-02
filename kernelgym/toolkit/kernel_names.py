"""Extract CUDA ``__global__`` kernel names from generated kernel source."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import List, Optional


_GLOBAL_RE = re.compile(r"\b__global__\b")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")

_RESERVED = {
    "void",
    "static",
    "inline",
    "extern",
    "const",
    "constexpr",
    "__device__",
    "__host__",
    "__forceinline__",
    "__noinline__",
    "__launch_bounds__",
    "__cluster_dims__",
    "__grid_constant__",
    "__restrict__",
    "__restrict",
    "int",
    "float",
    "double",
    "bool",
    "char",
    "short",
    "long",
    "unsigned",
    "signed",
    "template",
    "typename",
    "class",
    "struct",
}


def _strip_comments(src: str) -> str:
    """Strip comments while preserving CUDA code embedded in Python strings."""
    try:
        ast.parse(src)
    except SyntaxError:
        pass
    else:
        try:
            tokens = [
                tok
                for tok in tokenize.generate_tokens(io.StringIO(src).readline)
                if tok.type != tokenize.COMMENT
            ]
            src = tokenize.untokenize(tokens)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass

    src = _BLOCK_COMMENT_RE.sub("", src)
    src = _LINE_COMMENT_RE.sub("", src)
    return src


def _skip_balanced(src: str, open_ch: str, close_ch: str, pos: int) -> int:
    depth = 0
    while pos < len(src):
        char = src[pos]
        if char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    return len(src)


def _find_kernel_name_after(src: str, start: int) -> tuple[Optional[str], int]:
    """Find the function name after a ``__global__`` token."""
    pos = start
    candidate: Optional[str] = None

    while pos < len(src):
        char = src[pos]
        if char.isspace() or char in "*&":
            pos += 1
            continue

        if char == "(":
            if candidate is not None:
                return candidate, pos + 1
            pos = _skip_balanced(src, "(", ")", pos)
            continue

        if char == "<":
            pos = _skip_balanced(src, "<", ">", pos)
            continue

        if char.isalpha() or char == "_":
            match = _IDENT_RE.match(src, pos)
            if not match:
                return None, pos + 1

            ident = match.group(0)
            new_pos = match.end()
            tmp = new_pos
            while tmp < len(src) and src[tmp].isspace():
                tmp += 1

            if ident in _RESERVED:
                if tmp < len(src) and src[tmp] == "(":
                    pos = _skip_balanced(src, "(", ")", tmp)
                else:
                    pos = new_pos
                continue

            candidate = ident
            pos = new_pos
            continue

        return None, pos + 1

    return None, pos


def extract_global_kernel_names(custom_src: str) -> List[str]:
    """Return source-ordered CUDA ``__global__`` function names.

    KernelBench submissions commonly put CUDA in Python triple-quoted strings
    passed to ``torch.utils.cpp_extension.load_inline``. This parser scans the
    full Python source so those embedded declarations are visible.
    """
    if not custom_src:
        return []

    cleaned = _strip_comments(custom_src)
    seen = set()
    names: List[str] = []
    pos = 0

    while pos < len(cleaned):
        match = _GLOBAL_RE.search(cleaned, pos)
        if not match:
            break
        name, end = _find_kernel_name_after(cleaned, match.end())
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        pos = end

    return names
