"""Extract exposed (host-facing) operator names + forward call-site analysis.

Stage 1 of the anti-hack decoy detection pipeline: pure AST, zero GPU cost.

Key design difference from ``kernel_names.py``:
  * ``kernel_names.py`` extracts ``__global__`` (device-side) kernel names
    for Stage 2 profiler kernel-coverage matching.
  * This module extracts **host wrapper** exposed names for Stage 1 AST
    analysis.  The host name (e.g. ``my_relu``) is what appears in
    ``ModelNew.forward``, NOT the ``__global__`` name (e.g.
    ``my_relu_kernel``).

Sources of exposed names:
  1. ``load_inline(..., functions=["op1", "op2"])`` — parsed via ast.
  2. ``m.def("op_name", ...)`` in cpp_sources strings — regex over raw text.

Forward analysis:
  * Finds ``class ModelNew(nn.Module)`` and its ``forward`` method.
  * Walks forward + any ``self.<helper>(...)`` helper methods within
    the same class.
  * Collects all ``ast.Call`` target names and matches against exposed set.
  * Detects opaque dispatch patterns that prevent reliable static analysis
    (getattr / alias / torch.ops.* / functools.partial).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Regex patterns for cpp_sources extraction
# ---------------------------------------------------------------------------

_M_DEF_RE = re.compile(r'm\.def\s*\(\s*"([^"]+)"')
"""Match ``m.def("op_name", ...)`` in cpp source strings."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForwardCallReport:
    """Result of analyzing forward-method call sites against exposed names.

    Attributes:
        called_exposed: Exposed operator names that appear as call targets
            in ``ModelNew.forward`` (or its helpers).
        uncalled_exposed: Exposed operator names that were declared but
            never appear as call targets — dead-code candidates.
        has_opaque_dispatch: True when forward contains patterns that
            cannot be statically resolved (e.g. ``getattr``, alias calls,
            ``torch.ops.*``, ``functools.partial``).  When True, Stage 1
            MUST NOT early-exit; fall back to Stage 2 profiler.
    """

    called_exposed: List[str] = field(default_factory=list)
    uncalled_exposed: List[str] = field(default_factory=list)
    has_opaque_dispatch: bool = False


# ---------------------------------------------------------------------------
# extract_exposed_op_names
# ---------------------------------------------------------------------------

def extract_exposed_op_names(custom_src: str) -> List[str]:
    """Extract host-facing operator names from kernel source.

    Sources scanned (in order):
      1. ``load_inline(..., functions=["op1", "op2"])`` — ast parse to
         extract string literals from the ``functions`` keyword argument.
      2. ``m.def("op_name", ...)`` in cpp_sources / raw text — regex
         over the full source.

    Returns a deduplicated list in declaration order.  Empty list if
    no exposed names are found.

    Examples
    --------
    >>> src = '''
    ... mod = load_inline(name="ext", cuda_sources=cuda_src,
    ...                   functions=["my_relu", "my_gelu"])
    ... class ModelNew(nn.Module):
    ...     def forward(self, x):
    ...         return mod.my_relu(x)
    ... '''
    >>> extract_exposed_op_names(src)
    ['my_relu', 'my_gelu']

    >>> extract_exposed_op_names("import torch\\nclass ModelNew(nn.Module):\\n    def forward(self, x): return x+1")
    []
    """
    if not custom_src:
        return []

    exposed: List[str] = []
    seen: Set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            exposed.append(name)

    # 1. AST: parse load_inline(..., functions=[...])
    _extract_from_load_inline_ast(custom_src, _add)

    # 2. Regex: m.def("...") across the raw source (catches strings inside
    #    cpp_sources / TORCH_LIBRARY blocks)
    for m in _M_DEF_RE.finditer(custom_src):
        _add(m.group(1))

    return exposed


def _extract_from_load_inline_ast(
    src: str, add: callable
) -> None:
    """Parse *src* with ast and invoke *add(name)* for every string in
    ``load_inline(..., functions=[...])`` keyword arguments."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return  # mixed Python/CUDA sources can fail — best-effort

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_load_inline_call(node):
            continue
        for kw in node.keywords:
            if kw.arg != "functions" or kw.value is None:
                continue
            _collect_str_list(kw.value, add)


def _is_load_inline_call(node: ast.Call) -> bool:
    """Return True if *node* is a call to ``load_inline`` (bare or qualified)."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "load_inline":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "load_inline":
        return True
    return False


def _collect_str_list(node: ast.expr, add: callable) -> None:
    """Walk *node* (expected to be a list of string constants) and invoke
    *add* for each string element.  Handles ``ast.List`` and single
    ``ast.Constant(str)`` gracefully."""
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                add(elt.value)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        add(node.value)
    elif isinstance(node, ast.Call):
        # Unwrap callable wrappers around a list literal:
        #   frozenset(["op1","op2"]), tuple([...]), etc.
        # The first string-valued arg inside any wrapping call
        # is treated as the source list.
        for arg in node.args:
            _collect_str_list(arg, add)


# ---------------------------------------------------------------------------
# analyze_forward_calls
# ---------------------------------------------------------------------------

def analyze_forward_calls(
    custom_src: str, exposed_names: List[str]
) -> ForwardCallReport:
    """Analyze whether exposed operators are called in ``ModelNew.forward``.

    Parses the Python source to find ``class ModelNew(nn.Module)`` →
    ``forward`` method, collects all call-target names in forward and any
    helper methods called via ``self.<helper>(...)``, then matches against
    the *exposed_names* set.

    **Sub-module recursion**: when ``ModelNew.forward`` calls
    ``self.child(x)`` and ``child`` is an instance of a custom
    ``nn.Module`` subclass defined in the same source, this function
    recursively walks that child's ``forward`` and its transitive
    helpers / sub-modules.

    Also detects opaque dispatch patterns that prevent reliable static
    analysis (see ``_is_opaque_call`` and ``_has_alias_pattern``).

    Args:
        custom_src: Full Python source (typically LLM-generated).
        exposed_names: Host-facing operator names from
            ``extract_exposed_op_names``.

    Returns:
        ``ForwardCallReport`` with called/uncalled breakdown and opaque flag.
    """
    if not custom_src or not exposed_names:
        return ForwardCallReport(
            uncalled_exposed=list(exposed_names) if exposed_names else [],
            has_opaque_dispatch=False,
        )

    exposed_set = set(exposed_names)

    # --- parse ---
    try:
        tree = ast.parse(custom_src)
    except SyntaxError:
        # Can't parse — conservatively mark opaque to avoid false decoy
        return ForwardCallReport(
            uncalled_exposed=list(exposed_names),
            has_opaque_dispatch=True,
        )

    # --- locate all nn.Module subclasses and the model class ---
    all_modules = _find_all_nn_module_classes(tree)
    model_class = _find_class_by_name(tree, "ModelNew")
    if model_class is None:
        model_class = _find_any_nn_module_class(tree)

    if model_class is None:
        return ForwardCallReport(
            uncalled_exposed=list(exposed_names),
            has_opaque_dispatch=False,
        )

    forward_node = _find_method(model_class, "forward")
    if forward_node is None:
        return ForwardCallReport(
            uncalled_exposed=list(exposed_names),
            has_opaque_dispatch=False,
        )

    # --- build self.attr → CustomClass mapping (sub-module instances) ---
    self_attr_to_class: Dict[str, str] = {}
    if all_modules:
        self_attr_to_class = _build_self_attr_module_map(model_class, all_modules)

    # --- recursively collect every forward in the call chain ------------
    #  * model_class.forward
    #  * helper methods of model_class called via self.<helper>(...)
    #  * forward of custom sub-nn.Module instances (self.child(x))
    all_forwards: List[ast.FunctionDef] = []
    _collect_forward_chain(
        model_class, forward_node, self_attr_to_class,
        all_modules, all_forwards,
    )

    # --- walk every collected forward: collect call targets & detect opaque ---
    call_targets: List[str] = []
    has_opaque = False

    for func_node in all_forwards:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                callee = _extract_call_target(node)
                if callee is not None:
                    call_targets.append(callee)
                if _is_opaque_call(node):
                    has_opaque = True

    # --- alias pattern detection (separate pass, across all forwards) ---
    if not has_opaque:
        has_opaque = _has_alias_pattern(forward_node, all_forwards)

    # --- compute breakdown ---
    seen_calls: Set[str] = set(call_targets)
    called = [n for n in exposed_names if n in seen_calls]
    uncalled = [n for n in exposed_names if n not in seen_calls]

    return ForwardCallReport(
        called_exposed=called,
        uncalled_exposed=uncalled,
        has_opaque_dispatch=has_opaque,
    )


# ---------------------------------------------------------------------------
# AST helpers — class / method discovery
# ---------------------------------------------------------------------------

def _find_class_by_name(tree: ast.AST, name: str) -> Optional[ast.ClassDef]:
    """Find a top-level class definition by *name*."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_any_nn_module_class(tree: ast.AST) -> Optional[ast.ClassDef]:
    """Find any class inheriting from ``nn.Module`` (walking all nodes)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if _is_nn_module_base(base):
                    return node
    return None


def _find_all_nn_module_classes(tree: ast.AST) -> Dict[str, ast.ClassDef]:
    """Return all classes inheriting from ``nn.Module``, keyed by name.

    Walks the entire AST so nested class definitions are included.
    """
    result: Dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if _is_nn_module_base(base):
                    result[node.name] = node
                    break
    return result


def _is_nn_module_base(node: ast.expr) -> bool:
    """True if *node* represents ``nn.Module``."""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "nn"
            and node.attr == "Module"
        )
    return False


def _find_method(
    class_node: ast.ClassDef, method_name: str
) -> Optional[ast.FunctionDef]:
    """Find a method by name inside a class body (non-recursive)."""
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _find_called_self_methods(
    forward_node: ast.FunctionDef, class_node: Optional[ast.ClassDef]
) -> List[ast.FunctionDef]:
    """Return methods of *class_node* that are called via
    ``self.<method>(...)`` from within *forward_node*."""
    if class_node is None:
        return []

    called: Set[str] = set()
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if (
                isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                called.add(func.attr)

    if not called:
        return []

    # Build a method lookup for the class
    methods: Dict[str, ast.FunctionDef] = {}
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            methods[node.name] = node

    return [methods[name] for name in called if name in methods]


# ---------------------------------------------------------------------------
# AST helpers — sub-module instance tracking & forward-chain collection
# ---------------------------------------------------------------------------

def _build_self_attr_module_map(
    model_class: ast.ClassDef,
    all_module_classes: Dict[str, ast.ClassDef],
) -> Dict[str, str]:
    """Map ``self.<attr>`` → class name for custom ``nn.Module`` instances.

    Scans the model class's ``__init__`` for ``self.attr = CustomClass()``
    and the class body for module-level ``attr = CustomClass()`` assignments
    where *CustomClass* is a key in *all_module_classes*.
    """
    mapping: Dict[str, str] = {}

    # Class-level assignments (outside methods): attr = CustomClass()
    for node in model_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    cls_name = _extract_constructor_class_name(
                        node.value, all_module_classes
                    )
                    if cls_name:
                        mapping[target.id] = cls_name
        elif isinstance(node, ast.AnnAssign):
            # attr: CustomClass = CustomClass()
            if isinstance(node.target, ast.Name) and node.value is not None:
                cls_name = _extract_constructor_class_name(
                    node.value, all_module_classes
                )
                if cls_name:
                    mapping[node.target.id] = cls_name

    # __init__ assignments: self.attr = CustomClass()
    init_node = _find_method(model_class, "__init__")
    if init_node is not None:
        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            cls_name = _extract_constructor_class_name(
                                node.value, all_module_classes
                            )
                            if cls_name:
                                mapping[target.attr] = cls_name

    return mapping


def _extract_constructor_class_name(
    node: ast.expr,
    all_module_classes: Dict[str, ast.ClassDef],
) -> Optional[str]:
    """If *node* is a direct ``CustomClass()`` call whose name is a known
    ``nn.Module`` subclass, return the class name.  Otherwise ``None``.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in all_module_classes:
            return func.id
    return None


def _collect_forward_chain(
    model_class: ast.ClassDef,
    forward_node: ast.FunctionDef,
    self_attr_to_class: Dict[str, str],
    all_module_classes: Dict[str, ast.ClassDef],
    out_forwards: List[ast.FunctionDef],
) -> None:
    """Recursively collect every ``forward`` in the call graph into *out_forwards*.

    Starting from *model_class* and *forward_node*, follows:
      * ``self.<helper>(...)`` → helper methods defined on the same class.
      * ``self.<attr>(...)``  → custom sub-``nn.Module`` instances → their
        ``forward`` (and transitively their helpers / sub-modules).

    Each function is appended at most once (by identity).
    """
    visited_ids: Set[int] = set()
    queue: List[tuple] = [(model_class, forward_node)]

    while queue:
        cls, fwd = queue.pop(0)
        fwd_id = id(fwd)
        if fwd_id in visited_ids:
            continue
        visited_ids.add(fwd_id)
        out_forwards.append(fwd)

        # 1. Helper methods on the same class (self.<method>(...))
        helpers = _find_called_self_methods(fwd, cls)
        for h in helpers:
            if id(h) not in visited_ids:
                queue.append((cls, h))

        # 2. Sub-module instances (self.<attr>(...) where attr is a custom module)
        if self_attr_to_class:
            for node in ast.walk(fwd):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if not (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                ):
                    continue
                attr_name = func.attr
                child_cls_name = self_attr_to_class.get(attr_name)
                if child_cls_name is None:
                    continue
                child_cls = all_module_classes.get(child_cls_name)
                if child_cls is None:
                    continue
                child_fwd = _find_method(child_cls, "forward")
                if child_fwd is not None and id(child_fwd) not in visited_ids:
                    # Build self_attr mapping for the child class as well
                    # so transitive sub-module chains are followed.
                    child_attr_map = _build_self_attr_module_map(
                        child_cls, all_module_classes
                    )
                    queue.append((child_cls, child_fwd))
                    # Merge the child's attr map for future lookups inside
                    # this same collect call (sub-modules of sub-modules).
                    for k, v in child_attr_map.items():
                        if k not in self_attr_to_class:
                            self_attr_to_class[k] = v


# ---------------------------------------------------------------------------
# AST helpers — call-target extraction
# ---------------------------------------------------------------------------

def _extract_call_target(call_node: ast.Call) -> Optional[str]:
    """Extract the call-target name from an ``ast.Call`` node.

    Returns the last identifier / attribute name of the callee:
      * ``func(args)`` → ``"func"``
      * ``mod.method(args)`` → ``"method"``
      * ``a.b.c(args)`` → ``"c"``
      * ``torch.ops.lib.op(args)`` → ``None`` (skipped)
      * ``getattr(x, name)(args)`` → ``None`` (opaque)
    """
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if _is_torch_ops_chain(func):
            return None
        return func.attr
    # Complex callee (e.g. another Call, Lambda, ...) — cannot statically resolve
    return None


# ---------------------------------------------------------------------------
# AST helpers — opaque dispatch detection
# ---------------------------------------------------------------------------

def _is_opaque_call(call_node: ast.Call) -> bool:
    """Return True if *call_node* represents an opaque dispatch pattern.

    Detects:
      1. ``getattr(obj, name)(args)`` — immediate call on getattr result.
      2. ``getattr(obj, name)`` — bare call (stored then called later).
      3. ``functools.partial(fn)(args)`` / ``partial(fn)(args)`` — immediate.
      4. ``functools.partial(fn)`` / ``partial(fn)`` — bare (stored, then called).
      5. ``torch.ops.<lib>.<op>(args)`` — opaque PyTorch operator dispatch.
    """
    func = call_node.func

    # Pattern 1: getattr(...)(...) — getattr called, then result called
    if isinstance(func, ast.Call):
        inner_callee = func.func
        if isinstance(inner_callee, ast.Name) and inner_callee.id == "getattr":
            return True
        # Pattern 3a: functools.partial(...)(...) or partial(...)(...)
        if isinstance(inner_callee, ast.Attribute) and inner_callee.attr == "partial":
            return True
        if isinstance(inner_callee, ast.Name) and inner_callee.id == "partial":
            return True

    # Pattern 2: getattr(...) — bare (stored, then used via alias later)
    if isinstance(func, ast.Name) and func.id == "getattr":
        return True

    # Pattern 4: bare partial(...) or functools.partial(...) — stored, then called
    if isinstance(func, ast.Attribute) and func.attr == "partial":
        return True
    if isinstance(func, ast.Name) and func.id == "partial":
        return True

    # Pattern 5: torch.ops.<lib>.<op>(args)
    if _is_torch_ops_chain(func):
        return True

    return False


def _is_torch_ops_chain(node: ast.expr) -> bool:
    """True if *node* is like ``torch.ops.xxx.yyy`` or deeper."""
    if isinstance(node, ast.Attribute):
        value = node.value
        if isinstance(value, ast.Attribute):
            grandparent = value.value
            if isinstance(grandparent, ast.Attribute):
                great = grandparent.value
                if (
                    isinstance(great, ast.Name)
                    and great.id == "torch"
                    and grandparent.attr == "ops"
                ):
                    return True
    return False


# ---------------------------------------------------------------------------
# Alias-pattern detection
# ---------------------------------------------------------------------------

def _has_alias_pattern(
    forward_node: ast.FunctionDef,
    helper_methods: List[ast.FunctionDef],
) -> bool:
    """Return True if forward (or helpers) uses alias patterns like
    ``op = mod.my_func; op(x)`` that prevent static call-site analysis.

    Heuristic: if a simple-name variable is assigned from an
    ``ast.Attribute`` chain and later appears as a call target in the
    same function body, flag it as opaque.
    """
    for func_node in [forward_node] + helper_methods:
        detector = _AliasDetector()
        detector.visit(func_node)
        if detector.has_alias_call:
            return True
    return False


class _AliasDetector(ast.NodeVisitor):
    """Walk a function body to detect alias-then-call patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.assigned_from_attr: Set[str] = set()
        self.has_alias_call: bool = False

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track simple names assigned from an Attribute chain
        if isinstance(node.value, ast.Attribute):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assigned_from_attr.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.assigned_from_attr:
                self.has_alias_call = True
        self.generic_visit(node)
