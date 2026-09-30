# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The AST compiler (RFC-0002): a decorated Python function -> Surface IR, without executing it.

Every expression is classified by taint. A sub-tree with no dynamic name and no DSL marker call
is handed to Python (``static_eval``); everything else is compiled node by node into ops through
the :class:`~ascriptor.ir.builder.Builder`. Dynamic values are :class:`Dyn` wrappers around IR
values; the IR type of the value decides what an operator or method means.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import os
import textwrap
from dataclasses import dataclass, replace
from typing import Any

from ..ir import Builder, FuncRef, FunctionBuilder, Ident, Literal, Loc, Module, Value, check
from ..ir.core import Op
from ..ir.lexer import TokenStream, tokenize
from ..ir.registry import REGISTRY
from ..ir.types import (
    DTYPES,
    BufType,
    CellType,
    Dim,
    DimValue,
    DType,
    EventType,
    FlagType,
    MaskType,
    MemType,
    Product,
    Ragged,
    RegType,
    ScalarType,
    Type,
    UnalignRegType,
    is_scalar_int,
    parse_dim,
)
from . import dsl
from .errors import (
    E_ACLNN_NAME_COLLISION,
    E_BAD_COPY,
    E_BAD_OPERAND,
    E_BAD_SHAPE,
    E_BAD_SIGNATURE,
    E_CONDITIONAL_REBIND,
    E_DYNAMIC_BREAK_IN_UNROLL,
    E_NOT_CALLABLE,
    E_OUT_OF_REGION,
    E_REBIND_CELL,
    E_RETURN_NOT_LAST,
    E_STATIC_EVAL,
    E_UNKNOWN_NAME,
    E_UNSUPPORTED,
    CompileError,
)

# --------------------------------------------------------------------------- values
from .rules_mem import MemRules  # noqa: E402
from .rules_reg import RegRules  # noqa: E402
from .rules_sync import SyncRules  # noqa: E402
from .rules_vec import VecRules  # noqa: E402
from .values import (  # noqa: E402
    BoundMethod,
    Dyn,
    ElemOffset,
    Img2col,
    RegExpr,
    RegList,
    is_dynamic,
)

__all__ = ["compile_kernel", "Dyn", "ElemOffset", "BoundMethod", "RegExpr", "RegList", "Img2col", "is_dynamic"]


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


_SCALAR_BINOPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "div", ast.Mod: "mod",
    ast.LShift: "shl", ast.RShift: "shr", ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "xor",
}
_CONTAINER_BUILTINS = (zip, enumerate, list, tuple, len, reversed, dict, isinstance, type, id, repr, str)
_CMP_PREDS = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
_MUTEX_METHODS = {"lock": "sync.mutex_lock", "ready": "sync.mutex_ready", "wait": "sync.mutex_wait", "free": "sync.mutex_free"}


# --------------------------------------------------------------------------- entry points


def compile_kernel(kernel: dsl.KernelFn) -> Module:
    return Frontend(kernel).compile()


class Frontend:
    """Compiles one kernel and the vf / simt functions it calls into one module."""

    def __init__(self, kernel: dsl.KernelFn) -> None:
        self.kernel = kernel
        self.file = _source_path(kernel.fn)
        self.builder = Builder(kernel.name, device=kernel.device, mode=kernel.mode, source=self.file)
        self.callees: dict[Any, _Callee] = {}
        self.function_names: set[str] = set()

    def compile(self) -> Module:
        fc = FunctionCompiler(self, self.kernel.fn, "kernel", self.kernel.name)
        fc.compile_kernel(self.kernel)
        module = self.builder.finish()
        return check(module)

    def unique_function_name(self, base: str) -> str:
        name, k = base, 1
        while name in self.function_names:
            name = f"{base}.{k}"
            k += 1
        self.function_names.add(name)
        return name

    def compile_callee(self, target: Any, kind: str, args: list[Any], node: ast.AST, caller: FunctionCompiler) -> _Callee:
        """Compile a vf / simt function for one call-site signature (cached), RFC-0002 §4.2."""
        key_parts: list[Any] = [id(target)]
        for a in args:
            key_parts.append(("dyn", str(a.type)) if isinstance(a, Dyn) else ("static", repr(a)))
        key = tuple(key_parts)
        if key in self.callees:
            return self.callees[key]
        fn = target.fn
        params = list(inspect.signature(fn).parameters)
        if len(params) != len(args):
            raise CompileError(E_BAD_SIGNATURE, f"{target.name} takes {len(params)} argument(s), got {len(args)}", caller.file, node)
        # Parameter types come from the arguments; a tensor argument's dimensions that name kernel
        # values become the callee's own parameters (explicit when the value is also passed, implicit otherwise).
        value_to_param: dict[str, str] = {}
        for pname, a in zip(params, args, strict=True):
            if isinstance(a, Dyn) and not isinstance(a.type, MemType):
                value_to_param[a.value.name] = pname
        implicit: list[tuple[str, Value]] = []
        typed_params: list[tuple[str, Type]] = []
        static_bindings: dict[str, Any] = {}
        for pname, a in zip(params, args, strict=True):
            if isinstance(a, Dyn):
                t = a.type
                if isinstance(t, MemType):
                    t = _rewrite_dims(t, value_to_param, implicit, caller)
                elif isinstance(t, CellType):
                    t = ScalarType(t.dtype)
                typed_params.append((pname, t))
            elif isinstance(a, (int, float, bool, str, dsl.DTypeName, dsl.EnumValue, dsl.CastConfig)) or a is None:
                static_bindings[pname] = a
            else:
                raise CompileError(E_BAD_OPERAND, f"cannot pass {a!r} to {target.name}", caller.file, node)
        typed_params.extend((n, ScalarType(DTYPES["i32"])) for n, _ in implicit)
        name = self.unique_function_name(target.name)
        fc = FunctionCompiler(self, fn, kind, name)
        access = fc.compile_callee(typed_params, static_bindings)
        callee = _Callee(name, [n for n, _ in implicit], [v for _, v in implicit], access, static_bindings)
        self.callees[key] = callee
        return callee


@dataclass
class _Callee:
    name: str
    implicit_names: list[str]
    implicit_values: list[Value]
    access: dict[str, str]  # param name -> "read" | "write" | "readwrite"
    static_bindings: dict[str, Any]


def _rewrite_dims(t: MemType, value_to_param: dict[str, str], implicit: list[tuple[str, Value]], caller: FunctionCompiler) -> MemType:
    def one(d: Dim) -> Dim:
        if isinstance(d, DimValue):
            return DimValue(_param_for(d.name))
        if isinstance(d, Product):
            return Product(tuple(one(f) for f in d.factors))  # type: ignore[arg-type]
        return d

    def _param_for(name: str) -> str:
        if name in value_to_param:
            return value_to_param[name]
        for n, v in implicit:
            if v.name == name:
                return n
        src = caller.named_values[name]
        pname = name if name not in value_to_param.values() else f"{name}_dim"
        implicit.append((pname, src.value))
        value_to_param[name] = pname
        return pname

    return MemType(t.space, t.dtype, tuple(one(d) for d in t.dims), t.layout)


def _is_marker_failure(exc: BaseException) -> bool:
    msg = str(exc)
    return "is compiled, not executed" in msg or "takes no arguments" in msg


def _source_path(fn: Any) -> str:
    path = inspect.getsourcefile(fn) or "<unknown>"
    try:
        rel = os.path.relpath(path)
        return rel if not rel.startswith("..") else path
    except ValueError:
        return path


# --------------------------------------------------------------------------- the per-function compiler


class FunctionCompiler(RegRules, MemRules, SyncRules, VecRules):
    def __init__(self, fe: Frontend, fn: Any, kind: str, name: str) -> None:
        self.fe = fe
        self.fn = fn
        self.kind = kind
        self.name = name
        self.file = _source_path(fn)
        self.globals: dict[str, Any] = fn.__globals__
        self.closure: dict[str, Any] = dict(inspect.getclosurevars(fn).nonlocals)
        self.scope: dict[str, Any] = {}
        self.named_values: dict[str, Dyn] = {}  # IR value name -> Dyn (for dims that reference values)
        self.roots: dict[str, str] = {}  # view value name -> root value name (params) for access sets
        self.fb: FunctionBuilder | None = None
        self.loop_depth = 0
        self.unroll_depth = 0
        self.geoms: dict[str, Any] = {}  # value name -> Geom (rules_mem)
        self.l0_transposed: dict[str, bool] = {}  # L0 value name -> loaded from a transposed L1 view
        self.atomic: Ident | None = None  # inside `with atomic_add():` etc.
        self.vanished: dict[str, tuple[str, ast.AST]] = {}  # dynamic names bound inside a closed loop / branch (E0111 on use)
        self.tree = self._parse()

    # -- setup ---------------------------------------------------------------------------------

    def _parse(self) -> ast.FunctionDef:
        src = inspect.getsource(self.fn)
        dedented = textwrap.dedent(src)
        self.col_shift = len(src.splitlines()[0]) - len(dedented.splitlines()[0])
        tree = ast.parse(dedented, filename=self.file)
        ast.increment_lineno(tree, self.fn.__code__.co_firstlineno - 1)
        fdef = tree.body[0]
        if not isinstance(fdef, ast.FunctionDef):
            raise CompileError(E_UNSUPPORTED, "expected a function definition", self.file, fdef)
        return fdef

    def loc(self, node: ast.AST | None) -> Loc | None:
        if node is None or not hasattr(node, "lineno"):
            return None
        return Loc.of(f"{self.file}:{node.lineno}:{node.col_offset + self.col_shift}")

    def err(self, code: str, msg: str, node: ast.AST | None = None, note: str | None = None) -> CompileError:
        return CompileError(code, msg, self.file, node, note)

    def need_dtype(self, v: Any, node: ast.AST) -> DType:
        return _need_dtype(v, self, node)

    def attr(self, v: Any) -> Any:
        return self._attr_value(v)

    def dim_value(self, d: Dim, node: ast.AST) -> Any:
        return self._dim_value(d, node)

    # -- kernel signature ----------------------------------------------------------------------

    def compile_kernel(self, kernel: dsl.KernelFn) -> None:
        fdef = self.tree
        params: list[tuple[str, Type]] = []
        symbols: dict[str, ast.AST] = {}  # implicit shape scalar -> its first signature occurrence
        self.list_counts: dict[str, int] = {}  # GMList[..., count=n]: a pinned member count (RFC-0002 §3.3)
        seen_scalar = False
        for a in fdef.args.args:
            ann = self._annotation(a)
            if isinstance(ann, dsl.GMSpec):
                if seen_scalar:
                    raise self.err(E_BAD_SIGNATURE, f"tensor parameter {a.arg!r} after a scalar parameter; order is inputs, outputs, scalars", a)
                dims: list[Dim] = []
                for d in ann.dims:
                    if isinstance(d, int):
                        dims.append(d)
                    elif d == "?":
                        if not ann.is_list:
                            raise self.err(E_BAD_SHAPE, "'?' dims are only allowed in GMList parameters", a)
                        dims.append(Ragged())
                    else:
                        dims.append(_parse_symbol_dim(d, symbols, a, self))
                if ann.is_list and ann.count is not None:
                    self.list_counts[a.arg] = int(ann.count)
                params.append((a.arg, MemType("gmlist" if ann.is_list else "gm", ann.dtype.ir, tuple(dims))))
            elif isinstance(ann, dsl.DTypeName):
                seen_scalar = True
                params.append((a.arg, ScalarType(ann.ir)))
            elif ann is dsl.Var:
                seen_scalar = True
                params.append((a.arg, ScalarType(DTYPES["i32"])))
            elif ann in (dsl.GMTensor, dsl.GMTensorList):
                raise self.err(E_BAD_SIGNATURE, f"parameter {a.arg!r}: write GM[dtype, dims] (or GMList[...]); the bare {ann.__name__} annotation carries no rank", a)
            else:
                raise self.err(E_BAD_SIGNATURE, f"parameter {a.arg!r} needs an annotation: GM[...], GMList[...] or a dtype such as i32", a)
        declared = {n for n, _ in params}
        for sym, _ in symbols.items():
            if sym in declared:
                t = dict(params)[sym]
                if not is_scalar_int(t):
                    raise self.err(E_BAD_SIGNATURE, f"shape symbol {sym!r} is declared as {t}; it must be an integer scalar", fdef)
            else:
                params.append((sym, ScalarType(DTYPES["i32"])))
        parameter_nodes: dict[str, ast.AST] = {a.arg: a for a in fdef.args.args}
        for sym, node in symbols.items():
            parameter_nodes.setdefault(sym, node)
        attrs: dict[str, Any] = {}
        if kernel.block_dim is not None:
            attrs["block_dim"] = kernel.block_dim
        with self.fe.builder.function("kernel", self.name, params, attrs) as fb:
            self.fb = fb
            for n, _ in params:
                self._bind_param(fb.p(n))
            self.compile_body(fdef.body, top=True)
            outputs = self._outputs
            if outputs is not None:
                fb.attrs["outputs"] = [o.value for o in outputs]
                output_names = {o.value.name for o in outputs}
                aclnn_params = [p for p in params if isinstance(p[1], ScalarType) or p[0] not in output_names]
                self._check_aclnn_parameter_names(aclnn_params, parameter_nodes)

    def _check_aclnn_parameter_names(
        self, params: list[tuple[str, Type]], nodes: dict[str, ast.AST]
    ) -> None:
        """Reject input/attribute names CANN merges in the host-visible ACLNN API."""
        seen: dict[str, str] = {}
        for name, _ in params:
            api_name = _aclnn_parameter_name(name)
            previous = seen.get(api_name)
            if previous is not None:
                raise self.err(
                    E_ACLNN_NAME_COLLISION,
                    f"kernel input/scalar parameters {previous!r} and {name!r} both become {api_name!r} "
                    "in the ACLNN API; rename one so their lowerCamelCase names differ",
                    nodes.get(name, self.tree),
                    note="CANN lowercases the first character and camelizes underscore-separated components",
                )
            seen[api_name] = name

    def _annotation(self, a: ast.arg) -> Any:
        if a.annotation is None:
            return None
        resolved = self.fn.__annotations__.get(a.arg)
        if resolved is not None and not isinstance(resolved, str):
            return resolved
        try:
            return self.static_eval(a.annotation)
        except CompileError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self.err(E_BAD_SIGNATURE, f"cannot evaluate the annotation of {a.arg!r}: {exc}", a) from None

    def _bind_param(self, v: Value) -> None:
        d = Dyn(v)
        self.scope[v.name] = d
        self.named_values[v.name] = d
        self.roots[v.name] = v.name

    # -- vf / simt callee ----------------------------------------------------------------------

    def compile_callee(self, params: list[tuple[str, Type]], static_bindings: dict[str, Any]) -> dict[str, str]:
        fdef = self.tree
        with self.fe.builder.function(self.kind, self.name, params) as fb:
            self.fb = fb
            for n, _ in params:
                self._bind_param(fb.p(n))
            self.scope.update(static_bindings)
            self.compile_body(fdef.body, top=True)
        return self._access_sets(fb)

    def _access_sets(self, fb: FunctionBuilder) -> dict[str, str]:
        access: dict[str, str] = {}

        def note(name: str, mode: str) -> None:
            root = self.roots.get(name)
            if root is None or root not in fb.params:
                return
            cur = access.get(root)
            access[root] = mode if cur is None or cur == mode else "readwrite"

        for op in fb.finish_preview().walk():
            spec = REGISTRY.find(op.opcode)
            if spec is None:
                continue
            for i, x in enumerate(op.operands):
                ospec = spec.operands[min(i, len(spec.operands) - 1)] if spec.operands else None
                if ospec is None or not isinstance(x, Value):
                    continue
                if ospec.access == "read":
                    note(x.name, "read")
                elif ospec.access == "write":
                    note(x.name, "write")
                elif ospec.access == "readwrite":
                    note(x.name, "readwrite")
            for aspec in spec.attrs:
                if aspec.access in ("read", "write", "readwrite") and isinstance(op.attrs.get(aspec.name), Value):
                    note(op.attrs[aspec.name].name, aspec.access)
        return access

    # -- statements ----------------------------------------------------------------------------

    _outputs: list[Dyn] | None = None

    def compile_body(self, stmts: list[ast.stmt], top: bool = False) -> None:
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Return):
                if not top:
                    raise self.err(E_RETURN_NOT_LAST, "return is only allowed at the end of the function body", stmt)
                if i != len(stmts) - 1:
                    raise self.err(E_RETURN_NOT_LAST, "return must be the last statement", stmt)
                self.compile_return(stmt)
                return
            self.compile_stmt(stmt)
        if top and self.kind == "kernel":
            raise self.err(E_RETURN_NOT_LAST, "a kernel must end with `return <outputs>`", self.tree)

    def compile_stmt(self, node: ast.stmt) -> None:
        match node:
            case ast.Expr(value=value):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return  # docstring
                self.ev(value)
            case ast.Assign(targets=[target], value=value):
                self.compile_assign(target, value, node)
            case ast.Assign():
                raise self.err(E_UNSUPPORTED, "chained assignment is not supported", node)
            case ast.AnnAssign(target=target, value=value) if value is not None:
                self.compile_assign(target, value, node)
            case ast.AugAssign(target=target, op=op, value=value):
                self.compile_augassign(target, op, value, node)
            case ast.For():
                self.compile_for(node)
            case ast.If():
                self.compile_if(node)
            case ast.With():
                self.compile_with(node)
            case ast.Pass():
                return
            case ast.Break():
                if self.loop_depth > 0 and self.unroll_depth == 0:
                    self.emit("cf.break", (), {}, None, None, node)
                elif self.unroll_depth > 0 and self.loop_depth == 0:
                    raise _Break()
                else:
                    raise self.err(E_DYNAMIC_BREAK_IN_UNROLL, "break must be inside a loop", node)
            case ast.Continue():
                if self.loop_depth > 0 and self.unroll_depth == 0:
                    self.emit("cf.continue", (), {}, None, None, node)
                elif self.unroll_depth > 0 and self.loop_depth == 0:
                    raise _Continue()
                else:
                    raise self.err(E_DYNAMIC_BREAK_IN_UNROLL, "continue must be inside a loop", node)
            case ast.Assert(test=test, msg=msg):
                if not self.contains_dynamic(test):
                    if not self.static_eval(test):
                        raise self.err(E_STATIC_EVAL, "static assertion failed" + (f": {self.static_eval(msg)}" if msg else ""), node)
                else:
                    cond = self.as_bool(self.ev(test), node)
                    self.emit("debug.assert", (cond,), {"msg": self.static_eval(msg)} if msg else {}, None, None, node)
            case ast.Return():
                raise self.err(E_RETURN_NOT_LAST, "return is only allowed at the end of the function body", node)
            case ast.FunctionDef() | ast.While() | ast.Try() | ast.Raise() | ast.Global() | ast.Nonlocal() | ast.Delete() | ast.Import() | ast.ImportFrom() | ast.ClassDef():
                raise self.err(E_UNSUPPORTED, f"{type(node).__name__} is not supported in kernel code", node)
            case _:
                raise self.err(E_UNSUPPORTED, f"unsupported statement {type(node).__name__}", node)

    def compile_return(self, node: ast.Return) -> None:
        if self.kind != "kernel":
            if node.value is not None:
                raise self.err(E_UNSUPPORTED, f"a {self.kind} function cannot return a value", node)
            return
        if node.value is None:
            raise self.err(E_BAD_SIGNATURE, "a kernel returns its output parameters", node)
        vals = self.ev(node.value)
        outs = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        assert self.fb is not None
        for o in outs:
            if not (isinstance(o, Dyn) and o.value.name in self.fb.params and self.fb.params[o.value.name] is o.value):
                raise self.err(E_BAD_SIGNATURE, "a kernel may only return its GM parameters", node)
        self._outputs = outs
        self.emit("cf.return", tuple(outs), {}, None, None, node)

    def compile_assign(self, target: ast.expr, value: ast.expr, node: ast.stmt) -> None:
        if isinstance(target, ast.Name):
            v = self.ev(value)
            self.bind(target.id, v, node, name_result=True)
        elif isinstance(target, ast.Tuple):
            v = self.ev(value)
            if not isinstance(v, (list, tuple)) or len(v) != len(target.elts):
                raise self.err(E_BAD_OPERAND, "tuple unpacking needs a static container of matching length", node)
            for t, x in zip(target.elts, v, strict=True):
                if not isinstance(t, ast.Name):
                    raise self.err(E_UNSUPPORTED, "nested unpacking is not supported", node)
                self.bind(t.id, x, node)
        elif isinstance(target, ast.Subscript):
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, "item assignment is only allowed in simt functions; use `<<=` to copy", node)
            base = self.ev_target(target.value)
            if not (isinstance(base, Dyn) and isinstance(base.type, MemType)):
                raise self.err(E_BAD_OPERAND, "item assignment needs a tensor", node)
            index = self.rvalue(self.ev(target.slice), node)
            val = self.rvalue(self.ev(value), node)
            self.emit("simt.store", (base, index, val), {}, None, None, node)
        else:
            raise self.err(E_UNSUPPORTED, "unsupported assignment target", node)

    def bind(self, name: str, v: Any, node: ast.AST, name_result: bool = False) -> None:
        old = self.scope.get(name)
        if isinstance(old, Dyn) and isinstance(old.type, CellType) and isinstance(v, Dyn) and not isinstance(v.type, CellType):
            raise self.err(E_REBIND_CELL, f"'{name}' is a Var; assign into it with `{name}.set(...)` or `{name} <<= ...`", node,
                           note="plain `=` rebinds the name at compile time; it does not store into the Var")
        if name_result and isinstance(v, Dyn) and self.fb is not None:
            ops = self.fb._stack[-1]
            if ops and len(ops[-1].results) == 1 and ops[-1].results[0] is v.value and v.value.name not in self.fb.params:
                new = self.fb.rename_last_result(name)
                self.named_values.pop(v.value.name, None)
                root = self.roots.pop(v.value.name, None)
                # An allocation is its own root, and the root is a VALUE NAME: carrying the
                # pre-rename one forward would leave `roots` pointing at a name the IR no longer
                # has. Views keep pointing at whatever they were a view of.
                root = new.name if root == v.value.name else root
                v = Dyn(new)
                if root is not None:
                    self.roots[new.name] = root
        if name_result and isinstance(v, RegList) and not v.name and self.fb is not None:
            ops = self.fb._stack[-1]
            tail = ops[-len(v.elems):]
            if len(tail) == len(v.elems) and all(
                    op.opcode == "vf.reg" and op.results == (elem.value,) for op, elem in zip(tail, v.elems, strict=True)):
                for i, (op, elem) in enumerate(zip(tail, v.elems, strict=True)):
                    self.fb._names.discard(elem.value.name)
                    new = Value(self.fb.fresh(f"{name}_{i}"), elem.type)
                    ops[len(ops) - len(v.elems) + i] = replace(op, results=(new,), attrs={"name": new.name})
                    self.named_values.pop(elem.value.name, None)
                    v.elems[i] = Dyn(new)
                    self.named_values[new.name] = v.elems[i]
                v.name = name
        if isinstance(v, Dyn):
            self.named_values[v.value.name] = v
        self.scope[name] = v
        self.vanished.pop(name, None)

    def compile_augassign(self, target: ast.expr, op: ast.operator, value: ast.expr, node: ast.stmt) -> None:
        if isinstance(op, ast.LShift):
            self.compile_copy(target, value, node)
            return
        rhs = self.ev(value)
        if isinstance(op, ast.RShift) and (isinstance(rhs, ElemOffset) or (isinstance(rhs, Dyn) and isinstance(rhs.type, MemType))):
            scalar = self.ev_target(target)
            if isinstance(scalar, Dyn) and isinstance(scalar.type, CellType):
                self.store_scalar(rhs, scalar, node)
                return
        if isinstance(target, ast.Name):
            cur = self.lookup(target.id, target)
            if isinstance(cur, Dyn) and isinstance(cur.type, CellType):
                res = self.binop(op, cur, rhs, node)
                self.emit("scalar.set", (cur, res), {}, None, None, node)
            elif is_dynamic(cur) or is_dynamic(rhs):
                self.bind(target.id, self.binop(op, cur, rhs, node), node, name_result=True)
            else:
                self.scope[target.id] = _py_binop(op, cur, rhs)
        else:
            raise self.err(E_UNSUPPORTED, "augmented assignment needs a plain name on the left", node)

    def ev_target(self, node: ast.expr) -> Any:
        """Evaluate an assignment target (Store context) as the value it denotes."""
        import copy

        load = copy.deepcopy(node)
        for n in ast.walk(load):
            if isinstance(n, (ast.Name, ast.Subscript, ast.Attribute)):
                n.ctx = ast.Load()
        return self.ev(load)

    def compile_copy(self, target: ast.expr, value: ast.expr, node: ast.stmt) -> None:
        dst = self.ev_target(target)
        src = self.ev(value)
        if isinstance(dst, Dyn) and isinstance(dst.type, RegType):
            self.copy_into_reg(dst, src, node)
        elif isinstance(dst, Dyn) and isinstance(dst.type, MaskType):
            self.copy_into_mask(dst, src, node)
        elif isinstance(dst, RegList):
            self.copy_into_reglist(dst, src, node)
        elif self.is_ub(dst) and (self._is_reg_like(src) or isinstance(src, RegList)):
            self.store_to_ub(dst, src, node)
        elif isinstance(dst, Dyn) and isinstance(dst.type, MemType) and isinstance(src, Dyn) and isinstance(src.type, MemType):
            self.copy_memory(dst, src, node)
        elif isinstance(dst, Dyn) and isinstance(dst.type, CellType) and (isinstance(src, ElemOffset) or (isinstance(src, Dyn) and isinstance(src.type, MemType))):
            self.load_scalar(dst, src, node)
        elif isinstance(dst, Dyn) and isinstance(dst.type, CellType):
            self.emit("scalar.set", (dst, self.rvalue(src, node)), {}, None, None, node)
        else:
            raise self.err(E_BAD_COPY, f"cannot copy {_describe(src)} into {_describe(dst)}", node)

    def compile_for(self, node: ast.For) -> None:
        if node.orelse:
            raise self.err(E_UNSUPPORTED, "for/else is not supported", node)
        if not isinstance(node.target, (ast.Name, ast.Tuple)):
            raise self.err(E_UNSUPPORTED, "the loop variable must be a plain name or a tuple of names", node)
        if isinstance(node.target, ast.Tuple) and not all(isinstance(e, ast.Name) for e in node.target.elts):
            raise self.err(E_UNSUPPORTED, "nested loop-variable unpacking is not supported", node)
        it = node.iter
        callee = self.static_eval(it.func) if isinstance(it, ast.Call) and not self.contains_dynamic(it.func) else None
        if callee is builtins.range or callee is dsl.unroll:
            # D-026: ``range`` is the device loop even with constant bounds (the generated code keeps the loop);
            # only ``unroll`` copies the body per iteration, and it needs static bounds.
            what = "unroll()" if callee is dsl.unroll else "range()"
            if isinstance(node.target, ast.Tuple):
                raise self.err(E_UNSUPPORTED, f"{what} yields one value per iteration", node)
            if it.keywords:
                raise self.err(E_UNSUPPORTED, f"{what} takes no keyword arguments; the loop variable names the loop", node)
            args = [self.ev(a) for a in it.args]
            if not 1 <= len(args) <= 3:
                raise self.err(E_BAD_OPERAND, f"{what} takes one to three integer arguments", node)
            for arg in args:
                integer = isinstance(arg, int) or (isinstance(arg, Dyn) and isinstance(arg.type, (CellType, ScalarType))
                                                  and (arg.type.dtype.is_integer or arg.type.dtype.name == "b1"))
                if not integer:
                    raise self.err(E_BAD_OPERAND, f"{what} bounds and step must be integer scalars; use // for an integral static trip count", node)
            if len(args) == 3 and isinstance(args[2], int) and args[2] == 0:
                raise self.err(E_BAD_OPERAND, f"{what} step must not be zero", node)
            if callee is dsl.unroll:
                if any(is_dynamic(a) for a in args):
                    raise self.err(E_UNSUPPORTED, "unroll() needs static bounds; a loop over dynamic bounds is range()", node)
                self._unroll(node, range(*args))
                return
            lo, hi, step = (0, args[0], 1) if len(args) == 1 else (args[0], args[1], 1) if len(args) == 2 else (args[0], args[1], args[2])
            assert self.fb is not None
            with self.fb.loop(self._operand(lo, node), self._operand(hi, node), self._operand(step, node), name=node.target.id, loc=self.loc(node)) as i:
                self.loop_depth += 1
                with self.region_scope(node):
                    self.bind(node.target.id, Dyn(i), node)
                    self.named_values[i.name] = Dyn(i)
                    self.compile_body(node.body)
                self.loop_depth -= 1
            return
        seq = self.ev(it)
        if isinstance(seq, Dyn) and isinstance(seq.type, MemType) and seq.type.space == "gmlist":
            self._for_list(node, seq)
            return
        if is_dynamic(seq) and not isinstance(seq, (list, tuple)):
            raise self.err(E_UNSUPPORTED, "iterating over a dynamic value is not supported (only range(...) and static containers)", node)
        self._unroll(node, list(seq))

    def _unroll(self, node: ast.For, items: Any) -> None:
        self.unroll_depth += 1
        try:
            for item in items:
                if isinstance(node.target, ast.Tuple):
                    if not isinstance(item, (list, tuple)) or len(item) != len(node.target.elts):
                        raise self.err(E_BAD_OPERAND, "loop unpacking needs items of matching length", node)
                    for t, x in zip(node.target.elts, item, strict=True):
                        self.bind(t.id, x, node)  # type: ignore[attr-defined]
                else:
                    self.bind(node.target.id, item, node)
                try:
                    self.compile_body(node.body)
                except _Continue:
                    continue
                except _Break:
                    break
        finally:
            self.unroll_depth -= 1

    def compile_if(self, node: ast.If) -> None:
        if not self.contains_dynamic(node.test):
            branch = node.body if self.static_eval(node.test) else node.orelse
            self.compile_body(branch)
            return
        cond = self.as_bool(self.ev(node.test), node)
        assert self.fb is not None
        with self.fb.if_(self._operand(cond, node), loc=self.loc(node)) as ifb:
            with self.region_scope(node):
                self.compile_body(node.body)
            if node.orelse:
                with ifb.else_():
                    with self.region_scope(node):
                        self.compile_body(node.orelse)

    def compile_with(self, node: ast.With) -> None:
        if len(node.items) != 1 or node.items[0].optional_vars is not None:
            raise self.err(E_UNSUPPORTED, "with supports exactly one context and no `as`", node)
        ctx = node.items[0].context_expr
        if not isinstance(ctx, ast.Call) or self.contains_dynamic(ctx.func):
            raise self.err(E_UNSUPPORTED, "with needs auto_sync(), vec_scope() or cube_scope()", node)
        marker = self.static_eval(ctx.func)
        rule = dsl.rule_of(marker)
        kwargs = {kw.arg: self.static_eval(kw.value) for kw in ctx.keywords}
        args = [self.static_eval(a) for a in ctx.args]
        assert self.fb is not None
        if rule == "auto_sync":
            mode = args[0] if args else kwargs.get("mode", "conservative")
            attrs = {} if mode == "conservative" else {"mode": Ident(str(mode))}
            with self.fb.region("region.autosync", (), attrs, loc=self.loc(node)):
                with self.region_scope(node):
                    self.compile_body(node.body)
        elif rule in ("vec_scope", "cube_scope"):
            with self.fb.region("region.side", (), {"side": Ident(rule.split("_")[0])}, loc=self.loc(node)):
                with self.region_scope(node):
                    self.compile_body(node.body)
        elif rule is not None and rule.startswith("atomic:"):
            self.compile_atomic_with(rule.split(":")[1], ctx, node.body, node)
        else:
            raise self.err(E_UNSUPPORTED, "with needs auto_sync(), vec_scope(), cube_scope() or an atomic_* context", node)

    def region_scope(self, node: ast.AST):
        """Names bound inside a region vanish after it; rebinding an outer name inside is an error."""
        return _RegionScope(self, node)

    # -- expressions ---------------------------------------------------------------------------

    def lookup(self, name: str, node: ast.AST) -> Any:
        if name in self.scope:
            return self.scope[name]
        if name in self.closure:
            return self.closure[name]
        if name in self.globals:
            return self.globals[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        self.check_vanished(name, node)
        raise self.err(E_UNKNOWN_NAME, f"unknown name {name!r}", node)

    def check_vanished(self, name: str, node: ast.AST) -> None:
        """A dynamic value bound inside a loop or branch is gone after it: say so instead of 'unknown name'."""
        if name in self.vanished:
            kind, region = self.vanished[name]
            raise self.err(E_OUT_OF_REGION, f"'{name}' was bound inside the {kind} at line {region.lineno}; a value bound in a {kind} body "
                           f"is not visible after it", node,
                           note=f"keep it in a Var declared before the {kind}: `{name} = Var(0)` there and `{name} <<= ...` inside")

    def contains_dynamic(self, node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if n.id in self.scope and is_dynamic(self.scope[n.id]):
                    return True
            elif isinstance(n, ast.Call):
                try:
                    callee = self.static_eval(n.func) if not self._names_dynamic(n.func) else None
                except CompileError:
                    callee = None
                if callee is None or dsl.rule_of(callee) is not None or isinstance(callee, (dsl.KernelFn, dsl.VfFn, dsl.SimtFn, dsl.InlineFn)):
                    if callee is None and self._names_dynamic(n.func):
                        return True
                    if callee is not None:
                        return True
        return False

    def _names_dynamic(self, node: ast.AST) -> bool:
        return any(isinstance(n, ast.Name) and n.id in self.scope and is_dynamic(self.scope[n.id]) for n in ast.walk(node))

    def static_eval(self, node: ast.AST) -> Any:
        env: dict[str, Any] = dict(self.globals)
        env.update(self.closure)
        env.update({k: v for k, v in self.scope.items() if not is_dynamic(v)})
        try:
            code = compile(ast.Expression(body=node), self.file, "eval")  # type: ignore[arg-type]
            return eval(code, env)  # noqa: S307 - static evaluation by design (RFC-0002 §4)
        except NameError as exc:
            if getattr(exc, "name", None):
                self.check_vanished(exc.name, node)
            raise self.err(E_STATIC_EVAL, f"static evaluation failed: {type(exc).__name__}: {exc}", node) from exc
        except Exception as exc:  # noqa: BLE001
            raise self.err(E_STATIC_EVAL, f"static evaluation failed: {type(exc).__name__}: {exc}", node) from exc

    def ev(self, node: ast.expr) -> Any:
        if not self.contains_dynamic(node):
            try:
                return self.static_eval(node)
            except CompileError as exc:
                # a static-looking call that builds DSL objects (`_cvmutex()` -> CvMutex(...)) cannot run
                # in Python: evaluate it node by node so the helper is compiled inline
                if not (isinstance(node, ast.Call) and _is_marker_failure(exc)):
                    raise
        match node:
            case ast.Name(id=name):
                return self.lookup(name, node)
            case ast.Attribute(value=value, attr=attr):
                return self.attribute(self.ev(value), attr, node)
            case ast.Subscript(value=value, slice=sl):
                return self.subscript(self.ev(value), sl, node)
            case ast.Call(func=func, args=args, keywords=keywords):
                callee = self.ev(func)
                a = []
                for x in args:
                    if isinstance(x, ast.Starred):
                        seq = self.ev(x.value)
                        if not isinstance(seq, (list, tuple)):
                            raise self.err(E_UNSUPPORTED, "*args needs a static container", node)
                        a.extend(seq)
                    else:
                        a.append(self.ev(x))
                kw = {}
                for k in keywords:
                    if k.arg is None:
                        raise self.err(E_UNSUPPORTED, "**kwargs in a call is not supported", node)
                    kw[k.arg] = self.ev(k.value)
                return self.call(callee, a, kw, node)
            case ast.BinOp(left=left, op=op, right=right):
                return self.binop(op, self.ev(left), self.ev(right), node)
            case ast.UnaryOp(op=op, operand=operand):
                return self.unaryop(op, self.ev(operand), node)
            case ast.Compare(left=left, ops=ops, comparators=comps):
                cur = self.ev(left)
                result = None
                for op, comp in zip(ops, comps, strict=True):
                    nxt = self.ev(comp)
                    c = self.compare(op, cur, nxt, node)
                    result = c if result is None else self.emit("scalar.and", (result, c), {}, ScalarType(DTYPES["b1"]), None, node)
                    cur = nxt
                return result
            case ast.BoolOp(op=op, values=values):
                vals = [self.as_bool(self.ev(v), node) for v in values]
                opcode = "scalar.and" if isinstance(op, ast.And) else "scalar.or"
                acc = vals[0]
                for v in vals[1:]:
                    acc = self.emit(opcode, (acc, v), {}, ScalarType(DTYPES["b1"]), None, node)
                return acc
            case ast.IfExp(test=test, body=body, orelse=orelse):
                if not self.contains_dynamic(test):
                    return self.ev(body) if self.static_eval(test) else self.ev(orelse)
                cond = self.as_bool(self.ev(test), node)
                a, b = self.rvalue(self.ev(body), node), self.rvalue(self.ev(orelse), node)
                return self.emit("scalar.select", (cond, a, b), {}, self._promote(a, b, node), None, node)
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                items = [self.ev(e) for e in elts]
                return items if isinstance(node, ast.List) else tuple(items)
            case ast.Constant(value=value):
                return value
            case ast.ListComp(elt=elt, generators=gens) | ast.GeneratorExp(elt=elt, generators=gens):
                return self.comprehension(elt, gens, node)
            case _:
                raise self.err(E_UNSUPPORTED, f"unsupported expression {type(node).__name__} with dynamic operands", node)

    def comprehension(self, elt: ast.expr, gens: list[ast.comprehension], node: ast.AST) -> list[Any]:
        """``[Reg(DT.float) for _ in range(4)]``: static iteration, the element compiled per item."""
        saved = dict(self.scope)
        out: list[Any] = []

        def run(i: int) -> None:
            if i == len(gens):
                out.append(self.ev(elt))
                return
            g = gens[i]
            if not isinstance(g.target, ast.Name):
                raise self.err(E_UNSUPPORTED, "comprehension targets must be plain names", node)
            seq = self.ev(g.iter)
            if is_dynamic(seq) and not isinstance(seq, (list, tuple)):
                raise self.err(E_UNSUPPORTED, "comprehensions iterate over static containers only", node)
            for item in list(seq):
                self.scope[g.target.id] = item
                if all(self._static_true(c) for c in g.ifs):
                    run(i + 1)

        try:
            run(0)
        finally:
            self.scope = saved
        return out

    def _static_true(self, cond: ast.expr) -> bool:
        if self.contains_dynamic(cond):
            raise self.err(E_UNSUPPORTED, "comprehension conditions must be static", cond)
        return bool(self.static_eval(cond))

    # -- attribute / subscript -----------------------------------------------------------------

    def attribute(self, base: Any, attr: str, node: ast.AST) -> Any:
        if isinstance(base, Dyn):
            t = base.type
            if isinstance(t, MemType) and t.space == "gmlist":
                if attr == "count":
                    return self.list_count(base, node)
                if attr == "dtype":
                    return _dtype_name(t.dtype)
                raise self.err(E_BAD_OPERAND, f"a GM tensor list has .count and [index], not .{attr}", node)
            if isinstance(t, MemType):
                return self.tensor_attribute(base, attr, node)
            if attr == "dtype" and isinstance(t, (RegType, ScalarType, CellType)):
                return _dtype_name(t.dtype)
            if attr == "name":
                return base.value.name
            return BoundMethod(base, attr)
        if isinstance(base, (RegExpr, RegList, Img2col)):
            if isinstance(base, RegList) and attr == "dtype":
                return _dtype_name(base.dtype)
            return BoundMethod(base, attr)
        if isinstance(base, ElemOffset):
            if attr in ("single", "brcb", "upsample", "downsample", "unpack", "unpack4"):
                return BoundMethod(base, attr)
            if attr == "dtype":
                return _dtype_name(base.base.type.dtype)
            raise self.err(E_BAD_OPERAND, "an indexed element has no attributes", node)
        try:
            return getattr(base, attr)
        except AttributeError:
            raise self.err(E_UNKNOWN_NAME, f"{base!r} has no attribute {attr!r}", node) from None

    def _dim_value(self, d: Dim, node: ast.AST) -> Any:
        if isinstance(d, int):
            return d
        if isinstance(d, DimValue):
            v = self.named_values.get(d.name)
            if v is None:
                raise self.err(E_UNKNOWN_NAME, f"dimension %{d.name} is not in scope", node)
            return v
        if isinstance(d, Product):
            acc: Any = None
            for f in d.factors:
                fv = self._dim_value(f, node)
                acc = fv if acc is None else self.binop(ast.Mult(), acc, fv, node)
            return acc
        raise self.err(E_BAD_SHAPE, "a '?' dimension must be read through list.item_dim", node)

    def list_count(self, xs: Dyn, node: ast.AST) -> Any:
        """``xs.count`` / ``len(xs)``: the pinned count (``GMList[..., count=n]``) or ``list.count`` (RFC-0002 §3.3)."""
        pinned = getattr(self, "list_counts", {}).get(xs.name)
        if pinned is not None:
            return pinned
        return self.emit("list.count", (xs.plain(),), {}, ScalarType(DTYPES["i32"]), None, node)

    def list_item(self, xs: Dyn, index: Any, node: ast.AST) -> Dyn:
        """``xs[i]``: a ``list.item_dim`` per ``?`` dim, then ``list.item`` typed with those runtime dims."""
        t = xs.type
        assert isinstance(t, MemType)
        dims: list[Dim] = []
        for d, dim in enumerate(t.dims):
            if isinstance(dim, Ragged):
                r = self.emit("list.item_dim", (xs.plain(), index), {"dim": d}, ScalarType(DTYPES["i32"]), None, node)
                dims.append(DimValue(r.name))
            else:
                dims.append(dim)
        return self.emit("list.item", (xs.plain(), index), {}, MemType("gm", t.dtype, tuple(dims)), None, node)

    def _for_list(self, node: ast.For, xs: Dyn) -> None:
        """``for t in xs``: a device loop over ``list.count`` binding ``t`` to ``xs[i]`` (unrolled when the count is pinned)."""
        if isinstance(node.target, ast.Tuple):
            raise self.err(E_UNSUPPORTED, "a loop over a GM tensor list yields one member per iteration", node)
        n = self.list_count(xs, node)
        if isinstance(n, int):
            self._unroll(node, [self.list_item(xs, i, node) for i in range(n)])
            return
        assert self.fb is not None
        with self.fb.loop(0, self._operand(n, node), 1, name=f"{node.target.id}_i", loc=self.loc(node)) as i:
            self.loop_depth += 1
            with self.region_scope(node):
                self.named_values[i.name] = Dyn(i)
                self.bind(node.target.id, self.list_item(xs, Dyn(i), node), node)
                self.compile_body(node.body)
            self.loop_depth -= 1

    def subscript(self, base: Any, sl: ast.expr, node: ast.AST) -> Any:
        if isinstance(base, Img2col):
            return self.img2col_subscript(base, list(sl.elts) if isinstance(sl, ast.Tuple) else [sl], node)
        if isinstance(base, RegList):
            index = self.ev(sl)
            if not isinstance(index, int) or isinstance(index, bool):
                raise self.err(E_UNSUPPORTED, "RegList indices must be static ints (unroll the loop)", node)
            if not -len(base) <= index < len(base):
                raise self.err(E_BAD_OPERAND, f"RegList index {index} out of range for length {len(base)}", node)
            return base.elems[index]
        if not isinstance(base, Dyn):
            index = self.ev(sl)
            if is_dynamic(index):
                raise self.err(E_UNSUPPORTED, "dynamic index into a static container", node)
            try:
                return base[index]
            except Exception as exc:  # noqa: BLE001
                raise self.err(E_STATIC_EVAL, f"static indexing failed: {exc}", node) from None
        t = base.type
        if isinstance(t, BufType):
            if isinstance(sl, ast.Tuple):  # buf[beat, a:b, c:d]: slot selection, then a slice of the slot
                index = self.rvalue(self.ev(sl.elts[0]), node)
                slot = self.get_buf(base, index, node)
                return self.slice_tensor(slot, list(sl.elts[1:]), node)
            index = self.rvalue(self.ev(sl), node)
            return self.get_buf(base, index, node)
        if isinstance(t, MemType) and t.space == "gmlist":
            if isinstance(sl, ast.Tuple):
                raise self.err(E_BAD_OPERAND, "a GM tensor list takes one index", node)
            return self.list_item(base, self.rvalue(self.ev(sl), node), node)
        if isinstance(t, MemType):
            if isinstance(sl, ast.Tuple):
                parts: list[ast.expr] = list(sl.elts)
            else:
                parts = [sl]
            return self.slice_tensor(base, parts, node)
        raise self.err(E_BAD_OPERAND, f"cannot index {_describe(base)}", node)

    # -- calls ---------------------------------------------------------------------------------

    def call(self, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if isinstance(callee, BoundMethod):
            return self.method(callee.obj, callee.name, args, kwargs, node)
        rule = dsl.rule_of(callee)
        if rule is not None:
            return self.rule(rule, callee, args, kwargs, node)
        if not callable(callee):
            raise self.err(E_NOT_CALLABLE, f"{callee!r} is not callable", node)
        if callee is len and len(args) == 1 and isinstance(args[0], Dyn) and isinstance(args[0].type, MemType) and args[0].type.space == "gmlist":
            return self.list_count(args[0], node)
        if callee in _CONTAINER_BUILTINS:  # these only arrange values, never look inside them
            try:
                return callee(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise self.err(E_STATIC_EVAL, f"{callee.__name__} failed: {exc}", node) from exc
        if not any(is_dynamic(a) for a in args) and not any(is_dynamic(v) for v in kwargs.values()):
            try:
                return callee(*args, **kwargs)
            except TypeError as exc:
                # a plain helper that builds DSL objects (`def _cvmutex(): return CvMutex(...)`) cannot run
                # in Python; compile it inline instead
                if inspect.isfunction(callee) and _is_marker_failure(exc):
                    return self.inline(callee, args, kwargs, node)
                raise self.err(E_STATIC_EVAL, f"static call failed: {type(exc).__name__}: {exc}", node) from exc
            except Exception as exc:  # noqa: BLE001
                raise self.err(E_STATIC_EVAL, f"static call failed: {type(exc).__name__}: {exc}", node) from exc
        if inspect.isfunction(callee):
            return self.inline(callee, args, kwargs, node)
        raise self.err(E_UNSUPPORTED, f"cannot call {callee!r} with dynamic arguments", node)

    def inline(self, fn: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        """Inline a plain Python helper called with a dynamic argument (RFC-0002 §4.2)."""
        sub = FunctionCompiler(self.fe, fn, self.kind, self.name)
        sub.fb = self.fb
        sub.named_values = self.named_values
        sub.roots = self.roots
        sub.geoms = self.geoms
        sub.l0_transposed = self.l0_transposed
        sub.atomic = self.atomic
        sub.loop_depth = self.loop_depth
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as exc:
            raise self.err(E_BAD_SIGNATURE, f"cannot bind arguments of {fn.__name__}: {exc}", node) from None
        bound.apply_defaults()
        sub.scope.update(bound.arguments)
        body = sub.tree.body
        try:
            for i, stmt in enumerate(body):
                if isinstance(stmt, ast.Return):
                    if i != len(body) - 1:
                        raise sub.err(E_RETURN_NOT_LAST, "return must be the last statement of an inlined helper", stmt)
                    return sub.ev(stmt.value) if stmt.value is not None else None
                sub.compile_stmt(stmt)
        except _Return as r:
            return r.value
        return None

    def mutex_guards(self, value: Any, node: ast.AST) -> list[tuple[str, int]]:
        """``guards=buf`` / ``guards=(score, p)`` -> [(root name, slot count)].

        The slot count is a static property of the allocation (a plain ``Tensor`` 1, ``DBuff`` 2,
        ``TBuff`` 3, ``QBuff`` 4), so the credit count a mutex needs is knowable here, where the
        author is looking at the buffer. With several buffers the tightest one wins: a credit the
        smallest rotation cannot honour is a credit that lets the producer retake a live slot.

        The names also travel into the IR, where `autosync` otherwise has to INFER which buffer an
        edge belongs to and goes quiet when it cannot.
        """
        if value is None:
            return []
        out: list[tuple[str, int]] = []
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(item, Dyn) and isinstance(item.type, BufType):
                out.append((self.roots.get(item.name, item.name), item.type.slots))
            elif isinstance(item, Dyn) and isinstance(item.type, MemType) and item.type.space != "gm":
                out.append((self.roots.get(item.name, item.name), 1))
            elif isinstance(item, Dyn) and isinstance(item.type, MemType):
                raise self.err(E_BAD_OPERAND, "guards= names an on-chip buffer, not GM; a GM ring is a "
                                              "GMBuff and carries its own depth check", node)
            else:
                raise self.err(E_BAD_OPERAND, "guards= takes the on-chip tensor or buffer this mutex "
                                              f"protects, or a tuple of them; got {item!r}", node)
        return out

    def rule(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if rule == "tensor":
            return self.rule_tensor(args, kwargs, node, slots=None)
        if rule == "buf":
            return self.rule_tensor(args, kwargs, node, slots=callee.slots)
        if rule == "var":
            return self.rule_var(args, kwargs, node)
        if rule == "reg":
            return self.rule_reg(args, kwargs, node)
        if rule == "maskreg":
            return self.rule_maskreg(args, kwargs, node)
        if rule == "reglist":
            return self.rule_reglist(args, kwargs, node)
        if rule.startswith("vf.") or rule == "debug.print_reg":
            return self.rule_vf(rule, callee, args, kwargs, node)
        if rule == "cast":
            return self.rule_vf("vf.cast", callee, args, kwargs, node)
        if rule in ("unalign_load", "unalign_store"):
            role = rule.split("_")[1]
            return self.emit("vf.unalign", (), _name_attr(args, kwargs, 0), UnalignRegType(role), kwargs.get("name") or "ureg", node)
        if rule in ("mutex_vc", "mutex_cv"):
            flag_id = args[0] if args else kwargs.get("flag_id", kwargs.get("id"))
            if not isinstance(flag_id, int):
                raise self.err(E_BAD_OPERAND, "the mutex flag id must be a static int", node)
            attrs: dict[str, Any] = {"kind": Ident(rule.split("_")[1]), "id": flag_id}
            guarded = self.mutex_guards(kwargs.get("guards"), node)
            if len(args) > 1:
                raise self.err(E_BAD_OPERAND, "the credit count is keyword-only: write `depth=" +
                                              f"{args[1]}`, or name the buffer it guards with `guards=`. "
                                              "As a second positional it reads as (id, depth) only to "
                                              "someone who remembers which is which", node)
            given = kwargs.get("depth")
            # `guards` never overrides `depth`: it SUPPLIES it when absent, and the lint compares
            # the two when both are written. One credit per slot is the answer for a mutex that
            # runs once per rotation, which is nearly all of them; a mutex that cycles twice per
            # rotation legitimately carries more, and only the author knows that.
            if given is None and not guarded:
                raise self.err(E_BAD_OPERAND, "a mutex needs its credit count, and there is no default "
                                              "that is right: the value belongs to the buffer being handed "
                                              "over, not to the mutex. Write `guards=<that buffer>` and it "
                                              "is read off the slot count, or `depth=N` to state it. Too "
                                              "many credits lets the producer retake a slot the consumer "
                                              "is still reading: a wrong answer, not a hang", node)
            depth = int(given) if given is not None else min(s for _, s in guarded)
            if depth < 1:
                raise self.err(E_BAD_OPERAND, "a mutex needs at least one credit: with depth=0 the "
                                              "producer's first lock() waits for a free() that can never "
                                              "come, because the consumer has nothing published to free", node)
            attrs["depth"] = depth
            if guarded:
                attrs["guards"] = [root for root, _ in guarded]
            for k in ("src_start_pipe", "dst_start_pipe", "src_end_pipe", "dst_end_pipe"):
                if k in kwargs:
                    attrs[k] = Ident(_enum_name(kwargs[k]))
            return self.emit("sync.mutex", (), attrs, FlagType(None), "mutex", node)
        if rule == "matmul":
            return self.rule_matmul(args, kwargs, node)
        if rule == "matmul_mx":
            return self.rule_matmul_mx(args, kwargs, node)
        if rule == "conv2d":
            return self.rule_conv2d(args, kwargs, node)
        if rule == "img2col":
            return self.rule_img2col(args, kwargs, node)
        if rule.startswith("dma."):
            return self.rule_dma(rule, callee, args, kwargs, node)
        if rule in ("cube.mmad", "cube.mmad.mx"):
            return self.rule_mmad(rule, args, kwargs, node)
        if rule == "workspace":
            return self.rule_workspace(args, kwargs, node)
        if rule == "gmbuff":
            return self.rule_gmbuff(args, kwargs, node)
        if rule == "reinterpret":
            src = args[0] if args else kwargs.get("src")
            if not (isinstance(src, Dyn) and isinstance(src.type, MemType)):
                raise self.err(E_BAD_OPERAND, "reinterpret(src, dtype, name='') needs a tensor", node)
            dt = _need_dtype(args[1] if len(args) > 1 else kwargs.get("target_dtype", kwargs.get("dtype")), self, node)
            return self.reinterpret_view(src, dt, (args[2] if len(args) > 2 else kwargs.get("name")) or "", node)
        if rule.startswith("event:"):
            return self.rule_event(int(rule.split(":")[1]), args, kwargs, node)
        if rule.startswith(("sync.", "barrier", "crosscore:")):
            return self.rule_sync(rule, callee, args, kwargs, node)
        if rule.startswith("debug."):
            return self.rule_debug(rule, callee, args, kwargs, node)
        if rule in ("core.set_hf32", "core.clean_dcache", "noop"):
            return self.rule_core_misc(rule, args, kwargs, node)
        if rule.startswith("atomic:"):
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is used as `with {callee.__name__}():`", node)
        if rule in ("vec.sort32", "vec.mergesort4", "vec.mergesort_2seq"):
            return self.rule_sort(rule, callee, args, kwargs, node)
        if rule.startswith("vec."):
            if self.kind != "kernel":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__} is a kernel-level instruction", node)
            if rule not in ("vec.set_mask", "vec.set_mask_by_count", "vec.set_mask_normal", "vec.reset_mask"):
                return self.rule_vec(rule, callee, args, kwargs, node)  # the a2 tensor-vector instructions (rules_vec)
            attrs: dict[str, Any] = {}
            if rule == "vec.set_mask":
                attrs = {"high": self._attr_value(self.rvalue(args[0], node)), "low": self._attr_value(self.rvalue(args[1], node))}
            elif rule == "vec.set_mask_by_count":
                attrs = {"count": self._attr_value(self.rvalue(args[0], node))}
            return self.emit(rule, (), attrs, None, None, node)
        if rule in ("simt.block_idx", "simt.block_num"):
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__}() is only available inside simt functions", node)
            return self.emit(rule, (), {}, ScalarType(DTYPES["i32"]), None, node)
        if rule.startswith("simt.atomic:"):
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__}() is only available inside simt functions", node)
            target = args[0]
            if isinstance(target, ElemOffset):
                base, index = target.base, target.offset
            elif isinstance(target, Dyn) and isinstance(target.type, MemType):
                base, index = target, 0
            else:
                raise self.err(E_BAD_OPERAND, "the atomic target is tensor[index]", node)
            kind = rule.split(":")[1]
            val = self.rvalue(args[-1], node)
            operands = [base.plain(), index, val]
            if kind == "cas":
                if len(args) != 3:
                    raise self.err(E_BAD_OPERAND, "simt_atomic_cas(tensor[index], compare, value)", node)
                operands.append(self.rvalue(args[1], node))
            elif len(args) != 2:
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}(tensor[index], value)", node)
            return self.emit("simt.atomic", tuple(operands), {"op": Ident(kind)}, ScalarType(base.type.dtype), None, node)
        if rule == "scalar.cast":
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, "cvt() is only available inside simt functions", node)
            v = self.rvalue(args[0], node)
            dt = _need_dtype(args[1], self, node)
            return self.emit("scalar.cast", (v,), {}, ScalarType(dt), None, node)
        if rule.startswith("align:"):
            a = self.rvalue(args[0], node)
            return self.emit("scalar.align", (a,), {"n": int(rule.split(":")[1])}, self._scalar_type(a, node), None, node)
        if rule in ("scalar.ceil_div", "scalar.min", "scalar.max", "scalar.add", "scalar.sub", "scalar.mul", "scalar.div",
                    "scalar.mod", "scalar.and", "scalar.or", "scalar.xor", "scalar.shl", "scalar.shr"):
            a, b = self.rvalue(args[0], node), self.rvalue(args[1], node)
            mode = kwargs.get("rounding", "floor")
            attrs = {}
            if rule in ("scalar.div", "scalar.mod"):
                if mode not in ("floor", "trunc"):
                    raise self.err(E_BAD_OPERAND, "rounding must be 'floor' or 'trunc'", node)
                attrs["rounding"] = Ident(mode)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):  # both static: fold as the operators would
                import operator as _op

                if rule == 'scalar.div' and (isinstance(a, float) or isinstance(b, float)):
                    return a / b
                if rule in ("scalar.div", "scalar.mod") and isinstance(a, int) and isinstance(b, int):
                    from ..ir.scalar_math import integer_divmod
                    return integer_divmod(a, b, mode)[rule == "scalar.mod"]
                fold = {"scalar.ceil_div": lambda x, y: -(-x // y), "scalar.min": min, "scalar.max": max, "scalar.add": _op.add,
                        "scalar.sub": _op.sub, "scalar.mul": _op.mul, "scalar.div": _op.floordiv, "scalar.mod": _op.mod,
                        "scalar.and": _op.and_, "scalar.or": _op.or_, "scalar.xor": _op.xor, "scalar.shl": _op.lshift,
                        "scalar.shr": _op.rshift}
                return fold[rule](a, b)
            return self.emit(rule, (a, b), attrs, self._promote(a, b, node), None, node)
        if rule == "scalar.not":
            a = self.rvalue(args[0], node)
            if isinstance(a, int) and not isinstance(a, bool):
                return ~a
            return self.emit(rule, (a,), {}, self._scalar_type(a, node), None, node)
        if rule in ("scalar.sqrt", "scalar.abs"):
            a = self.rvalue(args[0], node)
            return self.emit(rule, (a,), {}, self._scalar_type(a, node), None, node)
        if rule == "core.set_sat_flag":
            from ..ir.saturation import SAT_BITS

            if len(args) != 2 or not isinstance(args[0], str) or args[0] not in SAT_BITS:
                raise self.err(E_BAD_OPERAND, "set_saturation_flag('float'|'float8'|'int'|'cast'|'global', enable)", node)
            enable = self.rvalue(args[1], node)
            if isinstance(enable, (bool, int)):
                enable = bool(enable)
            elif isinstance(enable, Dyn) and isinstance(enable.type, (ScalarType, CellType)) and enable.type.dtype.is_integer:
                enable = self.as_bool(enable, node)
            else:
                raise self.err(E_BAD_OPERAND, "saturation flag value must be a boolean or integer scalar", node)
            return self.emit(rule, (), {"mode": Ident(args[0]), "enable": enable}, None, None, node)
        if rule == "core.get_sat_flag":
            from ..ir.saturation import SAT_BITS

            if len(args) != 1 or not isinstance(args[0], str) or args[0] not in SAT_BITS:
                raise self.err(E_BAD_OPERAND, "get_saturation_flag('float'|'float8'|'int'|'cast'|'global')", node)
            return self.emit(rule, (), {"mode": Ident(args[0])}, ScalarType(DTYPES["i32"]), None, node)
        if rule.startswith("core."):
            return self.emit(rule, (), {}, ScalarType(DTYPES["i32"]), None, node)
        if rule in ("simt.thread_id", "simt.thread_num"):
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__}() is only available inside simt functions", node)
            return self.emit(rule, (), {}, ScalarType(DTYPES["i32"]), None, node)
        if rule == "simt.barrier":
            return self.emit(rule, (), {}, None, None, node)
        _SIMT_MATH_ARITY = {
            **{f"simt.{n}": (1, "f32") for n in ("exp", "exp2", "log", "log2", "log1p", "sin", "cos",
                                                 "tanh", "rsqrt", "rint", "round", "floor", "ceil", "trunc")},
            **{f"simt.{n}": (1, "i32") for n in ("isnan", "isinf", "isfinite", "popc", "ffs")},
            "simt.fmod": (2, "f32"), "simt.fma": (3, "f32"), "simt.mul_hi": (2, None),
        }
        if rule in _SIMT_MATH_ARITY:
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__}() is only available inside simt functions", node)
            arity, res = _SIMT_MATH_ARITY[rule]
            if len(args) != arity:
                raise self.err(E_BAD_OPERAND, f"{callee.__name__} takes {arity} argument(s)", node)
            vals = tuple(self.rvalue(a, node) for a in args)
            rtype = self._promote(vals[0], vals[1], node) if res is None else ScalarType(DTYPES[res])
            return self.emit(rule, vals, {}, rtype, None, node)
        if rule in ("simt.threadfence", "simt.threadfence_block"):
            if self.kind != "simt":
                raise self.err(E_UNSUPPORTED, f"{callee.__name__}() is only available inside simt functions", node)
            return self.emit(rule, (), {}, None, None, node)
        if rule == "static_print":
            print("[static_print]", *[_describe(a) if is_dynamic(a) else a for a in args])
            return None
        if rule == "static_assert":
            if is_dynamic(args[0]):
                raise self.err(E_STATIC_EVAL, "static_assert needs a static condition", node)
            if not args[0]:
                raise self.err(E_STATIC_EVAL, f"static_assert failed: {args[1] if len(args) > 1 else ''}", node)
            return None
        if rule == "call_vf":
            return self.rule_call(callee, "vf", args, node)
        if rule == "call_simt":
            return self.rule_call(callee, "simt", args, node)
        if rule == "call_inline":
            return self.inline(callee.fn, args, kwargs, node)
        if rule == "composite.radix_topk":
            from .composite import compile_radix_topk

            return compile_radix_topk(self, callee, args, kwargs, node)
        if rule in ("auto_sync", "vec_scope", "cube_scope"):
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is used as `with {callee.__name__}():`", node)
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    def rule_var(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        init = args[0] if args else kwargs.get("value", kwargs.get("init"))
        dt_arg = args[1] if len(args) > 1 else kwargs.get("dtype")
        init = self.rvalue(init, node) if init is not None else None
        if dt_arg is not None:
            dt = _need_dtype(dt_arg, self, node)
        elif isinstance(init, bool):
            dt = DTYPES["b1"]
        elif isinstance(init, int):
            dt = DTYPES["i32"]
        elif isinstance(init, float):
            dt = DTYPES["f32"]
        elif isinstance(init, Dyn):
            dt = self._scalar_type(init, node).dtype
        else:
            raise self.err(E_BAD_OPERAND, "Var takes the initial value first: Var(0), Var(0, DT.int32) "
                                          "or Var(dtype=DT.int32). A dtype alone in the first position "
                                          "is not an initial value", node)
        attrs: dict[str, Any] = {}
        if init is not None:
            attrs["init"] = init.value if isinstance(init, Dyn) else init
        return self.emit("scalar.cell", (), attrs, CellType(dt), kwargs.get("name") or "v", node)

    def rule_call(self, target: Any, kind: str, args: list[Any], node: ast.AST) -> Any:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, f"{kind} functions are called from kernels only", node)
        callee = self.fe.compile_callee(target, kind, args, node, self)
        params = list(inspect.signature(target.fn).parameters)
        operands: list[Any] = [FuncRef(callee.name)]
        reads: list[Value] = []
        writes: list[Value] = []
        for pname, a in zip(params, args, strict=True):
            if pname in callee.static_bindings:
                continue
            operands.append(a)
            if isinstance(a, Dyn) and isinstance(a.type, MemType):
                mode = callee.access.get(pname)
                if mode in ("read", "readwrite"):
                    reads.append(a.value)
                if mode in ("write", "readwrite"):
                    writes.append(a.value)
        operands.extend(Dyn(v) for v in callee.implicit_values)
        attrs: dict[str, Any] = {}
        if reads:
            attrs["read"] = reads
        if writes:
            attrs["write"] = writes
        if kind == "simt":
            attrs = {"threads": target.num_threads, **attrs}
            return self.emit("simt.launch", tuple(operands), attrs, None, None, node)
        return self.emit("cf.call", tuple(operands), attrs, None, None, node)

    # -- methods on dynamic values -------------------------------------------------------------

    def method(self, obj: Any, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if isinstance(obj, ElemOffset):
            return ElemOffset(self.tensor_method(obj.base, name, args, kwargs, node), obj.offset)
        t = obj.type if isinstance(obj, Dyn) else None
        if isinstance(t, (RegType, MaskType)) or isinstance(obj, (RegList, RegExpr)):
            return self.reg_method(obj, name, args, kwargs, node)
        if isinstance(t, FlagType) and name in _MUTEX_METHODS:
            return self.emit(_MUTEX_METHODS[name], (obj,), {}, None, None, node)
        if isinstance(t, CellType):
            return self.cell_method(obj, name, args, kwargs, node)
        if isinstance(t, EventType):
            return self.event_method(obj, name, args, node)
        if isinstance(t, MemType):
            return self.tensor_method(obj, name, args, kwargs, node)
        if isinstance(obj, Img2col) and name == "window":
            return obj.window(*[self.rvalue(a, node) for a in args])
        raise self.err(E_UNSUPPORTED, f"{_describe(obj)} has no method {name!r}", node)

    # -- arithmetic ----------------------------------------------------------------------------

    def binop(self, op: ast.operator, lhs: Any, rhs: Any, node: ast.AST) -> Any:
        if not is_dynamic(lhs) and not is_dynamic(rhs):
            return _py_binop(op, lhs, rhs)
        if self._is_reg_like(lhs) or self._is_reg_like(rhs):
            return self.reg_binop(op, lhs, rhs, node)
        l, r = self.rvalue(lhs, node), self.rvalue(rhs, node)  # noqa: E741
        if type(op) not in _SCALAR_BINOPS:
            raise self.err(E_UNSUPPORTED, f"operator {type(op).__name__} is not supported on scalars", node)
        rtype = self._promote(l, r, node)
        return self.emit(f"scalar.{_SCALAR_BINOPS[type(op)]}", (l, r), {}, rtype, None, node)

    @staticmethod
    def _is_reg_like(v: Any) -> bool:
        return isinstance(v, (RegExpr, RegList)) or (isinstance(v, Dyn) and isinstance(v.type, (RegType, MaskType)))

    def unaryop(self, op: ast.unaryop, v: Any, node: ast.AST) -> Any:
        if not is_dynamic(v):
            return _py_unary(op, v)
        if isinstance(op, ast.Invert) and isinstance(v, Dyn) and isinstance(v.type, MaskType):
            return RegExpr("mask_not", (v,))
        if isinstance(op, ast.USub) and self._is_reg_like(v):
            return self.reg_binop(ast.Mult(), v, -1, node)
        v = self.rvalue(v, node)
        if isinstance(op, ast.USub):
            return self.emit("scalar.neg", (v,), {}, self._scalar_type(v, node), None, node)
        if isinstance(op, (ast.Not, ast.Invert)):
            return self.emit("scalar.not", (v,), {}, self._scalar_type(v, node), None, node)
        if isinstance(op, ast.UAdd):
            return v
        raise self.err(E_UNSUPPORTED, "unsupported unary operator", node)

    def compare(self, op: ast.cmpop, lhs: Any, rhs: Any, node: ast.AST) -> Any:
        if not is_dynamic(lhs) and not is_dynamic(rhs):
            return _py_compare(op, lhs, rhs)
        if self._is_reg_like(lhs) or self._is_reg_like(rhs):
            return self.reg_compare(op, lhs, rhs, node)
        if type(op) not in _CMP_PREDS:
            raise self.err(E_UNSUPPORTED, f"comparison {type(op).__name__} is not supported on scalars", node)
        l, r = self.rvalue(lhs, node), self.rvalue(rhs, node)  # noqa: E741
        return self.emit("scalar.cmp", (l, r), {"pred": Ident(_CMP_PREDS[type(op)])}, ScalarType(DTYPES["b1"]), None, node)

    def as_bool(self, v: Any, node: ast.AST) -> Any:
        v = self.rvalue(v, node)
        if isinstance(v, Dyn):
            t = self._scalar_type(v, node)
            if t.dtype.name == "b1":
                return v
            return self.emit("scalar.cmp", (v, 0), {"pred": Ident("ne")}, ScalarType(DTYPES["b1"]), None, node)
        return bool(v)

    def rvalue(self, v: Any, node: ast.AST) -> Any:
        """An expression operand: element offsets become loads in simt; everything else passes through."""
        if isinstance(v, ElemOffset):
            if self.kind == "simt":
                bt = v.base.type
                assert isinstance(bt, MemType)
                return self.emit("simt.load", (v.base, v.offset), {}, ScalarType(bt.dtype), None, node)
            raise self.err(E_BAD_OPERAND, "an indexed element can only be the source or target of `<<=` here", node)
        if isinstance(v, BoundMethod):
            raise self.err(E_BAD_OPERAND, f"method {v.name!r} must be called", node)
        return v

    def _scalar_type(self, v: Any, node: ast.AST) -> ScalarType:
        if isinstance(v, Dyn):
            if isinstance(v.type, ScalarType):
                return v.type
            if isinstance(v.type, CellType):
                return ScalarType(v.type.dtype)
            raise self.err(E_BAD_OPERAND, f"expected a scalar, got {_describe(v)}", node)
        if isinstance(v, bool):
            return ScalarType(DTYPES["b1"])
        if isinstance(v, int):
            return ScalarType(DTYPES["i32"])
        if isinstance(v, float):
            return ScalarType(DTYPES["f32"])
        raise self.err(E_BAD_OPERAND, f"expected a scalar, got {v!r}", node)

    def _promote(self, a: Any, b: Any, node: ast.AST) -> ScalarType:
        ta, tb = self._scalar_type(a, node), self._scalar_type(b, node)
        if ta.dtype.is_float or tb.dtype.is_float:
            fa = ta if ta.dtype.is_float else tb
            fb = tb if tb.dtype.is_float else ta
            return fa if fa.dtype.bits >= fb.dtype.bits else fb
        if ta.dtype.name == "b1" and tb.dtype.name == "b1":
            return ta
        # literals adopt the dynamic operand's integer type
        if not isinstance(a, Dyn):
            return tb
        if not isinstance(b, Dyn):
            return ta
        return ta if ta.dtype.bits >= tb.dtype.bits else tb

    # -- emitting ------------------------------------------------------------------------------

    def emit(self, opcode: str, operands: tuple[Any, ...], attrs: dict[str, Any], result: Type | None, name: str | None,
             node: ast.AST | None) -> Any:
        assert self.fb is not None
        ops = tuple(self._operand(x, node) for x in operands)
        clean = {k: self._attr_value(v) for k, v in attrs.items() if v is not None}
        res = self.fb.op(opcode, ops, clean, result, name, loc=self.loc(node))
        if res is None:
            return None
        assert isinstance(res, Value)
        d = Dyn(res)
        self.named_values[res.name] = d
        return d

    def _operand(self, x: Any, node: ast.AST | None) -> Any:
        if isinstance(x, Dyn):
            return x.value
        if isinstance(x, (Value, Literal, FuncRef)):
            return x
        if isinstance(x, (bool, int, float, complex)):
            return x
        if isinstance(x, ElemOffset):
            raise self.err(E_BAD_OPERAND, "an indexed element cannot be used here", node)
        raise self.err(E_BAD_OPERAND, f"cannot use {x!r} as an operand", node)

    def _attr_value(self, v: Any) -> Any:
        if isinstance(v, Dyn):
            return v.value
        if isinstance(v, dsl.EnumValue):
            return Ident(v.name)
        if isinstance(v, dsl.DTypeName):
            return Ident(v.ir.name)
        if isinstance(v, (list, tuple)):
            return [self._attr_value(x) for x in v]
        return v


class _RegionScope:
    def __init__(self, fc: FunctionCompiler, node: ast.AST) -> None:
        self.fc = fc
        self.node = node

    def __enter__(self) -> None:
        self.before = dict(self.fc.scope)

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is not None:
            return
        after = self.fc.scope
        for name, old in self.before.items():
            if name in after and after[name] is not old and not (isinstance(old, Dyn) and isinstance(old.type, CellType)):
                if is_dynamic(after[name]) or is_dynamic(old):
                    raise self.fc.err(E_CONDITIONAL_REBIND, f"'{name}' is rebound inside a loop or branch; use a Var for values that change",
                                      self.node, note="a value bound inside a region is not visible after it")
        kind = "loop" if isinstance(self.node, ast.For) else "branch" if isinstance(self.node, ast.If) else "block"
        for name, v in after.items():
            if name not in self.before and is_dynamic(v):
                self.fc.vanished[name] = (kind, self.node)  # remembered for the diagnostic on a later use
        self.fc.scope = self.before


# --------------------------------------------------------------------------- helpers


def _aclnn_parameter_name(name: str) -> str:
    """The parameter spelling emitted by CANN's ACLNN API generator (M10-080)."""
    parts = name.split("_")
    camel = parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:] if part)
    return camel[:1].lower() + camel[1:]


def _parse_symbol_dim(text: str, symbols: dict[str, ast.AST], node: ast.AST, fc: FunctionCompiler) -> Dim:
    """A signature dimension string: a symbol, an int, or a product such as ``"M*K"`` (D-014)."""
    ts = TokenStream(tokenize(text.replace(" ", "")))
    factors: list[Any] = []
    while True:
        if ts.at("int"):
            factors.append(int(ts.advance().text))
        elif ts.at("ident"):
            sym = ts.advance().text
            symbols.setdefault(sym, node)
            factors.append(DimValue(sym))
        else:
            raise fc.err(E_BAD_SHAPE, f"bad dimension {text!r}: use a symbol, an int, or a product like 'M*K'", node)
        if ts.at_punct("*"):
            ts.advance()
            continue
        break
    if not ts.at("eof"):
        raise fc.err(E_BAD_SHAPE, f"bad dimension {text!r}: only multiplication is allowed", node)
    try:
        parse_dim(TokenStream(tokenize(text.replace(" ", "").replace("*", "*%").replace("%%", "%") if False else "1")))
    except Exception:  # pragma: no cover - the grammar above already validated the text
        pass
    return factors[0] if len(factors) == 1 else Product(tuple(factors))


def _need_dtype(v: Any, fc: FunctionCompiler, node: ast.AST) -> DType:
    if isinstance(v, dsl.DTypeName):
        return v.ir
    if isinstance(v, DType):
        return v
    raise fc.err(E_BAD_OPERAND, f"expected a dtype such as DT.float, got {v!r}", node)


def _dtype_name(dt: DType) -> dsl.DTypeName:
    for attr in vars(dsl._DT):
        val = getattr(dsl.DT, attr, None)
        if isinstance(val, dsl.DTypeName) and val.ir == dt:
            return val
    return dsl.DTypeName(dt.name, dt)


def _name_attr(args: list[Any], kwargs: dict[str, Any], pos: int) -> dict[str, Any]:
    name = args[pos] if len(args) > pos else kwargs.get("name")
    return {"name": name} if name else {}


def _enum_name(v: Any) -> str:
    return v.name if isinstance(v, dsl.EnumValue) else str(v)


def _describe(v: Any) -> str:
    if isinstance(v, Dyn):
        return f"{v.value} : {v.type}"
    if isinstance(v, ElemOffset):
        return f"{v.base.value}[{v.offset if not isinstance(v.offset, Dyn) else v.offset.value}]"
    return repr(v)


def _py_binop(op: ast.operator, lhs: Any, rhs: Any) -> Any:
    import operator as o

    table = {ast.Add: o.add, ast.Sub: o.sub, ast.Mult: o.mul, ast.Div: o.truediv, ast.FloorDiv: o.floordiv, ast.Mod: o.mod,
             ast.Pow: o.pow, ast.LShift: o.lshift, ast.RShift: o.rshift, ast.BitAnd: o.and_, ast.BitOr: o.or_, ast.BitXor: o.xor}
    return table[type(op)](lhs, rhs)


def _py_unary(op: ast.unaryop, v: Any) -> Any:
    if isinstance(op, ast.USub):
        return -v
    if isinstance(op, ast.UAdd):
        return +v
    if isinstance(op, ast.Not):
        return not v
    return ~v


def _py_compare(op: ast.cmpop, lhs: Any, rhs: Any) -> Any:
    import operator as o

    table = {ast.Lt: o.lt, ast.LtE: o.le, ast.Gt: o.gt, ast.GtE: o.ge, ast.Eq: o.eq, ast.NotEq: o.ne, ast.In: lambda a, b: a in b,
             ast.NotIn: lambda a, b: a not in b, ast.Is: o.is_, ast.IsNot: o.is_not}
    return table[type(op)](lhs, rhs)


# FunctionBuilder needs a preview of the function under construction for access-set analysis.
def _finish_preview(self: FunctionBuilder):
    from ..ir.core import Block, Function

    return Function(self.kind, self.name, tuple(self.params.values()), self.attrs, Block(tuple(self._stack[0])))


FunctionBuilder.finish_preview = _finish_preview  # type: ignore[attr-defined]

_ = (replace, Op)  # re-exported names kept for type checkers

__all__ = ["compile_kernel", "Frontend", "FunctionCompiler", "Dyn", "ElemOffset", "CompileError"]
