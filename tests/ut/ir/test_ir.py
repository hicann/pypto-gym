# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
import pypto
from pypto import ir


def test_error_types():
    with pytest.raises(ValueError):
        pypto.raise_error("ValueError", "test value error")
    with pytest.raises(TypeError):
        pypto.raise_error("TypeError", "test type error")
    with pytest.raises(RuntimeError):
        pypto.raise_error("RuntimeError", "test runtime error")
    with pytest.raises(NotImplementedError):
        pypto.raise_error("NotImplementedError", "test not implemented error")
    with pytest.raises(IndexError):
        pypto.raise_error("IndexError", "test index error")
    with pytest.raises(AssertionError):
        pypto.raise_error("AssertionError", "test assertion error")
    with pytest.raises(pypto.InternalError):
        pypto.raise_error("InternalError", "test internal error")


def test_dtypes():
    dtypes = [
        # bits, is_signed, is_unsigned, is_float, name, c_type
        (ir.BOOL, 8, False, False, False, "bool", "bool"),
        (ir.INT4, 4, True, False, False, "int4", "unknown"),
        (ir.INT8, 8, True, False, False, "int8", "int8_t"),
        (ir.INT16, 16, True, False, False, "int16", "int16_t"),
        (ir.INT32, 32, True, False, False, "int32", "int32_t"),
        (ir.INT64, 64, True, False, False, "int64", "int64_t"),
        (ir.INDEX, 64, True, False, False, "index", "int64_t"),
        (ir.UINT4, 4, False, True, False, "uint4", "unknown"),
        (ir.UINT8, 8, False, True, False, "uint8", "uint8_t"),
        (ir.UINT16, 16, False, True, False, "uint16", "uint16_t"),
        (ir.UINT32, 32, False, True, False, "uint32", "uint32_t"),
        (ir.UINT64, 64, False, True, False, "uint64", "uint64_t"),
        (ir.FP4, 4, False, False, True, "fp4", "unknown"),
        (ir.FP8E4M3FN, 8, False, False, True, "fp8e4m3fn", "unknown"),
        (ir.FP8E5M2, 8, False, False, True, "fp8e5m2", "unknown"),
        (ir.FP16, 16, False, False, True, "fp16", "half"),
        (ir.FP32, 32, False, False, True, "fp32", "float"),
        (ir.FP64, 64, False, False, True, "fp64", "double"),
        (ir.BF16, 16, False, False, True, "bfloat16", "bfloat16"),
        (ir.HF4, 4, False, False, True, "hf4", "unknown"),
        (ir.HF8, 8, False, False, True, "hf8", "unknown"),
    ]
    for dtype, *info in dtypes:
        assert info == [dtype.bits(), dtype.is_signed(),
                        dtype.is_unsigned(), dtype.is_float(),
                        str(dtype), dtype.c_type()]


def test_span():
    span = ir.Span("span", 1, 2, 3, 4)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == 3
    assert span.end_column == 4
    assert not span.is_unknown()

    span = ir.Span("span", 1, 2)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == -1
    assert span.end_column == -1
    assert not span.is_unknown()


def test_logging(capfd):
    pypto.set_log_level(pypto.LogLevel.INFO)
    assert pypto.get_log_level() == pypto.LogLevel.INFO

    pypto.log_debug("test debug message")
    pypto.log_info("test info message")
    pypto.log_warn("test warn message")
    pypto.log_error("test error message")
    pypto.log_event("test event message")
    pypto.log_fatal("test fatal message")

    captured = capfd.readouterr()
    assert "event" in captured.err
    assert "fatal" in captured.err
    assert "error" in captured.err
    assert "warn" in captured.err
    assert "info" in captured.err
    assert "debug" not in captured.err


def test_check():
    with pytest.raises(ValueError, match="test check message"):
        pypto.check(False, "test check message")
    pypto.check(True, "test check message")

    with pytest.raises(pypto.InternalError, match="test internal check message"):
        pypto.internal_check(False, "test internal check message")
    pypto.internal_check(True, "test internal check message")


def test_basic_types():
    # UnknownType
    ut = ir.UnknownType()
    assert str(ut) == "ir.Unknown"

    # ScalarType
    st = ir.ScalarType(ir.INT32)
    assert st.dtype == ir.INT32
    assert str(st) == "ir.Scalar[ir.INT32]"

    # TensorType with int64_t shape
    tt = ir.TensorType([16, 32], ir.FP32)
    assert tt.dtype == ir.FP32
    assert len(tt.shape) == 2
    assert tt.memref is None
    assert str(tt) == "ir.Tensor[[16, 32], ir.FP32]"

    # TensorType with memref
    offset = ir.ConstInt(0, ir.INT64, ir.Span("test", 1, 1))
    memref = ir.MemRef(ir.MemorySpace.DDR, offset, 1024)
    tt2 = ir.TensorType([16, 32], ir.FP16, memref)
    assert tt2.memref is not None
    assert tt2.memref.size == 1024
    assert str(tt2) == "ir.Tensor[[16, 32], ir.FP16, ir.MemRef(ir.MemorySpace.DDR, 0, 1024)]"

    # TupleType
    tup = ir.TupleType([ir.ScalarType(ir.INT32), ir.ScalarType(ir.FP32)])
    assert len(tup.types) == 2
    assert str(tup) == "ir.Tuple[ir.Scalar[ir.INT32], ir.Scalar[ir.FP32]]"

    pt = ir.PtrType()
    assert str(pt) == "ir.Ptr"

    # TokenType
    tt3 = ir.TokenType()
    assert str(tt3) == "ir.Token"

    # LogicalTensorType
    lt = ir.LogicalTensorType()
    assert str(lt) == "ir.LogicalTensor"


def test_basic_expr():
    span = ir.Span("test", 1, 1)
    st = ir.ScalarType(ir.INT32)

    # ConstInt
    ci = ir.ConstInt(42, ir.INT32, span)
    assert str(ci) == "42"

    # ConstFloat
    cf = ir.ConstFloat(3.14, ir.FP32, span)
    assert str(cf) == "3.14"

    # ConstBool
    cb = ir.ConstBool(True, span)
    assert str(cb) == "True"

    # Var
    var = ir.Var("x", st, span)
    assert str(var) == "x"
    assert var.name == "x"

    # IterArg
    init_val = ir.ConstInt(0, ir.INT32, span)
    iter_arg = ir.IterArg("acc", st, init_val, span)
    assert str(iter_arg) == "acc"
    assert iter_arg.name == "acc"
    assert isinstance(iter_arg.initValue, ir.ConstInt)
    assert iter_arg.initValue.value == 0

    # Binary expressions
    for bop in [
        (ir.Add, '+'),
        (ir.Sub, '-'),
        (ir.Mul, '*'),
        (ir.FloorDiv, '//'),
        (ir.FloatDiv, '/'),
        (ir.FloorMod, '%'),
        (ir.Pow, '**'),
        (ir.Eq, '=='),
        (ir.Ne, '!='),
        (ir.Lt, '<'),
        (ir.Le, '<='),
        (ir.Gt, '>'),
        (ir.Ge, '>='),
        (ir.And, 'and'),
        (ir.Or, 'or'),
        (ir.Xor, 'xor'),
        (ir.BitAnd, '&'),
        (ir.BitOr, '|'),
        (ir.BitXor, '^'),
        (ir.BitShiftLeft, '<<'),
        (ir.BitShiftRight, '>>'),
    ]:
        a = ir.ConstInt(1, ir.INT32, span)
        b = ir.ConstInt(2, ir.INT32, span)
        expr = bop[0](a, b, ir.INT32, span)
        assert str(expr) == f"1 {bop[1]} 2"

    a = ir.ConstInt(1, ir.INT32, span)
    b = ir.ConstInt(2, ir.INT32, span)
    expr = ir.Min(a, b, ir.INT32, span)
    assert str(expr) == "ir.min(1, 2)"

    a = ir.ConstInt(1, ir.INT32, span)
    b = ir.ConstInt(2, ir.INT32, span)
    expr = ir.Max(a, b, ir.INT32, span)
    assert str(expr) == "ir.max(1, 2)"

    # Unary expressions
    neg = ir.Neg(a, ir.INT32, span)
    assert str(neg) == "-1"

    not_expr = ir.Not(cb, ir.BOOL, span)
    assert str(not_expr) == "not True"

    abs_expr = ir.Abs(a, ir.INT32, span)
    assert str(abs_expr) == "ir.abs(1)"

    cast = ir.Cast(a, ir.FP32, span)
    assert str(cast) == "ir.cast(1, ir.FP32)"

    # MakeTuple and TupleGetItemExpr
    mt = ir.MakeTuple([a, b], span)
    assert str(mt) == "[1, 2]"
    tgi = ir.TupleGetItem(mt, 0, span)
    assert str(tgi) == "[1, 2][0]"

    # Call with Op
    call = ir.Call("my_op", [a, b], span)
    assert str(call) == "ir.call @my_op(1, 2)"

    # MemRef
    offset = ir.ConstInt(0, ir.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Vec, offset, 2048, span)
    assert str(memref) == "ir.MemRef(ir.MemorySpace.Vec, 0, 2048)"


def test_basic_stmt():
    span = ir.Span("test", 1, 1)
    st = ir.ScalarType(ir.INT32)

    # Helper variables and expressions
    x = ir.Var("x", st, span)
    y = ir.Var("y", st, span)
    val42 = ir.ConstInt(42, ir.INT32, span)
    val0 = ir.ConstInt(0, ir.INT32, span)
    val1 = ir.ConstInt(1, ir.INT32, span)
    val10 = ir.ConstInt(10, ir.INT32, span)

    # AssignStmt
    assign = ir.AssignStmt(x, val42, span)
    assert str(assign) == "x: ir.Scalar[ir.INT32] = 42"

    # SeqStmts
    assign_x = ir.AssignStmt(x, val42, span)
    assign_y = ir.AssignStmt(y, val0, span)
    seq = ir.SeqStmts([assign_x, assign_y], span)
    assert str(seq) == "\n".join([
        "x: ir.Scalar[ir.INT32] = 42",
        "y: ir.Scalar[ir.INT32] = 0"
    ])

    # IfStmt
    cond = ir.ConstBool(True, span)
    if_stmt = ir.IfStmt(cond, assign_x, None, [], span)
    assert str(if_stmt) == "\n".join([
        "if True:",
        "    x: ir.Scalar[ir.INT32] = 42"
    ])

    if_else = ir.IfStmt(cond, assign_x, assign_y, [], span)
    assert str(if_else) == "\n".join([
        "if True:",
        "    x: ir.Scalar[ir.INT32] = 42",
        "else:",
        "    y: ir.Scalar[ir.INT32] = 0"
    ])

    # ForStmt
    i = ir.Var("i", st, span)
    init = ir.ConstInt(0, ir.INT32, span)
    iter_arg = ir.IterArg("sum", st, init, span)
    ret_var = ir.Var("sum_out", st, span)
    body = ir.YieldStmt([val1], span)
    for_stmt = ir.ForStmt(i, val0, val10, val1, [iter_arg], body, [ret_var], span)
    assert str(for_stmt) == "\n".join([
        "for i, (sum,) in ir.range(0, 10, 1, init_values=(0,)):",
        "    sum_out: ir.Scalar[ir.INT32] = ir.yield_(1)"
    ])

    # WhileStmt
    while_stmt = ir.WhileStmt(cond, [], assign_x, [], span)
    assert str(while_stmt) == "\n".join([
        "while True:",
        "    x: ir.Scalar[ir.INT32] = 42"
    ])

    # YieldStmt and ReturnStmt
    yield_stmt = ir.YieldStmt([val42], span)
    assert str(yield_stmt) == "ir.yield_(42)"
    empty_yield = ir.YieldStmt(span)
    assert str(empty_yield) == "ir.yield_()"

    return_stmt = ir.ReturnStmt([val42], span)
    assert str(return_stmt) == "return 42"
    empty_return = ir.ReturnStmt(span)
    assert str(empty_return) == "return"

    # BreakStmt and ContinueStmt
    break_stmt = ir.BreakStmt(span)
    assert str(break_stmt) == "break"
    continue_stmt = ir.ContinueStmt(span)
    assert str(continue_stmt) == "continue"

    # EvalStmt
    call = ir.Call("some_op", [val42], span)
    eval_stmt = ir.EvalStmt(call, span)
    assert str(eval_stmt) == "ir.eval(ir.call @some_op(42))"

    # Function
    func = ir.Function("test_func", [x], [st], assign_x, span)
    assert str(func) == "\n".join([
        "@ir.function",
        "def test_func(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:",
        "    x: ir.Scalar[ir.INT32] = 42"
    ])

    # Program
    func2 = ir.Function("test_func2", [x], [st], assign_x, span)
    prog = ir.Program([func, func2], "test_prog", span)
    assert str(prog) == "\n".join([
        "# ir.program: test_prog",
        "@ir.function",
        "def test_func(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:",
        "    x: ir.Scalar[ir.INT32] = 42",
        "@ir.function",
        "def test_func2(x: ir.Scalar[ir.INT32]) -> ir.Scalar[ir.INT32]:",
        "    x: ir.Scalar[ir.INT32] = 42",
    ])
    assert prog["test_func"] is not None
