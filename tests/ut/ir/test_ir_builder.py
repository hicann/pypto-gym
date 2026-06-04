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


def _span():
    return ir.Span("test", 1, 1)


# ---------- Context state queries ----------


def test_builder_context_state():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    assert not b.in_function()
    assert not b.in_loop()
    assert not b.in_if()
    assert not b.in_program()

    b.begin_function("f", sp)
    assert b.in_function()
    assert not b.in_loop()
    assert not b.in_if()
    assert not b.in_program()

    i = b.var("i", st, sp)
    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    assert b.in_function()
    assert b.in_loop()
    assert not b.in_if()
    b.end_for_loop(sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    assert b.in_function()
    assert b.in_if()
    b.end_if(sp)

    b.end_function(sp)

    b.begin_program("prog", sp)
    assert b.in_program()
    b.end_program(sp)


# ---------- Function building ----------


def test_builder_empty_function():
    b = ir.IRBuilder()
    sp = _span()

    b.begin_function("empty_func", sp)
    func = b.end_function(sp)

    assert isinstance(func, ir.Function)
    assert func.name == "empty_func"
    assert len(func.params) == 0
    assert len(func.return_types) == 0


def test_builder_function_with_params_and_returns():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("add_func", sp)
    x = b.func_arg("x", st, sp)
    y = b.func_arg("y", st, sp)
    b.return_type(st)
    b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    func = b.end_function(sp)

    assert func.name == "add_func"
    assert len(func.params) == 2
    assert str(func.params[0]) == "x"
    assert str(func.params[1]) == "y"
    assert len(func.return_types) == 1


def test_builder_function_str():
    """Builder-constructed function should match manually constructed one."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("test_func", sp)
    x = b.func_arg("x", st, sp)
    b.return_type(st)
    b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    built_func = b.end_function(sp)

    manual_x = ir.Var("x", st, sp)
    manual_assign = ir.AssignStmt(manual_x, ir.ConstInt(42, ir.INT32, sp), sp)
    manual_func = ir.Function("test_func", [manual_x], [st], manual_assign, sp)

    assert str(built_func) == str(manual_func)


# ---------- Statement helpers ----------


def test_builder_var():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    v = b.var("tmp", st, sp)
    assert isinstance(v, ir.Var)
    assert v.name == "tmp"


def test_builder_assign():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    stmt = b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    b.end_function(sp)

    assert isinstance(stmt, ir.AssignStmt)
    assert str(stmt) == "x: ir.Scalar[ir.INT32] = 42"


def test_builder_return():
    b = ir.IRBuilder()
    sp = _span()

    # Return with values
    b.begin_function("f", sp)
    val = ir.ConstInt(42, ir.INT32, sp)
    stmt = b.return_([val], sp)
    assert isinstance(stmt, ir.ReturnStmt)
    b.end_function(sp)

    # Empty return
    b.begin_function("g", sp)
    stmt2 = b.return_(sp)
    assert isinstance(stmt2, ir.ReturnStmt)
    b.end_function(sp)


def test_builder_emit():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    b.func_arg("x", st, sp)
    call = ir.Call("some_op", [ir.ConstInt(42, ir.INT32, sp)], sp)
    b.emit(ir.EvalStmt(call, sp))
    func = b.end_function(sp)

    assert isinstance(func.body[0], ir.EvalStmt)


# ---------- For loop building ----------


def test_builder_for_loop():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    i = b.var("i", st, sp)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    for_stmt = b.end_for_loop(sp)

    assert isinstance(for_stmt, ir.ForStmt)
    assert str(for_stmt.loop_var) == "i"
    b.end_function(sp)


def test_builder_for_loop_with_iter_args():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    i = b.var("i", st, sp)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )

    init_val = ir.ConstInt(0, ir.INT32, sp)
    iter_arg = ir.IterArg("sum", st, init_val, sp)
    b.add_iter_arg(iter_arg)

    ret_var = b.var("sum_out", st, sp)
    b.add_return_var(ret_var)

    b.emit(ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp))
    for_stmt = b.end_for_loop(sp)

    assert isinstance(for_stmt, ir.ForStmt)
    assert len(for_stmt.iter_args) == 1
    assert len(for_stmt.return_vars) == 1
    b.end_function(sp)


def test_builder_for_loop_str():
    """Builder-constructed for loop should match manually constructed one."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    i = b.var("i", st, sp)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    init_val = ir.ConstInt(0, ir.INT32, sp)
    iter_arg = ir.IterArg("sum", st, init_val, sp)
    b.add_iter_arg(iter_arg)
    ret_var = ir.Var("sum_out", st, sp)
    b.add_return_var(ret_var)
    b.emit(ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp))
    built_for = b.end_for_loop(sp)
    b.end_function(sp)

    manual_i = ir.Var("i", st, sp)
    manual_iter_arg = ir.IterArg("sum", st, ir.ConstInt(0, ir.INT32, sp), sp)
    manual_ret_var = ir.Var("sum_out", st, sp)
    manual_for = ir.ForStmt(
        manual_i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        [manual_iter_arg],
        ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp),
        [manual_ret_var],
        sp,
    )

    assert str(built_for) == str(manual_for)


# ---------- While loop building ----------


def test_builder_while_loop():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)

    b.begin_while_loop(ir.ConstBool(True, sp), sp)
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    while_stmt = b.end_while_loop(sp)

    assert isinstance(while_stmt, ir.WhileStmt)
    b.end_function(sp)


def test_builder_while_loop_with_iter_args():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)

    b.begin_while_loop(ir.ConstBool(True, sp), sp)

    init_val = ir.ConstInt(0, ir.INT32, sp)
    iter_arg = ir.IterArg("sum", st, init_val, sp)
    b.add_while_iter_arg(iter_arg)

    ret_var = b.var("sum_out", st, sp)
    b.add_while_return_var(ret_var)

    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    while_stmt = b.end_while_loop(sp)

    assert isinstance(while_stmt, ir.WhileStmt)
    assert len(while_stmt.iter_args) == 1
    assert len(while_stmt.return_vars) == 1
    b.end_function(sp)


def test_builder_while_loop_set_condition():
    """Test updating the while loop condition after setting up iter args."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)

    # Begin with a placeholder condition
    b.begin_while_loop(ir.ConstBool(True, sp), sp)

    init_val = ir.ConstInt(0, ir.INT32, sp)
    iter_arg = ir.IterArg("cnt", st, init_val, sp)
    b.add_while_iter_arg(iter_arg)

    # Update the condition
    new_cond = ir.ConstBool(False, sp)
    b.set_while_loop_condition(new_cond)

    # Must add matching return var for the iter_arg
    ret_var = b.var("cnt_out", st, sp)
    b.add_while_return_var(ret_var)

    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    while_stmt = b.end_while_loop(sp)

    assert isinstance(while_stmt, ir.WhileStmt)
    assert len(while_stmt.iter_args) == 1
    assert len(while_stmt.return_vars) == 1
    b.end_function(sp)


# ---------- If statement building ----------


def test_builder_if():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    if_stmt = b.end_if(sp)

    assert isinstance(if_stmt, ir.IfStmt)
    assert if_stmt.else_body is None
    b.end_function(sp)


def test_builder_if_else():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    b.begin_else(sp)
    b.assign(x, ir.ConstInt(0, ir.INT32, sp), sp)
    if_stmt = b.end_if(sp)

    assert isinstance(if_stmt, ir.IfStmt)
    assert if_stmt.else_body is not None
    b.end_function(sp)


def test_builder_if_with_return_vars():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    ret_var = b.var("out", st, sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    b.add_if_return_var(ret_var)
    b.assign(x, ir.ConstInt(42, ir.INT32, sp), sp)
    if_stmt = b.end_if(sp)

    assert isinstance(if_stmt, ir.IfStmt)
    assert len(if_stmt.return_vars) == 1
    b.end_function(sp)


def test_builder_if_else_str():
    """Builder-constructed if-else should match manually constructed one."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    val42 = ir.ConstInt(42, ir.INT32, sp)
    val0 = ir.ConstInt(0, ir.INT32, sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    b.assign(x, val42, sp)
    b.begin_else(sp)
    b.assign(x, val0, sp)
    built_if = b.end_if(sp)
    b.end_function(sp)

    manual_x = ir.Var("x", st, sp)
    manual_then = ir.AssignStmt(manual_x, val42, sp)
    manual_else = ir.AssignStmt(manual_x, val0, sp)
    manual_if = ir.IfStmt(ir.ConstBool(True, sp), manual_then, manual_else, [], sp)

    assert str(built_if) == str(manual_if)


# ---------- Program building ----------


def test_builder_program():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("func_a", sp)
    x = b.func_arg("x", st, sp)
    b.return_type(st)
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    func_a = b.end_function(sp)

    b.begin_function("func_b", sp)
    y = b.func_arg("y", st, sp)
    b.return_type(st)
    b.assign(y, ir.ConstInt(2, ir.INT32, sp), sp)
    func_b = b.end_function(sp)

    b.begin_program("test_prog", sp)
    b.add_function(func_a)
    b.add_function(func_b)
    prog = b.end_program(sp)

    assert isinstance(prog, ir.Program)
    assert prog.name == "test_prog"
    assert len(prog.functions) == 2
    assert prog["func_a"] is not None
    assert prog["func_b"] is not None


def test_builder_program_functions_sorted():
    """Program functions are sorted by name."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("zebra", sp)
    z = b.func_arg("z", st, sp)
    b.assign(z, ir.ConstInt(0, ir.INT32, sp), sp)
    func_z = b.end_function(sp)

    b.begin_function("alpha", sp)
    a = b.func_arg("a", st, sp)
    b.assign(a, ir.ConstInt(1, ir.INT32, sp), sp)
    func_a = b.end_function(sp)

    b.begin_program("prog", sp)
    b.add_function(func_z)
    b.add_function(func_a)
    prog = b.end_program(sp)

    # Program.functions is a string -> Function mapping ordered by function name.
    function_names = [func.name for func in prog.functions.values()]
    assert function_names == ["alpha", "zebra"]


def test_builder_get_function_return_types():
    """get_function_return_types takes a string function name."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_program("prog", sp)

    b.begin_function("foo", sp)
    b.func_arg("x", st, sp)
    b.return_type(st)
    b.return_type(st)
    func = b.end_function(sp)
    b.add_function(func)

    # Return types are accessible from the Function object directly
    assert len(func.return_types) == 2

    # get_function_return_types now takes a string function name
    ret_types = b.get_function_return_types("foo")
    assert len(ret_types) == 2

    b.end_program(sp)


# ---------- Nested constructs ----------


def test_builder_nested_for_loops():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    i = b.var("i", st, sp)
    j = b.var("j", st, sp)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    b.begin_for_loop(
        j,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(5, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    inner = b.end_for_loop(sp)
    outer = b.end_for_loop(sp)

    assert isinstance(inner, ir.ForStmt)
    assert isinstance(outer, ir.ForStmt)
    assert str(outer.loop_var) == "i"
    assert str(inner.loop_var) == "j"

    func = b.end_function(sp)
    assert isinstance(func.body, ir.SeqStmts)
    assert isinstance(func.body[0], ir.ForStmt)


def test_builder_for_with_if():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    i = b.var("i", st, sp)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    b.begin_if(ir.ConstBool(True, sp), sp)
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    b.end_if(sp)
    b.end_for_loop(sp)

    func = b.end_function(sp)
    for_stmt = func.body[0]
    assert isinstance(for_stmt, ir.ForStmt)
    assert isinstance(for_stmt.body[0], ir.IfStmt)


def test_builder_if_with_nested_for():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    b.begin_function("f", sp)
    x = b.func_arg("x", st, sp)
    i = b.var("i", st, sp)

    b.begin_if(ir.ConstBool(True, sp), sp)
    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(5, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    b.assign(x, ir.ConstInt(1, ir.INT32, sp), sp)
    b.end_for_loop(sp)
    b.end_if(sp)

    func = b.end_function(sp)
    if_stmt = func.body[0]
    assert isinstance(if_stmt, ir.IfStmt)
    assert isinstance(if_stmt.then_body[0], ir.ForStmt)


# ---------- Complex end-to-end ----------


def test_builder_complex_program():
    """Build a complete program with nested constructs."""
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    # Build a function: def compute(x: int32) -> int32:
    #   for i in range(0, 10, 1):
    #     if i < 5:
    #       x = x + 1
    #     else:
    #       x = x - 1
    #   return x
    b.begin_function("compute", sp)
    x = b.func_arg("x", st, sp)
    i = b.var("i", st, sp)
    b.return_type(st)

    b.begin_for_loop(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        sp,
    )
    cond = ir.Lt(i, ir.ConstInt(5, ir.INT32, sp), ir.INT32, sp)
    b.begin_if(cond, sp)
    b.assign(x, ir.Add(x, ir.ConstInt(1, ir.INT32, sp), ir.INT32, sp), sp)
    b.begin_else(sp)
    b.assign(x, ir.Sub(x, ir.ConstInt(1, ir.INT32, sp), ir.INT32, sp), sp)
    b.end_if(sp)
    b.end_for_loop(sp)

    b.return_([x], sp)
    compute = b.end_function(sp)

    # Build the program
    b.begin_program("my_prog", sp)
    b.add_function(compute)
    prog = b.end_program(sp)

    assert isinstance(prog, ir.Program)
    assert prog.name == "my_prog"
    assert len(prog.functions) == 1
    assert prog["compute"] is not None

    # Verify structure: function body is SeqStmts (for + return)
    func = prog["compute"]
    assert func is not None
    body = func.body
    assert isinstance(body, ir.SeqStmts)
    assert isinstance(body[0], ir.ForStmt)
    assert isinstance(body[1], ir.ReturnStmt)
    for_stmt_body = body[0]
    assert isinstance(for_stmt_body, ir.ForStmt)
    for_body = for_stmt_body.body
    if_stmt = for_body[0]
    assert isinstance(if_stmt, ir.IfStmt)
    assert if_stmt.else_body is not None


# ---------- CreateSymbolicScalar ----------


def test_builder_create_scalar_var():
    b = ir.IRBuilder()
    ss = b.create_scalar_var("n")
    ss = b.create_const_int(42)


# ---------- create_tensor_var ----------


def test_builder_create_tensor_var():
    b = ir.IRBuilder()
    tv = b.create_tensor_var(pypto.DT_FP32, [8, 16], name="my_tensor")
    assert tv is not None


def test_builder_create_tensor_var_symbolic_shape():
    b = ir.IRBuilder()
    n = b.create_scalar_var("n")
    tv = b.create_tensor_var(pypto.DT_FP32, [n, 16])
    assert tv is not None


def test_builder_create_tensor_op_stmt():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)
    result_var = ir.Var("out", st, sp)
    token = ir.Var("tok", st, sp)
    arg1 = ir.ConstInt(1, ir.INT32, sp)
    arg2 = ir.ConstInt(2, ir.INT32, sp)
    attrs = {
        "a": 1,
        "b": [1, 1],
        "c": b.create_scalar_var("n"),
        "d": [b.create_scalar_var('b'), b.create_scalar_var('c')],
    }
    created = b.create_tensor_op_stmt([result_var], token, "add", [arg1, arg2], [], attrs, sp)
    manual = ir.TensorOpStmt([result_var], token, "add", [arg1, arg2], [], attrs, sp)
    assert str(created) == str(manual)


def test_builder_create_seq_stmts():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)
    x = ir.Var("x", st, sp)
    stmt1 = ir.AssignStmt(x, ir.ConstInt(1, ir.INT32, sp), sp)
    stmt2 = ir.AssignStmt(x, ir.ConstInt(2, ir.INT32, sp), sp)

    seq = b.create_seq_stmts([stmt1, stmt2], sp)
    assert isinstance(seq, ir.SeqStmts)
    assert len(seq.stmts) == 2


def test_builder_create_return_stmt():
    b = ir.IRBuilder()
    sp = _span()

    stmt = b.create_return_stmt([ir.ConstInt(42, ir.INT32, sp)], sp)
    assert isinstance(stmt, ir.ReturnStmt)
    assert str(stmt) == str(ir.ReturnStmt([ir.ConstInt(42, ir.INT32, sp)], sp))


def test_builder_create_yield_stmt():
    b = ir.IRBuilder()
    sp = _span()

    stmt = b.create_yield_stmt([ir.ConstInt(1, ir.INT32, sp)], sp)
    assert isinstance(stmt, ir.YieldStmt)
    assert str(stmt) == str(ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp))


def test_builder_create_if_stmt():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)
    x = ir.Var("x", st, sp)
    ret_var = ir.Var("out", st, sp)
    then_body = ir.AssignStmt(x, ir.ConstInt(1, ir.INT32, sp), sp)
    cond = ir.ConstBool(True, sp)
    else_body = ir.AssignStmt(x, ir.ConstInt(0, ir.INT32, sp), sp)

    stmt = b.create_if_stmt(cond, then_body, else_body, [ret_var], sp)
    assert isinstance(stmt, ir.IfStmt)
    assert len(stmt.return_vars) == 1


def test_builder_create_for_stmt():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)
    i = ir.Var("i", st, sp)
    iter_arg = ir.IterArg("sum", st, ir.ConstInt(0, ir.INT32, sp), sp)
    ret_var = ir.Var("sum_out", st, sp)
    body = ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp)

    stmt = b.create_for_stmt(
        i,
        ir.ConstInt(0, ir.INT32, sp),
        ir.ConstInt(10, ir.INT32, sp),
        ir.ConstInt(1, ir.INT32, sp),
        [iter_arg],
        body,
        [ret_var],
        sp,
    )
    assert isinstance(stmt, ir.ForStmt)
    assert len(stmt.iter_args) == 1
    assert len(stmt.return_vars) == 1
    assert str(stmt) == str(
        ir.ForStmt(
            i,
            ir.ConstInt(0, ir.INT32, sp),
            ir.ConstInt(10, ir.INT32, sp),
            ir.ConstInt(1, ir.INT32, sp),
            [iter_arg],
            body,
            [ret_var],
            sp,
        )
    )


def test_builder_create_while_stmt():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)
    iter_arg = ir.IterArg("cnt", st, ir.ConstInt(0, ir.INT32, sp), sp)
    ret_var = ir.Var("cnt_out", st, sp)
    body = ir.YieldStmt([ir.ConstInt(1, ir.INT32, sp)], sp)
    cond = ir.ConstBool(True, sp)

    stmt = b.create_while_stmt(cond, [iter_arg], body, [ret_var], sp)
    assert isinstance(stmt, ir.WhileStmt)
    assert len(stmt.iter_args) == 1
    assert len(stmt.return_vars) == 1
    assert str(stmt) == str(
        ir.WhileStmt(cond, [iter_arg], body, [ret_var], sp)
    )


def test_builder_create_break_stmt():
    b = ir.IRBuilder()
    sp = _span()

    stmt = b.create_break_stmt([], sp)
    assert isinstance(stmt, ir.BreakStmt)

    stmt = b.create_break_stmt([ir.ConstInt(42, ir.INT32, sp)], sp)
    assert isinstance(stmt, ir.BreakStmt)


def test_builder_create_continue_stmt():
    b = ir.IRBuilder()
    sp = _span()

    stmt = b.create_continue_stmt([], sp)
    assert isinstance(stmt, ir.ContinueStmt)

    stmt = b.create_continue_stmt([ir.ConstInt(1, ir.INT32, sp)], sp)
    assert isinstance(stmt, ir.ContinueStmt)


def test_builder_create_function():
    b = ir.IRBuilder()
    sp = _span()
    st = ir.ScalarType(ir.INT32)

    x = ir.Var("x", st, sp)
    func_a = b.create_function("func_a", [x], [st], ir.AssignStmt(x, ir.ConstInt(1, ir.INT32, sp), sp), sp)

    y = ir.Var("y", st, sp)
    func_b = b.create_function("func_b", [y], [st], ir.AssignStmt(y, ir.ConstInt(2, ir.INT32, sp), sp), sp)

    prog = b.create_program([func_a, func_b], "multi_prog", sp)
    assert isinstance(prog, ir.Program)
    assert len(prog.functions) == 2
