# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``core.*`` and ``scalar.*``: core identity, scalar arithmetic, cells, scalar GM access."""

from ._dsl import ALL, A, N, R, Res, W, op, retire

for _name, _legacy, _doc in (
    ("cube_num", "GetCubeNum", "number of cube cores in the launch"),
    ("cube_idx", "GetCubeIdx", "index of this cube core"),
    ("vec_num", "GetVecNum", "number of vector cores in the launch"),
    ("vec_idx", "GetVecIdx", "index of this vector core"),
    ("sub_block_idx", "GetSubBlockIdx", "index of this vector core within its cube core's pair"),
):
    op(f"core.{_name}", kinds=ALL, pipe="S", results=(Res("out", "i32"),), legacy=_legacy, doc=_doc)

op("core.set_hf32", pipe="S", attrs=(A("enable", "bool", required=True),), legacy="set_hf32", effects=("memory",),
   doc="toggle HF32 rounding for the cube")
op("core.set_sat_flag", pipe="S", attrs=(A("mode", "ident", required=True,
   doc="float (48) | float8 (50) | int (53) | cast (59) | global (60)"),
   A("enable", "bool|value", required=True, pattern="scalar")), effects=("memory",),
   doc="write a raw CTRL bit; global=0 selects vf.cast.saturate, global=1 selects cast (0 clamps, 1 truncates)")
op("core.get_sat_flag", pipe="S", attrs=(A("mode", "ident", required=True),), results=(Res("out", "i32"),),
   doc="read one CTRL saturation flag (see core.set_sat_flag)")
op("core.clean_dcache", pipe="S", attrs=(A("dst", "value", pattern="gm<*, *>", access="write"), A("dcci_dst", "any"),
   A("entire_type", "any"), A("mode", "any")), legacy="clean_dcache", effects=("memory",), doc="data-cache clean / invalidate")

# Cells: the mutable scalar variables of the DSL (RFC-0001 §5.2).
op("scalar.cell", kinds=ALL, pipe="S", attrs=(A("init", "int|float|bool|value", doc="initial value"),),
   results=(Res("cell", "cell<*>"),), legacy="create_var", doc="declare a mutable scalar variable")
op("scalar.set", kinds=ALL, pipe="S", operands=(W("cell", "cell<T>"), N("src", "value")), legacy="var_assign",
   doc="store a scalar into a cell")

for _name, _legacy in (("add", "var_add"), ("sub", "var_sub"), ("mul", "var_mul"), ("div", "var_div"), ("mod", "var_mod"),
                       ("and", "var_and"), ("or", "var_or"), ("xor", "var_xor"), ("shl", "var_shl"), ("shr", "var_shr"),
                       ("min", "Min"), ("max", "Max"), ("ceil_div", "CeilDiv")):
    op(f"scalar.{_name}", kinds=ALL, pipe="S", operands=(N("a", "value"), N("b", "value")), results=(Res("out", "scalar"),),
       attrs=(A("rounding", "ident", doc="floor (default) | trunc"),) if _name in ("div", "mod") else (),
       legacy=_legacy, doc=f"scalar {_name}; operands are scalars, cells (read) or literals")

op("scalar.not", kinds=ALL, pipe="S", operands=(N("a", "value"),), results=(Res("out", "scalar"),), legacy="var_inv",
   doc="bitwise not (logical not on b1)")
op("scalar.neg", kinds=ALL, pipe="S", operands=(N("a", "value"),), results=(Res("out", "scalar"),), doc="negate")
op("scalar.abs", kinds=ALL, pipe="S", operands=(N("a", "value"),), results=(Res("out", "scalar"),), legacy="scalar_abs", doc="absolute value")
op("scalar.sqrt", kinds=ALL, pipe="S", operands=(N("a", "value"),), results=(Res("out", "scalar"),), legacy="scalar_sqrt", doc="square root")
op("scalar.align", kinds=ALL, pipe="S", operands=(N("a", "value"),), attrs=(A("n", "int", required=True),),
   results=(Res("out", "scalar"),), legacy=("Align8", "Align16", "Align32", "Align64", "Align128", "Align256"),
   doc="round up to a multiple of n")
op("scalar.cmp", kinds=ALL, pipe="S", operands=(N("a", "value"), N("b", "value")),
   attrs=(A("pred", "ident", required=True, doc="lt | le | gt | ge | eq | ne"),), results=(Res("out", "b1"),),
   doc="compare two scalars")
op("scalar.select", kinds=ALL, pipe="S", operands=(N("cond", "b1"), N("a", "value"), N("b", "value")),
   results=(Res("out", "scalar"),), doc="cond ? a : b")
op("scalar.cast", kinds=ALL, pipe="S", operands=(N("a", "value"),), results=(Res("out", "scalar"),),
   doc="convert a scalar; the result type says the target dtype")
op("scalar.const", kinds=ALL, pipe="S", attrs=(A("value", "int|float|bool", required=True),), results=(Res("out", "scalar"),),
   doc="a typed constant (literals in operand position need no op; this gives a literal a name and a type)")

op("scalar.load", kinds=ALL, pipe="S", operands=(R("src", "mem<T, *>"), N("index", "value")),
   results=(Res("out", "scalar"),), legacy="GetValueFrom", doc="load one element of a GM/workspace or UB tensor into a scalar")
op("scalar.store", kinds=ALL, pipe="S", operands=(W("dst", "mem<T, *>"), N("index", "value"), N("src", "value")),
   effects=("memory",), legacy="SetValueTo", doc="store a scalar into one element of a GM/workspace or UB tensor")

retire("create_varlist", "a list of Vars is a static Python container of cells; nothing to emit")
retire("inline", "no escape hatch (D-007)")
retire("reset_cache", "simulator bookkeeping; the sim backend resets its cache per launch")
