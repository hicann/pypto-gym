# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What makes an Acc -> Vec fixpipe move more than a same-type plain copy.

The fixpipe's scalar path and its dual destination control are mutually exclusive in the
hardware. The old repository states the rule in full at its frontend
(``easyasc/stub_functions/cube.py:1884``):

    The fixpipe scalar quant rides on the deqScalar, which is only available when the dual
    destination control is off (SINGLE). SPLITM/SPLITN ... support neither relu nor any
    non-default requant parameter; they only carry the SAME-TYPE plain copy (fp32 -> fp32 or
    int32 -> int32). Even the deqScalar-free float downcasts (fp32 -> fp16/bf16) are NOT
    supported in split mode.

One fact, read in two directions, and this module exists so there is one place that decides
which side of it a move falls on:

* ``split`` + a rider is a move the machine does not perform, and ``ir/verify.py`` refuses it.
* ``single`` + NO rider is a move that could have used both vector sub-blocks and did not, and
  ``ir/lint.py`` says so on the performance channel.

The second reading is why the predicate moved out of the verifier: a lint that exempts
"SINGLE was forced here" has to agree, exactly, with the checker that forces it. Two copies of
this list would drift into a lint that either nags about moves with no alternative or stays
quiet about ones that had one.

The frontend has already normalised the inputs: ``rules_mem.py`` drops a ``scale`` that folds
to 1 and an ``offset`` that folds to 0, and lands ``relu`` only when true. A scale that is a
runtime value survives, which is the safe direction - nobody can prove it is 1.
"""

from .core import Ident, Literal, Op, Value
from .types import MemType

#: rider attribute -> the value that means "not present". ``hif8_hybrid`` rides the same
#: deqScalar as the quantised path.
RIDER_DEFAULT = {"relu": False, "scale": 1, "offset": 0, "hif8_hybrid": False}

#: modes in which the dual destination control is ON. The IR default is ``splitm``.
SPLIT_MODES = ("splitm", "splitn")


def dual_mode_of(op: Op) -> str:
    mode = op.attrs.get("dual_mode")
    return (mode.name if isinstance(mode, Ident) else str(mode)) if mode is not None else "splitm"


def fixpipe_riders(op: Op) -> list[tuple[str, str]]:
    """Every reason this move is not a same-type plain copy, as ``(what, detail)`` pairs.

    ``what`` is the rider's attribute name, or ``"dtype"`` for a conversion; ``detail`` is the
    part a caller's message needs. Structured rather than pre-worded because the two callers
    word it differently - the verifier names one refusal per rider, the lint only asks whether
    the list is empty.
    """
    out = []
    for name, default in RIDER_DEFAULT.items():
        v = op.attrs.get(name, default)
        if isinstance(v, Literal):
            v = v.value
        if isinstance(v, Value) or v != default:
            out.append((name, "relu" if name == "relu" else f"the requant rider {name!r}"))
    if len(op.operands) >= 2:
        dst, src = (getattr(v, "type", None) for v in op.operands[:2])
        if isinstance(dst, MemType) and isinstance(src, MemType) and dst.dtype != src.dtype:
            # the half PTO's static_assert misses - it covers the QUANTISED overload only, so
            # its unquantised arm would compile a split-mode f32 -> f16 the hardware does not do
            out.append(("dtype", f"{src.dtype.name} -> {dst.dtype.name}"))
    return out
