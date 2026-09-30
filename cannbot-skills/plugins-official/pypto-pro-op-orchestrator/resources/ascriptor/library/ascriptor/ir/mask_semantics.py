# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What a ``mask=`` does to the lanes it turns OFF, per vf operator.

Sixty-six vf operators take a predicate, and they do not agree on what an inactive lane means.
Most of them WRITE ZERO there - the arithmetic, the casts, the register copies, and the
mask-register booleans alike. A masked STORE does not: it never writes that lane, so the memory
keeps whatever it held. Nothing preserves a destination REGISTER lane; the register is always
written whole, and only memory can be left alone. Two answers, opposite, to the same keyword.

The cost of guessing is not a crash. It is a wrong number, in the direction of whichever
assumption you brought. And guessing conservatively is expensive too: the sparse-attention
kernels written against this DSL took two instructions - ``select`` to push masked lanes to
-inf, then ``exp`` - where the reference took one, because ``expsub(v, v, zero, mask=live)``
zeroes the inactive lanes itself and swallows whatever the cube left in the padding rows. Not
knowing that the zeroing was guaranteed is what made the one-instruction form unsafe to write.

The roles
---------
``zero``      the destination REGISTER lane is written 0. Most arithmetic, every cast, every
              register-to-register copy, the register side of a load, and the ``mask_*``
              booleans (M10-051 corrected an older belief that those preserved inactive bits;
              the interpreter ANDs the result with the mask and writes the whole register).
``skip``      the destination MEMORY is not written, and keeps its previous bytes. Every store,
              and the scatter. **Not** zeroing - a masked store does not clear what it skips.
``filter``    the mask selects which SOURCE lanes take part; the destination is packed or
              reduced, so it is not lane-aligned with the mask at all.
``select``    the mask is the selector, not a predicate: every lane is written, choosing
              between the two sources.
``predicate`` the result is a new mask, and inactive lanes come out FALSE (M10-051) - even when
              the destination is the execution mask itself.
``count``     the destination is derived from the mask bits themselves.
``block``     the bit at each UINT32 index lane selects a whole 32-byte block, copied with every
              lane; unselected blocks are written 0 (I014).
``ignored``   the instruction takes the mask but writes every element: A5 interleaved stores (RFC-0001).

Every entry below was read from the reference interpreter, which is the only place the answer
is written down: ``backends/sim/vf_ops.py`` and ``backends/sim/interp.py``. The table is
checked against the op registry for completeness, so a new masked operator cannot be added
without choosing its role here - which is the point. Adding one is the moment the question is
cheap to answer; every later moment is a kernel author guessing.
"""

from __future__ import annotations

ROLES = ("zero", "skip", "filter", "select", "predicate", "count", "block", "ignored")

#: opcode -> role. Complete over every ``vf.*`` op whose spec carries a ``mask`` attribute;
#: `tests/ir/test_mask_semantics.py` fails if the registry gains one that is not here.
MASK_ROLE: dict[str, str] = {
    # -- zero: the inactive destination lane is written 0 ------------------------------------
    **{f"vf.{name}": "zero" for name in (
        "abs", "abssub", "add", "adds", "and", "axpy", "cast", "copy", "cpadd", "div", "dup",
        "exp", "expsub", "gather_copy", "ln", "load", "log", "log10", "log2",
        "lrelu", "max", "maxs", "min", "mins", "mod", "mul", "muladddst", "muldstadd", "muls",
        "mulscast", "neg", "not", "or", "prelu", "relu", "shiftl", "shiftls", "shiftr",
        "shiftrs", "sqrt", "sub", "xor",
        # the mask-register booleans: the interpreter ANDs the result with the mask and writes
        # the whole register, so an inactive bit comes out 0. M10-051 corrected an older belief
        # that these preserved it; `docs/api/precision.md` has carried the correction since.
        "mask_and", "mask_mov", "mask_not", "mask_or", "mask_xor")},
    # -- skip: the memory of an inactive lane is not written at all ---------------------------
    **{f"vf.{name}": "skip" for name in (
        "scatter_copy", "store", "store_cont")},
    # -- filter: the mask picks SOURCE lanes; the destination is packed or reduced ------------
    **{f"vf.{name}": "filter" for name in (
        "cadd", "cgadd", "cgmax", "cgmin", "cmax", "cmin", "gathermask", "histograms",
        "squeeze")},
    # -- the one-offs -------------------------------------------------------------------------
    "vf.select": "select",
    "vf.mask_sel": "select",   # torch.where(mask, a, b): both sources contribute, nothing is off
    "vf.cmp": "predicate",
    "vf.cmps": "predicate",
    "vf.unsqueeze": "count",
    "vf.gatherb": "block",     # physical predicate bit 4*b selects block b, whatever its own lanes say
    "vf.store_interleave": "ignored",  # measured on A5: every pair is written
}

#: one line per role, for a message that has to explain itself on the spot
ROLE_SUMMARY: dict[str, str] = {
    "zero": "the inactive destination lane is written 0",
    "skip": "the inactive lane is not written; the destination memory keeps its bytes",
    "filter": "the mask selects source lanes; the destination is packed or reduced",
    "select": "the mask is the selector, not a predicate: every lane is written",
    "predicate": "the result is a new mask; inactive lanes come out false",
    "count": "the destination is derived from the mask bits themselves",
    "block": "the bit at each index lane selects a whole block; unselected blocks are written 0",
    "ignored": "the instruction takes the mask but writes every element",
}


def mask_role(opcode: str) -> str | None:
    """The role of ``opcode``'s mask, or ``None`` if it does not take one."""
    return MASK_ROLE.get(opcode)
