# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``mmad_settle``: give every A2-family L0C accumulation the M-pipe settle it needs.

M10-081 measured that c220 (910B, 910_93) does not interlock a short same-pipe L0C RAW: an
``is_init=False`` MMAD may read an accumulator whose preceding MMAD has not written back, and an
M-pipe barrier between them is the sufficient control.  ``desugar`` emits that barrier inside the
split-K expansion it generates, which is narrower than the hazard in a way the corpus makes plain:
a K split *written by hand* -- two ``matmul`` calls accumulating one L0C slot, or one call inside a
loop whose back edge carries no barrier -- gets nothing, on any dtype.  Eight catalogue kernels are
written that way, and until this pass the only thing that spoke there was a warning.

The rule is the hardware's, not the author's, so the pipeline owns it.  This pass asks
:func:`ascriptor.ir.lint.unsettled_mmad_accumulate` -- the same analysis the lint reports with, so a
quiet lint after the pass is evidence rather than coincidence -- and inserts
``sync.barrier(pipe=M)`` immediately before each accumulate it names.  Before, not after the
producer: that is one placement for the straight-line pair, the loop back edge and the branch join
at once, and it is where the ``is_init=False`` operand is actually read.

Inserting clears the hazard for the sites downstream of it, so the analysis is re-run after each
round until it names nothing; two rounds are enough for every kernel in the corpus.  The pass runs
late (after ``scalar_simplify``, before ``liveness``) because the hazard is a property of the final
lowered order, and because a barrier inserted here cannot disturb the event, address or mutex
decisions already made.  ``desugar``'s own insertion stays where it is: it is what M10-081 measured
bitwise on both boards, and this pass is silent wherever it already did the job.
"""

from __future__ import annotations

from ..ir import Ident, Module, Op
from ..ir.builder import Rewriter
from ..ir.lint import unsettled_mmad_accumulate
from .manager import Pass, PassContext

PASS = "mmad_settle"
ROUNDS = 8  # a bound, not a schedule: each round strictly reduces the named sites


def _targets(module: Module) -> set[int]:
    """The ids of the accumulates that still read an unsettled L0C."""
    out: set[int] = set()
    for f in module.functions:
        defs = {r.name: op for op in f.walk() for r in op.results}
        for op in f.walk():
            if op.id is not None and unsettled_mmad_accumulate(f, op, defs):
                out.add(op.id)
    return out


def _settle_round(rw: Rewriter, targets: set[int]) -> Module:
    """Rewrite one measured set of unsettled accumulates."""
    def settle(op: Op) -> list[Op] | None:
        if op.id not in targets:
            return None
        barrier = rw.make("sync.barrier", (), attrs={"pipe": Ident("M")}, from_ops=(op,),
                          note="a2 family: settle the L0C accumulator this MMAD reads")
        return [barrier, op]

    return rw.rewrite(settle)


def run(module: Module, ctx: PassContext) -> Module:
    if getattr(ctx.device, "family", None) != "a2":
        return module  # other families keep their own measured ordering policy (M10-081)
    inserted = 0
    for _ in range(ROUNDS):
        targets = _targets(module)
        if not targets:
            break
        rw = Rewriter(module, PASS)
        module = _settle_round(rw, targets)
        inserted += len(targets)
    else:
        raise AssertionError(f"{PASS}: still naming sites after {ROUNDS} rounds")
    if inserted and ctx is not None:
        ctx.explain.note(f"{inserted} L0C accumulation(s) settled with an M-pipe barrier", kind=PASS)
    return module


PASS_DEF = Pass(PASS, run, accepts="lowered/1", produces="lowered/1",
                doc="insert the c220 M-pipe settle before every unsettled L0C accumulation (M10-081)")

__all__ = ["PASS_DEF", "run"]
