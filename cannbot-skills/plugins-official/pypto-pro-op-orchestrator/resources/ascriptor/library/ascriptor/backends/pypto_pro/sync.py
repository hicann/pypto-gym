# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit intra-core flag printing, kept out of emit.py for its size bound (as PTO ISA's sync.py)."""


def raw_flag(side, op) -> None:
    from .emit import PIPE, PyptoGap  # emit.py imports this module

    code = op.opcode
    # The RAW intra-core flag pair. `sync.set`/`sync.wait` in emit.py carry an allocated
    # `sync.event` (autosync's ids); these two name the pipes and the id outright, which
    # is exactly pl's own shape: `sync_src`/`sync_dst(set_pipe=, wait_pipe=, event_id=)`.
    sp = str(getattr(op.attrs.get("src"), "name", op.attrs.get("src")))
    wp = str(getattr(op.attrs.get("dst"), "name", op.attrs.get("dst")))
    if sp not in PIPE or wp not in PIPE:
        raise PyptoGap(op, f"{code} pipes {sp}->{wp} have no pl.PipeType")
    # `src == dst` needs no check here: ir/verify.py refuses the raw flag pair with
    # equal pipes for every backend ("raw flag needs distinct source and destination
    # pipes"), which is where a rule pl and cce BOTH hold belongs.
    # Nor is the id range: `ir/verify.py` refuses anything outside [0, 8) with the
    # source line, and pl's own range is the same [0, 7]. Both rules live one layer up
    # because cce holds them too -- the printer carries neither duplicate.
    eid = op.attrs.get("event_id")
    fn = "sync_src" if code == "sync.set_flag" else "sync_dst"
    # pl takes a runtime Scalar here as well as a literal, so the id rides through
    # `env.ref` rather than being folded: the corpus really does compute one
    # (tests/runtime/test_raw_flags.py drives it from a kernel parameter).
    side.emit(f"pl.system.{fn}(set_pipe={PIPE[sp]}, wait_pipe={PIPE[wp]}, "
              f"event_id={side.env.ref(eid)})")
