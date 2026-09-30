# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the presence-table masking stage through OpExec and check it against the reference.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case wrap_after_four

A mask that is not known until the data arrives, kept as ordinary data rather than as a
predicate: a marker is scattered to each selected key column, and the read side compares it
back. The marker is the chunk's generation number, so a residue left by an earlier chunk
carries an earlier number and the `EQ` misses it -- a cleared table with no clearing pass.
`_clear_presence` runs only when the generation counter wraps.

The rebuilt predicate is published as `present_t` and compared EXACTLY. That is the point of
the unit: folding it into the exponential's tolerance is what would let a wrong predicate look
like a slightly wrong number.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import KEYS, QUERIES, presence_mask_stage
from reference import make_inputs, reference

# `prob_t` is one fused device exponential per element against an FP64 exponential rounded
# once, so the tolerance covers the exponential and nothing else. `denom` adds up to 1024 FP32
# additions in chunk order to that, and its residual norm is what rejects a denominator that
# silently lost a whole chunk. `present_t` gets no tolerance at all: it is 1.0 or 0.0, and a
# stale generation marker read as present is a wrong bit, not a rounding difference. This is
# not a formality -- the first PyPTO Pro acquisition of this unit returned present_t as 1.0 in
# all 65536 cells (the printer dropped the predicate from a masked scalar broadcast) while
# prob_t, whose mask it did carry, was right to one ULP.
TOLERANCE = {"default": {"rtol": 0.001, "atol": 1e-06},
             "present_t": None,
             "denom": {"rtol": 0.001, "atol": 1e-06, "max_relative_l2": 0.0001}}

CASES = [
    {"id": "spread", "seed": 4410, "block_dim": 1,
     "purpose": "The ordinary case: every chunk holds some of every query's keys, so each "
                "rebuild both writes new markers and steps over the previous generation's residue",
     "parameters": {"pattern": "spread"}},
    {"id": "stale_overlap", "seed": 4411, "block_dim": 1,
     "purpose": "Eight columns in the even chunks, one in the odd ones: every odd chunk must "
                "report seven local rows absent while they still carry the previous marker",
     "parameters": {"pattern": "stale_overlap"}},
    {"id": "wrap_after_four", "seed": 4412, "block_dim": 1,
     "purpose": "Chunk 0 and chunk EPOCHS carry the same generation number over disjoint "
                "column sets, so anything that survives the wrap shows up as a key that is not there",
     "parameters": {"pattern": "wrap_after_four"}},
    {"id": "one_chunk", "seed": 4413, "block_dim": 1,
     "purpose": "Every selected column lives in one chunk: seven rebuilds must produce an "
                "entirely absent predicate and contribute exactly zero to the denominator",
     "parameters": {"pattern": "one_chunk"}},
    {"id": "empty_rows", "seed": 4414, "block_dim": 1,
     "purpose": "Half the query rows select nothing at all, so their index slots are all the "
                "padding value and their denominator stays zero",
     "parameters": {"pattern": "empty_rows"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the stage. All three destinations are handed in poisoned with NaN and seeded into
    the launch: a key column no chunk writes must read back as NaN, not as the zero an absent
    key legitimately produces."""
    outputs = {"prob_t": torch.full((KEYS, QUERIES), float("nan")),
               "present_t": torch.full((KEYS, QUERIES), float("nan")),
               "denom": torch.full((1, QUERIES), float("nan"))}
    op = OpExec(presence_mask_stage, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(inputs["score_t"], inputs["index"], inputs["rowmax"], *outputs.values())
    return dict(zip(outputs, actual, strict=True))


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail, outside = torch.equal(got, want), "  bitwise", got != want
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        atol, rtol = bounds.get("atol", 0.0), bounds.get("rtol", 0.0)
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for each
        # element. Print the worst element as a fraction of its own allowance, so the number has a
        # bound of 1 and a passing line cannot read as a failing one.
        room = (atol + rtol * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        outside = ~torch.isclose(got.float(), want.float(), **bounds)
        ok = not outside.any().item()
        detail = f"  allclose={margin:.2f}x (atol={atol:g} rtol={rtol:g})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok and outside.any():
        # The destinations arrive NaN-poisoned, so an element that is still NaN was never written.
        # Saying which, and where, is the difference between "something is nan" and "row 127 is".
        idx, poison = outside.nonzero(), int((outside & torch.isnan(got.float())).sum())
        note = f", {poison} still NaN-poisoned (never written)" if poison else ""
        print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
              f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (pattern={case['parameters']['pattern']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            rule = TOLERANCE.get(name, TOLERANCE["default"])
            if not compare(name, actual[name], expected[name], rule):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
