# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the chunked row scan through OpExec and check it against the Torch reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case source_narrow33

One kernel with two storage branches selected on the host by H: H >= 64 loads the row at its
own pitch and must be a multiple of 64, H < 64 loads into a padded 64-column UB row. Both
branches are emitted every launch, so the cases below exercise each one directly.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import chunk_row_cumsum_kernel
from reference import make_inputs, reference

# Both sides add each row into the previous FP32 prefix in that same order, so the two agree
# bit for bit in practice; the 1e-5 pair is the source's own element bound, kept as it was.
# The 1e-7 relative L2 is the part that bites: a kernel that skips a chunk reset or
# reassociates the additions moves the whole vector, and the norm sees that long before a
# single element breaches atol.
TOLERANCE = {"rtol": 1e-5, "atol": 1e-5, "max_relative_l2": 1e-7}

CASES = [
    {"id": "source_wide_aligned", "seed": 13100, "block_dim": 1,
     "purpose": "The wide branch on exact geometry: two 64-column panels, M a multiple of chunk_size",
     "parameters": {"M": 16, "H": 128, "chunk_size": 8}},
    {"id": "source_wide_tail", "seed": 13101, "block_dim": 1,
     "purpose": "Row tail on the wide branch: the last chunk holds 3 rows of 8",
     "parameters": {"M": 19, "H": 128, "chunk_size": 8}},
    {"id": "source_narrow33", "seed": 13102, "block_dim": 1,
     "purpose": "The narrow branch: 33 columns padded to a 64-column UB row, plus a row tail",
     "parameters": {"M": 19, "H": 33, "chunk_size": 8}},
    {"id": "narrow_one", "seed": 13103, "block_dim": 1,
     "purpose": "One column: 63 of the 64 lanes are pad, and none of them may reach GM",
     "parameters": {"M": 17, "H": 1, "chunk_size": 8}},
    {"id": "narrow63", "seed": 13104, "block_dim": 1,
     "purpose": "63 columns: the widest row that still takes the narrow padded branch",
     "parameters": {"M": 33, "H": 63, "chunk_size": 8}},
    {"id": "multiple_groups", "seed": 13105, "block_dim": 2,
     "purpose": "Two core groups, four vector owners: the chunk partition must not let one owner "
                "write into another's rows",
     "parameters": {"M": 65, "H": 128, "chunk_size": 8}},
    {"id": "chunk_one", "seed": 13106, "block_dim": 2,
     "purpose": "chunk_size = 1: every row resets, so the output is the input and no recurrence runs",
     "parameters": {"M": 9, "H": 64, "chunk_size": 1}},
    {"id": "one_row", "seed": 13107, "block_dim": 1,
     "purpose": "A single narrow row: only the initial store runs, the recurrence loop never does",
     "parameters": {"M": 1, "H": 33, "chunk_size": 1}},
    {"id": "fp32_cancellation", "seed": 13108, "block_dim": 1,
     "purpose": "1e8, 1, -1e8 in one chunk: the FP32 addition order makes the last prefix exactly "
                "0.0, and an implementation that accumulates more precisely returns 1.0",
     "parameters": {"M": 3, "H": 64, "chunk_size": 3, "pattern": "cancellation"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. The destination is handed in poisoned with NaN and seeded into the
    launch, so a cell no owner writes reads back as NaN instead of as a plausible zero."""
    op = OpExec(chunk_row_cumsum_kernel, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    prefixes = torch.full_like(inputs["x"], float("nan"))
    return {"prefixes": op(inputs["x"], prefixes, inputs["M"], inputs["H"], inputs["chunk_size"])}


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
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (M={p['M']} H={p['H']} chunk_size={p['chunk_size']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
