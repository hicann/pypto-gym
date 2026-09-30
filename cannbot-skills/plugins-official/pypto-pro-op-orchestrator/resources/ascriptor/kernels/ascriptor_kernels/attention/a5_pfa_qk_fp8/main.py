# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the E4M3 Q/K prefill attention through OpExec and check it against the dense reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case fd_split

Only the score matmul is quantised here: Q and K are float8_e4m3fn and everything after the
QK^T -- V, the published P, the online softmax, PV, the flash-decoding merge and the output --
stays on the bf16/fp32 path. Read it beside a5_pfa_bf16, which is the same schedule with no
quantisation at all, and a5_pfa_qk_hif8, which is the same edit with HiFloat8 carriers.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import qk_softmax_pv_qk_e4m3_kernel
from reference import make_inputs, reference

# The reference computes one dense FP32 attention and rounds once at the end; the kernel
# publishes P in BF16, accumulates a K-tile at a time and merges split tiles from an
# unnormalised FP32 partial, so the two agree to a budget rather than bit for bit. The relative
# L2 residual is what rejects an output that is merely small -- an all-zero result would pass
# atol on its own.
TOLERANCE = {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.01}

# The flat (m_tile, k_tile) interval each core owns is derived from block_dim, so block_dim is
# part of the case, not a free knob: the merge path is only defined while at most two cores own
# one query tile and no two interior boundaries land on the same tile.
CASES = [
    {"id": "fd_matched", "seed": 0, "block_dim": 2,
     "purpose": "The core boundary falls exactly on an M-tile boundary: no split, no FD merge",
     "parameters": {"variant": "fd", "B": 1, "MQ": 256, "N": 512, "D": 128}},
    {"id": "fd_split", "seed": 1, "block_dim": 2,
     "purpose": "One M tile owned by two cores: the unnormalised partial and the two-way merge",
     "parameters": {"variant": "fd", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "fd_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "Two batches with independent KV, a one-row query tail and a 5-key tail K-tile",
     "parameters": {"variant": "fd", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "fd_reuse", "seed": 3, "block_dim": 3,
     "purpose": "Five M tiles over three cores: two interior boundaries inside different M tiles, "
                "plus a one-row query tail and a one-key tail K-tile",
     "parameters": {"variant": "fd", "B": 1, "MQ": 513, "N": 385, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """The output is allocated NaN-poisoned and seeded into the launch, so a query row no drain
    writes reads back as NaN rather than as a plausible zero."""
    out = torch.full(inputs["q"].shape, float("nan"), dtype=torch.bfloat16)
    op = OpExec(qk_softmax_pv_qk_e4m3_kernel, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["q"], inputs["k"], inputs["v"], out,
                      inputs["B"], inputs["MQ"], inputs["N"], 128)}


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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (B={p['B']}, MQ={p['MQ']}, N={p['N']}, "
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
