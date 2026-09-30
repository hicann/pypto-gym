# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the blockwise E5M2 quantized matrix product through OpExec and check the payload bytes
and the FP32 block scale against the independent reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case default_midpoints

The kernel computes `product = X.T @ Y` on the cube from K-major FP16 operands, hands the FP32
L0C tile to the vector unit through a `CvMutex`, and there each 128-column row block is scaled
by `max(abs(row)) / 224` and stored as E5M2. Two outputs leave: the raw payload bytes and the
FP32 scale that reconstructs them.

Two store paths share that body and are selected per case by `variant`. `default` writes the
normalized `RegList` straight into an E5M2 UB tile and lets the store convert; `pack4` casts
each half explicitly with a ZERO-layout `CastConfig` and stores two PACK4 halves. Same
arithmetic, two different ways to get eight-bit values out of a vector register.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import block_quant_default, block_quant_pack4
from reference import make_inputs, reference

ENTRIES = {"default": block_quant_default, "pack4": block_quant_pack4}

# The payload is the published ABI -- those bytes are what a consumer reads back, and the
# independent E5M2 grid plus the nearest-even tie rule fixes every one of them, so there is
# nothing for a tolerance to be about. `default_midpoints` and `pack4_midpoints` exist to put
# the normalized values exactly on the ties, where that rule is the entire answer. The scale is
# arithmetic rather than a format: it comes out of an FP32 row maximum divided by 224, so it
# gets a tight allowance, and the relative residual is what stops a missing or vacuous scale
# from passing on atol alone.
TOLERANCE = {"default": None,
             "scale": {"rtol": 1e-05, "atol": 1e-06, "max_relative_l2": 1e-05}}

CASES = [
    {"id": "default_minimum", "seed": 7000, "block_dim": 1,
     "purpose": "The smallest legal shape: 16 rows inside the physical M128 bridge, one K tile",
     "parameters": {"variant": "default", "M": 16, "N": 128, "K": 16, "distribution": "random"}},
    {"id": "default_source", "seed": 7001, "block_dim": 1,
     "purpose": "The original 128x128x128 shape: one full M tile and one full K tile",
     "parameters": {"variant": "default", "M": 128, "N": 128, "K": 128, "distribution": "random"}},
    {"id": "default_tail", "seed": 7002, "block_dim": 2,
     "purpose": "M=144 and K=144: a 16-row M tail and a 16-deep K tail, on two cores",
     "parameters": {"variant": "default", "M": 144, "N": 128, "K": 144, "distribution": "random"}},
    {"id": "default_reuse", "seed": 7003, "block_dim": 1,
     "purpose": "Three M tiles and three K tiles on one core: every L1/L0C double buffer cycles",
     "parameters": {"variant": "default", "M": 384, "N": 128, "K": 272, "distribution": "random"}},
    {"id": "default_idle", "seed": 7004, "block_dim": 3,
     "purpose": "One tile on three cores: two cube groups and their vector peers stay idle",
     "parameters": {"variant": "default", "M": 16, "N": 128, "K": 16, "distribution": "random"}},
    {"id": "default_midpoints", "seed": 7005, "block_dim": 1,
     "purpose": "Products constructed to land on E5M2 ties with an exact scale of 1, so only the "
                "nearest-even rule decides the byte",
     "parameters": {"variant": "default", "M": 128, "N": 128, "K": 128, "distribution": "midpoints"}},

    {"id": "pack4_minimum", "seed": 7010, "block_dim": 1,
     "purpose": "The smallest legal shape through the explicit ZERO-layout cast and PACK4 stores",
     "parameters": {"variant": "pack4", "M": 16, "N": 128, "K": 16, "distribution": "random"}},
    {"id": "pack4_source", "seed": 7011, "block_dim": 1,
     "purpose": "The original 128x128x128 shape, PACK4 store path",
     "parameters": {"variant": "pack4", "M": 128, "N": 128, "K": 128, "distribution": "random"}},
    {"id": "pack4_tail", "seed": 7012, "block_dim": 2,
     "purpose": "M and K tails on two cores, PACK4 store path",
     "parameters": {"variant": "pack4", "M": 144, "N": 128, "K": 144, "distribution": "random"}},
    {"id": "pack4_reuse", "seed": 7013, "block_dim": 1,
     "purpose": "Three M tiles and three K tiles on one core, PACK4 store path",
     "sim_gap": "one payload byte, row 376 column 113. The simulator computes a host FP32 "
                "matrix product and adds it to the FP32 destination; silicon uses the cube's "
                "own reduction tree, which the model does not specify. Here the two land on "
                "either side of an E5M2 midpoint (neighbours 20 and 24, midpoint 22): the "
                "model's normalized value is 22.000002 and encodes to 78, silicon's is "
                "21.999983 and encodes to 77, and the reference agrees with silicon. Run it "
                "under --launcher aclnn or board, where it passes. The tolerance is "
                "deliberately NOT widened to hide this",
     "parameters": {"variant": "pack4", "M": 384, "N": 128, "K": 272, "distribution": "random"}},
    {"id": "pack4_idle", "seed": 7014, "block_dim": 3,
     "purpose": "One tile on three cores with two idle, PACK4 store path",
     "parameters": {"variant": "pack4", "M": 16, "N": 128, "K": 16, "distribution": "random"}},
    {"id": "pack4_midpoints", "seed": 7015, "block_dim": 1,
     "purpose": "The tie construction through the explicit cast: the two store paths must agree "
                "byte for byte where rounding is decided",
     "parameters": {"variant": "pack4", "M": 128, "N": 128, "K": 128, "distribution": "midpoints"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the store path this case names. The payload is handed in poisoned with the byte
    0xDF and the scale with NaN, both seeded into the launch, so a row the kernel never writes
    reads back as poison rather than as a plausible byte."""
    p = case["parameters"]
    payload = torch.full((p["M"], 128), 0xDF, dtype=torch.uint8).view(torch.float8_e5m2)
    scale = torch.full((p["M"], 1), float("nan"))
    op = OpExec(ENTRIES[p["variant"]], launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    values = op(inputs["x"], inputs["y"], payload, scale, p["M"], 128, p["K"])
    return {"payload": values[0].view(torch.uint8), "scale": values[1]}


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
            gap = "  [no sim]" if case.get("sim_gap") else ""
            print(f"{case['id']:20s} {case['purpose']}{gap}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed, skipped = [], []
    for case in selected:
        if case.get("sim_gap") and args.launcher in ("sim", "pipesim"):
            skipped.append(case)
            continue
        p = case["parameters"]
        print(f"{case['id']}  (M={p['M']} N=128 K={p['K']}, {p['variant']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name],
                           TOLERANCE.get(name, TOLERANCE["default"])):
                failed.append(f"{case['id']}/{name}")
    ran = len(selected) - len(skipped)
    print(f"\n{ran - len({f.split('/')[0] for f in failed})}/{ran} cases passed")
    for case in skipped:
        print(f"skipped under {args.launcher}: {case['id']} — {case['sim_gap']}.")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
