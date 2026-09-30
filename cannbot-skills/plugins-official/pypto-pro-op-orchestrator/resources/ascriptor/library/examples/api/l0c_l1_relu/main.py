# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A ReLU applied at the fixpipe on the way from L0C back into L1, then a second cube product.

    python main.py                     # every case, functional simulator
    python main.py --list              # the case ids, with their purpose
    python main.py --case signed_seed1 # one of them
    python main.py --launcher pipesim  # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn    # the cce backend, on this machine's card

`out = FP16(ReLU(FP32(x @ y.T))) @ z.T`, in one launch on one cube core. The first accumulator is
converted straight out of L0C into an FP16 L1 tile by `l0c_mid.relu()`, and the second `matmul`
consumes that tile -- the feedback never passes through UB or GM.

`SEvent(Pipe.FIX, Pipe.MTE1)` is set and waited explicitly, inside `auto_sync()`. The fixpipe writes
L1 and MTE1 reads it back for the second product; that is the pair of pipes the event names.

The operands are integers in [-3, 3], so the first product is at most 288 in magnitude and the FP16
feedback boundary is exact. Every output bit is therefore defined, and the comparison is bitwise.

The planted rows are what make the ReLU observable: `x[0]` is all ones and `y[1]` all minus ones, so
the intermediate at (0, 1) is -32 before the clamp and 0 after; `z[0]` is zero except `z[0, 1] = 1`,
which reads that one cell straight into `output[0, 0]`. A kernel that skipped the ReLU prints -32
there. `check_domain` asserts that witness still exists.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import relu_reuse
from reference import make_inputs, reference

DEVICE = "a5"

POISON = float("nan")     # an unpublished output cell is a NaN, not a plausible product
GEOMETRY = {"M": 16, "N": 16, "P": 16, "K": 32}

OUTPUTS = ("output",)

CASES = [
    {"id": "signed_seed0", "seed": 9800, "block_dim": 1, "parameters": dict(GEOMETRY),
     "purpose": "The complete fixed tile -- M=N=P=16, K=32 -- with signed integer operands. The "
                "planted rows that witness the ReLU are the same in all three cases; the seed "
                "varies only the rest of the data"},
    {"id": "signed_seed1", "seed": 9801, "block_dim": 1, "parameters": dict(GEOMETRY),
     "purpose": "The same geometry on different data"},
    {"id": "signed_seed2", "seed": 9802, "block_dim": 1, "parameters": dict(GEOMETRY),
     "purpose": "And a third, so an answer that is right for one operand alignment is not the "
                "whole evidence"},
]


def check_domain(inputs, expected):
    """The exact-arithmetic domain, and the witness the whole example turns on: without a negative
    intermediate that `z` reads back, an omitted ReLU would not change a single output bit."""
    for name, shape in (("x", (16, 32)), ("y", (16, 32)), ("z", (16, 16))):
        value = inputs[name]
        if value.dtype != torch.float16 or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a complete FP16 {shape} tile")
        if not bool((value.abs() <= 3).all()) or not torch.equal(value.float(),
                                                                 value.float().round()):
            raise ValueError("the operands are integers in [-3, 3]; that is what keeps FP16 exact")
    first = inputs["x"].int() @ inputs["y"].int().T
    if int(first.abs().max()) > 2048:
        raise ValueError("the intermediate must stay exactly representable in FP16")
    if int(first[0, 1]) >= 0:
        raise ValueError("the intermediate cell the output reads back must be negative")
    if not torch.equal(inputs["z"][0], torch.zeros(16, dtype=torch.float16).index_fill_(
            0, torch.tensor([1]), 1.0)):
        raise ValueError("z's first row must select exactly that one intermediate cell")
    if expected["output"][0, 0] != 0:
        raise ValueError("the clamped cell must reach the output as zero, or nothing witnesses ReLU")


def execute(case, inputs, launcher, backend):
    """One launch. The output arrives filled with NaN and seeded in, so a cell the fixpipe never
    published is a NaN rather than a leftover, and the operands are checked to come back unchanged."""
    op = OpExec(relu_reuse, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    before = {name: inputs[name].clone() for name in ("x", "y", "z")}
    produced = op(inputs["x"], inputs["y"], inputs["z"],
                  torch.full((16, 16), POISON, dtype=torch.float32))
    if any(not torch.equal(inputs[name], value) for name, value in before.items()):
        raise ValueError("the launch must not modify its operands")
    return {"output": produced}


def compare(name, got, want):
    """Bitwise: bounded integer products, an exact FP16 feedback boundary and an exact second
    product define every output bit."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:7s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:7s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} FP32 cells; "
          f"ReLU witness output[0,0]={got[0, 0].item():g} (0 with the clamp)")
    if not ok:
        differ = got != want
        index = differ.nonzero()
        unwritten = int(torch.isnan(got).sum())
        print(f"      {int(differ.sum())}/{got.numel()} cells differ, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}, got {got[differ][:3].tolist()} against "
              f"{want[differ][:3].tolist()}"
              + (f"; {unwritten} are still the NaN fill (never published)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:14s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['M']}x{p['N']}x{p['K']} then x{p['P']}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
