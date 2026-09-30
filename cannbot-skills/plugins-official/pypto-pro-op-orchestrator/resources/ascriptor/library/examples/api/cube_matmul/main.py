# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One 16x16 FP16 cube tile, and the same source text on all four device facades.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --device a3          # the same kernel against another facade
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

`o = x @ y.T` at the smallest complete cube geometry: M = N = K = 16. The L1 operands are NZ tiles,
the accumulator is L0C, and the destination is a complete ND FP32 matrix. If you are looking for the
shortest cube kernel, this is it.

The facade is a factory argument -- `make_kernel(device)` imports `ascriptor.<device>` -- so the same
kernel body is A2, A3, A5 or A5PR depending on what you ask for. That is the thing worth taking from
this example: a kernel this simple is portable across the four, and `--device` runs it on each.

The operands are integers in [-4, 4] with K = 16, so every sum is at most 4096 and exact in FP32. The
comparison is a byte comparison for that reason and `check_domain` says so.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernel
from reference import BOUND, DEVICES, SIZE, make_inputs, reference

POISON = float("nan")     # an element the cube never published is a NaN

OUTPUTS = ("o",)

CASES = [
    {"id": "integers", "seed": 8501, "block_dim": 1, "parameters": {"M": SIZE, "N": SIZE, "K": SIZE},
     "purpose": "The original composition: integer-valued FP16 operands in [-4, 4] at the smallest "
                "complete tile, so every sum is exact and the comparison can be bitwise"},
    {"id": "second_seed", "seed": 8502, "block_dim": 1, "parameters": {"M": SIZE, "N": SIZE, "K": SIZE},
     "purpose": "The same program on different data, so a product that happened to be right for one "
                "operand alignment is not the whole evidence"},
]


def check_domain(inputs, expected):
    """Why a matrix product can be compared bit for bit here."""
    largest = BOUND * BOUND * SIZE
    if int(expected["o"].abs().max()) > largest:
        raise ValueError(f"no element can exceed {largest} in this domain")
    if largest > 2 ** 24:
        raise ValueError("the sums must stay inside the FP32 exact-integer range")
    if not torch.equal(expected["o"], expected["o"].round()):
        raise ValueError("every element must be integral, or the operands left their domain")
    if bool((expected["o"] == 0).all()):
        raise ValueError("an all-zero product would make the comparison prove nothing")


def execute(case, inputs, launcher, backend, device):
    """One launch, on the facade the run asked for. The destination arrives NaN-filled and seeded in."""
    op = OpExec(make_kernel(device), launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{device}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], inputs["y"], torch.full((SIZE, SIZE), POISON))}


def compare(name, got, want):
    """Bitwise: bounded integer operands and a 16-term sum are exact in FP32."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {SIZE}x{SIZE} FP32 elements")
    if not ok:
        differ = got != want
        index = differ.nonzero()
        unwritten = int(torch.isnan(got).sum())
        print(f"      {int(differ.sum())}/{got.numel()} differ, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}: got {got[differ][:3].tolist()} against "
              f"{want[differ][:3].tolist()}"
              + (f"; {unwritten} are still the NaN fill (never published)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--device", default="a5", choices=DEVICES)
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:14s} {case['purpose']}")
        print("\ndevices: " + ", ".join(DEVICES))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['M']}x{p['N']}x{p['K']}, device={args.device}, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.device)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
