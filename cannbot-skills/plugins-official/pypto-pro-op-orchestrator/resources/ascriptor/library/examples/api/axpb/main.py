# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The smallest complete vector kernel: load two rows, scale one, add, store.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card
    python main.py --inspect            # the kernel's identity and its surface IR

`o = 2 * x + y` over one row of 64 FP32 lanes, on one vector core. Three instructions in the `@vf`:
two loads into registers, `muls` by the literal 2.0, `add`, one store. If you are looking for the
shortest path from a Python function to something that runs, this is it.

The comparison is bitwise, and that is not luck: a multiply by two is exact in FP32 and the add is a
single rounding, so the host computing `2 * x + y` performs exactly the two operations the register
does, in the same order. `mixed_magnitudes` spreads the operands over 64 binary exponents to say that
the exactness is a property of the operations rather than of small numbers.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse
import math

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import axpb
from reference import LANES, make_inputs, reference

DEVICE = "a5"

POISON = float("nan")     # an unwritten lane is a NaN, which no sum of finite operands produces

OUTPUTS = ("o",)

CASES = [
    {"id": "default", "seed": 7101, "block_dim": 1, "parameters": {"spread": "normal"},
     "purpose": "The original composition: one row of 64 lanes, operands around three in magnitude"},
    {"id": "second_seed", "seed": 7102, "block_dim": 1, "parameters": {"spread": "normal"},
     "purpose": "The same program on different data"},
    {"id": "mixed_magnitudes", "seed": 7103, "block_dim": 1, "parameters": {"spread": "mixed"},
     "purpose": "One binary exponent per lane, from 2**-32 to 2**31, with the two operands' scales "
                "running opposite ways. The answer is still exact everywhere, which is what says "
                "the bitwise comparison rests on the operations and not on the numbers being small"},
]


def check_domain(inputs, expected):
    """What makes a bitwise comparison of floating point legitimate here."""
    doubled = inputs["x"].double() * 2
    if not torch.equal(doubled.float(), inputs["x"] * 2):
        raise ValueError("the multiply by two must be exact in FP32, or the answer has two roundings")
    if not bool(torch.isfinite(expected["o"]).all()):
        raise ValueError("the sum must stay finite, or the poison and the answer could be confused")
    if bool((expected["o"] == 0).all()):
        raise ValueError("an all-zero reference would make the comparison prove nothing")


def execute(case, inputs, launcher, backend, timeout=180.0):
    """One launch. The destination arrives filled with NaN and seeded in, so a lane the kernel never
    wrote reads back as a NaN rather than a plausible sum."""
    op = OpExec(axpb, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                timeout=timeout, seed_outputs=True)
    return {"o": op(inputs["x"], inputs["y"], torch.full((1, LANES), POISON))}


def compare(name, got, want):
    """Bitwise: the same two operations in the same order give the same bits."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} FP32 lanes")
    if not ok:
        differ = (got != want).flatten()
        index = differ.nonzero().flatten().tolist()
        unwritten = int(torch.isnan(got).sum())
        print(f"      {len(index)}/{got.numel()} lanes differ, first at {index[:6]}: got "
              f"{got.flatten()[index[:3]].tolist()} against {want.flatten()[index[:3]].tolist()}"
              + (f"; {unwritten} are still the NaN fill (never written)" if unwritten else ""))
    return ok


def inspect(out_dir):
    """What this kernel is, and the IR the frontend made of it. Nothing is compiled or run."""
    from pathlib import Path

    from ascriptor.ir import print_module

    module = axpb.ir()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "surface.ir").write_text(print_module(module))
    locations = sorted({str(op.loc) for op in module.walk() if op.loc is not None})
    print(f"kernel   {axpb.name}\ndevice   {axpb.device}\nmode     {axpb.mode}\n"
          f"ir       {module.attrs['ir']}\nwritten  {directory / 'surface.ir'}")
    print(f"source   {len(locations)} located operations, {locations[0]} first")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="seconds for one launch, forwarded to OpExec")
    parser.add_argument("--inspect", action="store_true",
                        help="print the kernel's identity and write its surface IR, then exit")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:18s} {case['purpose']}")
        return 0
    if args.inspect:
        return inspect("tmp/inspect")
    # The runtime accepts any launcher with any backend; this pairing is the one that cannot work,
    # because the pypto launcher runs the sources only the pypto_pro backend prints.
    if args.launcher == "pypto" and args.backend != "pypto_pro":
        parser.error("the pypto launcher requires --backend pypto_pro")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({case['parameters']['spread']} magnitudes, {LANES} lanes, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.timeout)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
