# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The A2/A3 cube-to-vector bridge, which goes through GM because there is no direct route.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --device a3          # the same source against the A3 facade
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

A5 can move an L0C accumulator straight into UB. A2 and A3 cannot, so the result goes out to GM and
comes back: the cube writes a 32x16 FP32 product into one slot of a two-slot `GMBuff`, and each vector
subblock reads its own 16 rows back with MTE2 and scales them.

`CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)` names the two pipes at the ends
of that handoff -- the fixpipe writing GM, MTE2 reading it -- and returns the slot only after *both*
vector readers have finished. `auto_sync()` covers each vector's own UB buffers and its store.

Three beats over two slots, so the third beat reuses the first. `GetSubBlockIdx() * 16` is how each
vector participant finds its own half of the product, and the two halves are compared together.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernel
from reference import BEATS, BOUND, DEVICES, ROWS, SCALE, WIDTH, make_inputs, reference

POISON = float("nan")     # a row no vector participant published is a NaN
SLOTS = 2

OUTPUTS = ("o",)

CASES = [
    {"id": "three_beats", "seed": 8601, "block_dim": 1,
     "parameters": {"beats": BEATS, "slots": SLOTS, "M": ROWS, "N": WIDTH, "K": WIDTH},
     "purpose": "The original three beats over two slots: the third beat reuses the first slot, "
                "which is the reuse the mutex has to hold back until both readers are done"},
    {"id": "second_seed", "seed": 8602, "block_dim": 1,
     "parameters": {"beats": BEATS, "slots": SLOTS, "M": ROWS, "N": WIDTH, "K": WIDTH},
     "purpose": "The same program on different data"},
]


def check_domain(inputs, expected):
    """Why the comparison is exact, and that both halves of every product are non-trivial."""
    largest = SCALE * BOUND * BOUND * WIDTH
    if int(expected["o"].abs().max()) > largest:
        raise ValueError(f"no element can exceed {largest:g} in this domain")
    if not torch.equal(expected["o"], expected["o"].round()):
        raise ValueError("every element must be integral, or the operands left their domain")
    half = ROWS // 2
    # Each vector participant owns one half; if a half were all zero it could not tell a missing
    # publication from a correct one.
    for beat in range(BEATS):
        for name, rows in (("upper", slice(0, half)), ("lower", slice(half, ROWS))):
            if not bool(expected["o"][beat, rows].count_nonzero()):
                raise ValueError(f"beat {beat}'s {name} half is all zero, so one participant's "
                                 f"result would be indistinguishable from an unwritten one")


def execute(case, inputs, launcher, backend, device):
    """One launch. The destination arrives NaN-filled and seeded in, and the operands are checked to
    come back unchanged -- the bridge writes GM scratch, and the caller's tensors are not it."""
    op = OpExec(make_kernel(device), launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{device}/{case['id']}",
                seed_outputs=True)
    before = (inputs["x"].clone(), inputs["y"].clone())
    produced = op(inputs["x"], inputs["y"], torch.full((BEATS, ROWS, WIDTH), POISON))
    if not torch.equal(inputs["x"], before[0]) or not torch.equal(inputs["y"], before[1]):
        raise ValueError("the launch must not modify its operands")
    return {"o": produced}


def compare(name, got, want):
    """Bitwise: bounded integer products and an exact doubling."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {BEATS} beats of "
          f"{ROWS}x{WIDTH} FP32 elements")
    if not ok:
        differ = got != want
        half = ROWS // 2
        # Which participant's half went wrong is the first thing to know about this kernel.
        halves = {f"beat {beat} {name}": int(differ[beat, rows].sum())
                  for beat in range(BEATS)
                  for name, rows in (("upper", slice(0, half)), ("lower", slice(half, ROWS)))
                  if bool(differ[beat, rows].any())}
        unwritten = int(torch.isnan(got).sum())
        print(f"      {int(differ.sum())}/{got.numel()} differ, by half: {halves}"
              + (f"; {unwritten} are still the NaN fill (never published)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
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
        print(f"{case['id']}  ({p['beats']} beats over {p['slots']} slots, "
              f"{p['M']}x{p['N']}x{p['K']}, device={args.device}, launcher={args.launcher})")
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
