# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two shared A2/A3 vector forms: a counted tail, and a packed mask driving a select.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --device a3          # the same source against the A3 facade
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

`tail` allocates 128 lanes and operates on 70. `count=` is how a vector instruction is told the
logical extent, and the 58 lanes past it are allocated, transferred and never computed. Choosing 70
rather than 64 or 128 is deliberate: it is neither a full tile nor a round fraction of one.

`select` builds its predicate as a *packed* mask -- one bit per lane, so 64 lanes fit in a
`DT.uint8[1, 32]` tile -- and `select` consumes it with a scratch address buffer. The answer is an
element-wise maximum, which is what makes a mask read in the wrong direction visible in half the
lanes rather than in none.

The facade is an argument, so the same source text is the A2 and the A3 kernel.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernels
from reference import CAPACITY, DEVICES, MODES, make_inputs, reference

POISON = float("nan")     # a lane the kernel never wrote is a NaN

OUTPUTS = ("o",)

CASES = [
    {"id": "counted_tail", "seed": 8201, "block_dim": 1, "parameters": {"mode": "tail"},
     "purpose": f"70 live lanes in a {CAPACITY}-lane allocation: `count=` states the logical extent "
                f"and the rest is never computed. A kernel that operated on the whole tile would "
                f"read the untouched staging lanes"},
    {"id": "counted_tail_second_seed", "seed": 8202, "block_dim": 1, "parameters": {"mode": "tail"},
     "purpose": "The same counted form on different data"},
    {"id": "packed_select", "seed": 8203, "block_dim": 1, "parameters": {"mode": "select"},
     "purpose": "64 lanes of predicate packed one bit each into 32 bytes, then used to choose "
                "between two tensors. The answer is an element-wise maximum, so a mask read the "
                "wrong way round is wrong in about half the lanes"},
    {"id": "packed_select_second_seed", "seed": 8204, "block_dim": 1, "parameters": {"mode": "select"},
     "purpose": "The same select on different data"},
]


def check_domain(inputs, expected):
    """What each mode needs to be observable at all."""
    mode = inputs["mode"]
    lanes = MODES[mode]
    if expected["o"].shape != (1, lanes):
        raise ValueError(f"{mode} publishes exactly its {lanes} logical lanes")
    if mode == "tail":
        if lanes >= CAPACITY or CAPACITY % lanes == 0:
            raise ValueError("the live count must be a partial, non-dividing fraction of the tile")
    else:
        chosen = inputs["x"] > inputs["y"]
        # Both branches have to be taken, or a select that always picked one side would pass.
        if not (bool(chosen.any()) and bool((~chosen).any())):
            raise ValueError("the predicate must select from both sides in this case")
        if not torch.equal(expected["o"], torch.maximum(inputs["x"], inputs["y"])):
            raise ValueError("the reference must be the element-wise maximum")


def execute(case, inputs, launcher, backend, device):
    """One launch, on the facade the run asked for. The destination arrives NaN-filled and seeded in,
    so a lane past the count -- or one the mask skipped -- reads back as a NaN."""
    mode = inputs["mode"]
    entry = kernels(device)[mode]
    op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{device}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], inputs["y"],
                    torch.full((1, MODES[mode]), POISON))}


def compare(name, got, want):
    """Bitwise. `tail` is one exact multiply and one add; `select` moves an operand without touching
    it. Neither has anything for a tolerance to absorb."""
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
        print(f"      {len(index)}/{got.numel()} lanes differ, first at {index[:6]}"
              + (f"; {unwritten} are still the NaN fill (never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:26s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        mode = case["parameters"]["mode"]
        print(f"{case['id']}  ({mode}, {MODES[mode]} of {CAPACITY} lanes, device={args.device}, "
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
