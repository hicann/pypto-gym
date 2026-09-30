# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete a burst's alignment tail with a literal pad value, and compare the tail too.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case pad_bits       # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Each row of the 4x16 FP32 output is one `gm_to_ub_pad` transfer of nine elements -- 36 bytes --
into a 64-byte UB tile. The 28 bytes that finish the tile are not copied from anywhere: the
instruction carries `pad=-1.5`, which the backend puts in the `set_mov_pad_val` SPR, and without
it the tail would keep whatever the transfer found in that UB. So the pad is mode state around
the transfer, not an operand of it.

The comparison covers all 16 columns, the seven padded ones included, and it is bitwise: -1.5 is
0xBFC00000, a bit pattern no input here carries, so a tail that was never written and a tail
written with the wrong value are two different failures rather than one rounding question.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import BURST, PAD, ROWS, TILE, dma_pad_value
from reference import make_inputs, reference

DEVICE = "a5"

OUTPUTS = ("o",)

CASES = [
    {"id": "pad_bits", "seed": 8711, "block_dim": 1,
     "purpose": "The nine-element burst and the 28-byte tail that completes its tile: all 64 "
                "output bytes of every row are compared, so the pad is checked as a bit pattern "
                "instead of being assumed",
     "parameters": {}},
    {"id": "different_input", "seed": 8712, "block_dim": 1,
     "purpose": "The same kernel over a shifted input. The pad is a literal in the instruction "
                "rather than anything derived from the data, and this case is what shows the "
                "seven padded columns did not move when the nine copied ones did",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """Two statements the cases rest on. The pad has to be absent from the input, or a tail that
    was never written would compare equal to one that was; and the copied region has to be the
    part the kernel actually bursts, or the comparison would not separate the two."""
    x = inputs["x"]
    if x.shape != (ROWS, TILE) or x.dtype != torch.float32:
        raise ValueError(f"the input must be float32[{ROWS}, {TILE}]")
    if (x == PAD).any():
        raise ValueError(f"the input carries the pad value {PAD}, which the comparison relies on "
                         f"being absent")
    if not torch.equal(expected["o"][:, :BURST], x[:, :BURST]):
        raise ValueError("the reference must copy the burst region unchanged")
    if not (expected["o"][:, BURST:] == PAD).all():
        raise ValueError(f"the reference must fill columns {BURST}: with {PAD}")


def execute(case, inputs, launcher, backend):
    """One launch for the whole tile. The destination arrives NaN-poisoned and is seeded into the
    launch, so a column neither the burst nor the pad reached reads back as NaN."""
    op = OpExec(dma_pad_value, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full((ROWS, TILE), float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: the pad is a bit pattern the instruction carries, and the copied region is a
    transfer. Neither has anything a tolerance could absorb."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} elements "
          f"({BURST} copied + {TILE - BURST} padded per row)")
    if not ok:
        outside = raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
        index = outside.nonzero().flatten()
        positions = [divmod(i, TILE) for i in index.tolist()]
        in_pad = sum(1 for _, col in positions if col >= BURST)
        poisoned = int(torch.isnan(got.cpu().reshape(-1)[index]).sum())
        print(f"      {len(index)}/{got.numel()} elements differ, {in_pad} of them in the pad "
              f"tail, {poisoned} still NaN-poisoned (never written); first {positions[:3]}")
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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (seed={case['seed']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
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
