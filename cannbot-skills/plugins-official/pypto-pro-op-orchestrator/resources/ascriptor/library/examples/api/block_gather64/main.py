# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gather eight 32-byte blocks by byte offset into 64-bit lanes, and compare every byte.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case repeated_blocks   # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

`ub_to_reg_gatherb` takes an index register and gathers 32-byte blocks. The first eight lanes of
that register are the ones it reads, and each holds a **byte** offset -- not an int64 element index.
A 512-byte source is therefore sixteen candidate blocks, the eight selected ones become 32 int64
output lanes, and the host multiplies its block numbers by 32 before handing them over.

The index arrives as an INT32 tensor and is read through `reinterpret(idx_s, DT.uint32)`, because
the gather wants unsigned offsets. Source values span +/-2^40, so both 32-bit halves of every lane
carry information, and the destination arrives filled with 0x1234567812345678 -- a pattern whose
two halves are both non-zero, so a lane nothing wrote cannot be mistaken for a gathered one.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import gatherb64
from reference import make_inputs, reference

DEVICE = "a5"

BLOCK_BYTES = 32           # the gather's granularity: four int64 lanes
BLOCKS = 8                 # the index lanes the instruction reads
POISON = 0x1234567812345678

OUTPUTS = ("o",)

CASES = [
    {"id": "permuted_blocks", "seed": 8751, "block_dim": 1,
     "purpose": "Eight distinct blocks in a random order out of the sixteen available: a gather "
                "that ignored its index and copied the first eight blocks in place would have to "
                "be lucky to pass, and this permutation is not the identity",
     "parameters": {"repeated": False}},
    {"id": "repeated_blocks", "seed": 8752, "block_dim": 1,
     "purpose": "Blocks 15, 0, 15, 7, 7, 8, 1, 14 -- two of them twice: repetition is legal and "
                "the reference predicts it, so a gather that deduplicated its offsets, or that "
                "wrote a lane it had already filled, shows up as a whole wrong block",
     "parameters": {"repeated": True}},
]


def check_domain(inputs, expected):
    """The offsets the instruction accepts, and the two facts that make a failure legible: the
    poison is not a value the source could produce, and the source spans both 32-bit halves."""
    offsets = inputs["indices"][0, :BLOCKS]
    source = inputs["src"]
    limit = source.numel() * source.element_size() - BLOCK_BYTES
    if bool(((offsets < 0) | (offsets > limit) | (offsets % BLOCK_BYTES != 0)).any()):
        raise ValueError(f"block offsets must be {BLOCK_BYTES}-byte aligned in [0, {limit}], "
                         f"got {offsets.tolist()}")
    if (source == POISON).any():
        raise ValueError("the source carries the poison value")
    if not bool((source.abs() > 2**32).any()):
        raise ValueError("the source must span both 32-bit halves, or a gather that moved only "
                         "the low words would pass")
    if expected["o"].numel() != BLOCKS * BLOCK_BYTES // source.element_size():
        raise ValueError("the reference must be as long as the eight gathered blocks")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with the poison and seeded into the launch."""
    op = OpExec(gatherb64, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((1, 32), POISON, dtype=torch.int64)
    return {"o": op(inputs["src"], inputs["indices"], destination)}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: a gather moves bytes, and both 32-bit halves of every lane are compared."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} int64 lanes "
          f"({BLOCKS} blocks)")
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        index = (flat_got != flat_want).nonzero().flatten().tolist()
        blocks = sorted({i // (BLOCK_BYTES // got.element_size()) for i in index})
        unwritten = [i for i in index if flat_got[i].item() == POISON]
        print(f"      {len(index)}/{got.numel()} lanes differ, in output blocks {blocks}"
              + (f"; {len(unwritten)} lanes still hold the poison (never gathered)"
                 if unwritten else ""))
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
            print(f"{case['id']:17s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        inputs = make_inputs(case)
        blocks = (inputs["indices"][0, :BLOCKS] // BLOCK_BYTES).tolist()
        print(f"{case['id']}  (blocks={blocks}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
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
