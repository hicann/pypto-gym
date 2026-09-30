# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 gathers addressed by byte offset -- one element each, or eight values a block.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case blocks_8981    # one of them
    python main.py --device a3           # the same source against the A3 facade
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`gather(dst, src, offsets, ...)` reads 64 four-byte-aligned addresses, one FP32 each.
`gather_block(dst, src, offsets, ...)` reads eight 32-byte-aligned addresses and publishes all eight
FP32 values of every selected block.

Both take **byte offsets**, which is a different contract from A5's register-index gather. That is
the thing to carry away from this example: an offset of 4 is the second element, not the fifth.

The instruction does not validate its addresses, so `check_domain` does: misaligned or out-of-bounds
offsets are refused before the launch, and in block mode only the first eight lanes are addresses at
all -- the rest are ignored rather than treated as sources.

Source values are a permutation of 1000..1127, so every gathered element identifies where it came
from, and the comparison is bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_gather
from reference import make_inputs, reference

DEVICES = ("a2", "a3")

SOURCE_BYTES = 512      # 128 FP32
LANES = 64
MODES = {"elements": (4, LANES), "blocks": (32, 8)}   # alignment, and how many offsets are read

OUTPUTS = ("o",)

CASES = [
    {"id": f"{mode}_{seed}", "seed": seed, "block_dim": 1,
     "purpose": ("64 four-byte-aligned offsets, including 0 and 508 -- the first and last legal "
                 "element address -- with repeats among the rest, so a gather that ignored its "
                 "offsets could not pass"
                 if mode == "elements" else
                 "Eight 32-byte-aligned offsets, including 0 and 480, and all eight FP32 values of "
                 "every selected block are published. A block gather that published only each "
                 "block's first value fills one eighth of the output")
                + (" A second draw of the same shape." if seed == 8982 else ""),
     "parameters": {"mode": mode}}
    for mode in MODES
    for seed in (8981, 8982)
]


def check_domain(inputs, expected):
    """The address checks the instruction does not do. Only the lanes the mode actually reads are
    checked -- in block mode the other 56 are not addresses and must not be treated as any."""
    x, offsets = inputs["x"], inputs["offsets"]
    mode = inputs["mode"]
    if x.dtype != torch.float32 or tuple(x.shape) != (1, SOURCE_BYTES // 4):
        raise ValueError(f"the source must be float32[1, {SOURCE_BYTES // 4}]")
    if offsets.dtype != torch.uint32 or tuple(offsets.shape) != (1, LANES):
        raise ValueError(f"the offsets must be uint32[1, {LANES}]")
    if mode not in MODES:
        raise ValueError(f"unknown gather mode {mode!r}")
    alignment, used = MODES[mode]
    addresses = offsets[0, :used].long()
    if bool((addresses % alignment).any()):
        raise ValueError(f"{mode} offsets must be {alignment}-byte aligned")
    if bool((addresses + alignment > SOURCE_BYTES).any()):
        raise ValueError(f"every {mode} read must land inside the {SOURCE_BYTES} source bytes")
    if len(set(x.flatten().tolist())) != x.numel():
        raise ValueError("the source values must be distinct, so a gathered element says where it "
                         "came from")


def execute(case, inputs, launcher, backend, device):
    """One launch. The destination arrives NaN-poisoned and is seeded in, so a lane the gather never
    wrote -- which is what a block gather publishing one value per block leaves behind -- reads back
    as NaN."""
    entry = make_gather(inputs["mode"], device)
    op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], inputs["offsets"], torch.full((1, LANES), float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: a gather is an exact copy of selected input elements."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {LANES} FP32 lanes")
    if not ok:
        got, want = got.cpu().flatten(), want.cpu().flatten()
        index = (got != want).nonzero().flatten().tolist()
        poisoned = [i for i in index if got[i] != got[i]]
        # Source values are 1000 + position, so the value names the element that arrived.
        arrived = [(i, int(got[i]) - 1000 if got[i] == got[i] else None) for i in index[:4]]
        print(f"      {len(index)}/{LANES} lanes differ; (lane, source position that arrived) "
              f"{arrived}"
              + (f"; {len(poisoned)} still NaN-poisoned (never gathered)" if poisoned else ""))
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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        mode = case["parameters"]["mode"]
        alignment, used = MODES[mode]
        print(f"{case['id']}  ({mode}: {used} offsets, {alignment}-byte aligned, "
              f"device={args.device}, launcher={args.launcher})")
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
