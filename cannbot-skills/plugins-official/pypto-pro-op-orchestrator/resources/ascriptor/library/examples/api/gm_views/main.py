# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read a GM window by element strides, and write one that overlaps what was already written.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case overlap        # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Three views over one 4x48 FP32 input. `x.view(shape, strides=..., offset=...)` counts in
elements, not bytes, and describes a window of the padded rows rather than a copy of them: the
gather case steps every fourth element, the rank3 case lands a three-dimensional window
row-major in a [4, 8] destination, and the overlap case writes the whole output and then
rewrites columns 16-47 from a window of the input shifted by eight.

Every output byte is defined, so the comparison is bitwise with no tolerance: these are
transfers, and a transfer that rounded anything would be a defect rather than a difference.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import make_inputs, reference

DEVICE = "a5"

OUTPUTS = ("o",)

KERNELS = {"overlap": "gm_view", "gather": "gm_view_gather", "rank3": "gm_view_rank3"}

CASES = [
    {"id": "overlap", "seed": 8701, "block_dim": 1,
     "purpose": "Two writes to the same GM buffer, the second one landing on columns 16-47 of "
                "what the first wrote: the whole output is compared, so a second burst that "
                "arrived before the first would show up as the baseline surviving where the "
                "shifted window should be",
     "parameters": {"variant": "overlap", "cols": 48}},
    {"id": "gather", "seed": 8702, "block_dim": 1,
     "purpose": "A stride-4 read of each padded row -- 12 of the 48 columns -- into a UB tile "
                "declared 16 elements wide, because the UB port steps whole 32-byte blocks and "
                "12 FP32 values are not one",
     "parameters": {"variant": "gather", "cols": 12}},
    {"id": "rank3", "seed": 8703, "block_dim": 1,
     "purpose": "A rank-three window, strides [96, 16, 1], landing row-major in [4, 8]: the "
                "reference reaches the same 32 elements as four flat slices at 0, 16, 96 and "
                "112, which is the arithmetic the view is claiming to do",
     "parameters": {"variant": "rank3", "cols": 8}},
]


def check_domain(inputs, expected):
    """What the cases assume of the input, checked rather than promised: a non-contiguous or
    differently shaped `x` would make every stride in every view mean something else."""
    if inputs["variant"] not in KERNELS:
        raise ValueError(f"unknown variant {inputs['variant']!r}")
    x = inputs["x"]
    if x.shape != (4, 48) or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("the view cases require a contiguous float32[4, 48] input")
    if expected["o"].dtype != torch.float32:
        raise ValueError("the reference must stay in FP32")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so a
    column no burst reached reads back as NaN rather than as a plausible value."""
    entry = getattr(kernel, KERNELS[inputs["variant"]])
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((4, case["parameters"]["cols"]), float("nan"))
    return {"o": op(inputs["x"], destination)}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise. Every byte of the output is a copied or padded byte with an exact reference, so
    there is nothing here a tolerance could legitimately absorb."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} elements")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
        index = outside.nonzero().flatten()
        poisoned = int(torch.isnan(got.cpu().reshape(-1)[index]).sum()) if index.numel() else 0
        note = f", {poisoned} still NaN-poisoned (never written)" if poisoned else ""
        print(f"      {len(index)}/{got.numel()} elements differ{note}; first at "
              f"{[divmod(i, got.shape[1]) for i in index[:3].tolist()]}")
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
            print(f"{case['id']:8s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (cols={case['parameters']['cols']}, launcher={args.launcher}, "
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
