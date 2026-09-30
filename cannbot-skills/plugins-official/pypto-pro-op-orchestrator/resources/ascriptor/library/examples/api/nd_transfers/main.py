# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five ND transfers in one launch: two paddings, a three-loop row selection, a transpose.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case all_paths      # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`gm_to_ub_nd_dma` takes three lists -- source strides, destination strides and iteration counts
-- with **loop 0 innermost** and every stride in elements. On top of that it takes padding: a
per-loop `loop_left_pad` / `loop_right_pad`, or a `config_left_pad` / `config_right_pad` that
applies to every loop, filled either with `constant_value` or, with `nearest_value_mode`, by
repeating the nearest real element of that loop.

One kernel body drives all five destinations, and all five are compared whole -- outer padded
rows and columns included -- against five independent Torch expressions. The comparison is
bitwise: these are transfers and literal pad values, so there is nothing a tolerance could
legitimately absorb, and the BF16 destination is checked in its stored width rather than after a
widening that would hide a wrong rounding.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import nd_dma_pad
from reference import make_inputs, reference

DEVICE = "a5"

# Every destination, with the shape and dtype the kernel signature declares. The order is the
# kernel's return order, and `execute` zips it -- a destination added to one and not the other
# would raise rather than silently pair the wrong tensors.
DESTINATIONS = (("constant", (16, 32), torch.float32),
                ("nearest", (16, 32), torch.float32),
                ("rows", (8, 32), torch.float32),
                ("transpose", (32, 16), torch.float32),
                ("bf16_pad", (32, 64), torch.bfloat16))

OUTPUTS = tuple(name for name, _, _ in DESTINATIONS)

CASES = [
    {"id": "all_paths", "seed": 8791, "block_dim": 1,
     "purpose": "All five transfers in one launch: constant padding, edge replication, a "
                "three-loop row selection, the transpose sugar and a BF16 config pad, each "
                "compared over its whole destination including the padded border",
     "parameters": {}},
    {"id": "second_seed", "seed": 8792, "block_dim": 1,
     "purpose": "The same five transfers over a different input. The pad values (-1.5 and 2.0), "
                "the selected rows and the transposed geometry are properties of the "
                "instructions rather than of the data, and this case is what shows none of them "
                "followed it",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """What the five references assume: the two inputs' shapes and dtypes, and that every
    destination the kernel declares has a reference of the declared shape and width. A reference
    that widened its BF16 result would compare a different dtype than the kernel stores."""
    if inputs["x"].shape != (32, 64) or inputs["x"].dtype != torch.float32:
        raise ValueError("the FP32 input must be float32[32, 64]")
    if inputs["xh"].shape != (32, 64) or inputs["xh"].dtype != torch.bfloat16:
        raise ValueError("the BF16 input must be bfloat16[32, 64]")
    for name, shape, dtype in DESTINATIONS:
        if name not in expected:
            raise ValueError(f"the reference does not name {name}")
        if expected[name].shape != shape or expected[name].dtype != dtype:
            raise ValueError(f"reference {name} is {expected[name].dtype}"
                             f"{tuple(expected[name].shape)}, not {dtype}{shape}")


def execute(case, inputs, launcher, backend):
    """One launch for all five destinations. Each arrives NaN-poisoned and is seeded into the
    launch, so a padded border no transfer reached reads back as NaN rather than as a plausible
    pad value -- which is the failure this unit is most likely to see."""
    op = OpExec(nd_dma_pad, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full(shape, float("nan"), dtype=dtype) for _, shape, dtype in DESTINATIONS]
    result = op(inputs["x"], inputs["xh"], *poisoned)
    return dict(zip(OUTPUTS, result, strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise, in the destination's own width."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:10s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:10s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} "
          f"{str(got.dtype).removeprefix('torch.')} elements")
    if not ok:
        width = got.element_size()
        outside = raw(got).view(-1, width).ne(raw(want).view(-1, width)).any(dim=1)
        index = outside.nonzero().flatten()
        poisoned = int(torch.isnan(got.cpu().float().reshape(-1)[index]).sum())
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
            print(f"{case['id']:14s} {case['purpose']}")
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
