# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A non-contiguous GM view: a logical shape over physical strides, counted in elements.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

`x.flatten().view([2, 16], [128, 2], offset=3)` reads 32 elements out of 256. Three numbers decide
which, and all three are in *elements* rather than bytes: the logical shape, the stride per axis, and
the offset where it starts. The last element the view touches is 161 -- `check_domain` computes that
from the four numbers rather than trusting the arithmetic.

A non-unit innermost stride is supported for a read. The matching write is a gap, so this example
loads through the view and stores a complete contiguous tile.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import strided_views
from reference import COLUMNS, OFFSET, ROWS, SHAPE, STRIDES, last_element_read, make_inputs, reference

DEVICE = "a5"

POISON = float("nan")     # an element the view never delivered is a NaN

OUTPUTS = ("o",)

CASES = [
    {"id": "every_other", "seed": 8401, "block_dim": 1,
     "parameters": {"shape": list(SHAPE), "strides": list(STRIDES), "offset": OFFSET},
     "purpose": "The declared view: two rows 128 elements apart, every other element, from element "
                "3. Reading it as contiguous, or from element 0, gives a different answer in every "
                "lane rather than in one"},
    {"id": "second_seed", "seed": 8402, "block_dim": 1,
     "parameters": {"shape": list(SHAPE), "strides": list(STRIDES), "offset": OFFSET},
     "purpose": "The same view on different data"},
]


def check_domain(inputs, expected):
    """Where the view actually reaches, and that it is not a contiguous read in disguise."""
    flat = inputs["x"].reshape(-1)
    if last_element_read() != OFFSET + (SHAPE[0] - 1) * STRIDES[0] + (SHAPE[1] - 1) * STRIDES[1]:
        raise ValueError("the reference's own arithmetic must agree with the declared view")
    if last_element_read() >= flat.numel():
        raise ValueError(f"the view reads element {last_element_read()} of {flat.numel()}")
    contiguous = flat[OFFSET:OFFSET + SHAPE[0] * SHAPE[1]].reshape(SHAPE)
    if torch.equal(expected["o"], contiguous):
        raise ValueError("a contiguous read gives the same answer here, so the strides prove nothing")
    if torch.equal(expected["o"], flat[:SHAPE[0] * SHAPE[1]].reshape(SHAPE)):
        raise ValueError("the offset makes no difference here, so it is not being observed")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with NaN and seeded in."""
    op = OpExec(strided_views, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full(SHAPE, POISON))}


def compare(name, got, want):
    """Bitwise: a strided read moves bits without touching them."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} FP32 elements "
          f"drawn from {ROWS * COLUMNS}")
    if not ok:
        differ = (got != want).flatten()
        index = differ.nonzero().flatten().tolist()
        unwritten = int(torch.isnan(got).sum())
        print(f"      {len(index)}/{got.numel()} differ, first at {index[:6]}"
              + (f"; {unwritten} are still the NaN fill (never delivered)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
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
        print(f"{case['id']}  (view {p['shape']} strides {p['strides']} from element {p['offset']}, "
              f"last element {last_element_read()}, launcher={args.launcher})")
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
