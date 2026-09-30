# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three A2/A3 multiply-add precision paths into an initialized accumulator.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case n5121_tile5120 # one of them
    python main.py --device a3           # the same source against the A3 facade
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

    y = c + a * b     computed three ways: FP32->FP32, FP16->FP16, FP16->FP32

`muladddst(dst, a, b)` accumulates into its destination, so the destination *is* the accumulator:
`adds(acc, c, 0.0)` puts `c` there first. This is not a three-operand FMA, and forgetting the
initialization gives `a * b` with whatever was in the slot.

Each path keeps four distinct two-slot `DBuff` allocations: a, b and c are filled by MTE2 and only
the vector pipeline writes the accumulator. Keeping the load and store roles on separate buffers is
the reviewed reuse fix, and the 40,000-element case revisits both slots of all four many times.

The mixed path runs in **repeat mode**: `repeat=ceil(valid/64)` with an FP32 destination stride of 8
and FP16 source strides of 4. Its final repeat may compute scratch lanes through the next multiple
of 64 inside the fixed 5,120-element allocation; only the logical prefix is published.

The three results have separate bounds -- 1e-4 for the all-FP32 path, 2e-3 where the sources are
FP16 -- each with a matching relative L2 bound. The reference multiplies and adds the actual typed
inputs in FP64 and rounds once to the output dtype.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernels
from reference import TYPES, make_inputs, reference

DEVICES = ("a2", "a3")

POISON = -777
LANES_PER_REPEAT = 64

# Per output: the all-FP32 path is an order of magnitude tighter than the two with FP16 sources.
TOLERANCE = {"f32": {"rtol": 1e-04, "atol": 1e-04, "max_relative_l2": 1e-04},
             "f16": {"rtol": 2e-03, "atol": 2e-03, "max_relative_l2": 2e-03},
             "mixed": {"rtol": 2e-03, "atol": 2e-03, "max_relative_l2": 2e-03}}

OUTPUTS = ("f32", "f16", "mixed")

CASES = [
    {"id": "n1024_tile1024", "seed": 103700, "block_dim": 1,
     "purpose": "n equals the tile: one tile, no tail, and the baseline the tail cases are read "
                "against", "parameters": {"n": 1024, "tile_len": 1024, "dataset": "uniform"}},
    {"id": "n549_tile128", "seed": 103701, "block_dim": 1,
     "purpose": "549 elements over 128-element tiles: five tiles with a 37-element last one, so "
                "the tail is neither empty nor a multiple of the 64-lane repeat",
     "parameters": {"n": 549, "tile_len": 128, "dataset": "uniform"}},
    {"id": "n1023_tile256", "seed": 103702, "block_dim": 1,
     "purpose": "One element short of four full tiles: the tail is 255, one lane below the tile",
     "parameters": {"n": 1023, "tile_len": 256, "dataset": "uniform"}},
    {"id": "n777_tile256", "seed": 103703, "block_dim": 1,
     "purpose": "777 over 256: three full tiles and a 9-element tail, far below one repeat",
     "parameters": {"n": 777, "tile_len": 256, "dataset": "uniform"}},
    {"id": "n40000_tile256", "seed": 103704, "block_dim": 1,
     "purpose": "40,000 elements in 157 tiles over two slots: every slot of all four double buffers "
                "is reused about 78 times, which is what makes a load/store role confusion show up "
                "rather than stay lucky",
     "parameters": {"n": 40000, "tile_len": 256, "dataset": "uniform"}},
    {"id": "n1_tile64", "seed": 103705, "block_dim": 1,
     "purpose": "One element in a 64-lane tile, with dyadic values: the shortest possible tail, and "
                "exact arithmetic so the result is checkable without leaning on the budget",
     "parameters": {"n": 1, "tile_len": 64, "dataset": "dyadic"}},
    {"id": "n63_tile64", "seed": 103706, "block_dim": 1,
     "purpose": "63 dyadic elements: one lane short of the repeat boundary",
     "parameters": {"n": 63, "tile_len": 64, "dataset": "dyadic"}},
    {"id": "n64_tile64", "seed": 103707, "block_dim": 1,
     "purpose": "Exactly 64: one full repeat and no scratch lanes computed at all",
     "parameters": {"n": 64, "tile_len": 64, "dataset": "dyadic"}},
    {"id": "n65_tile64", "seed": 103708, "block_dim": 1,
     "purpose": "65 elements: two tiles, the second holding one valid lane, so the mixed path's "
                "second repeat computes 63 scratch lanes that must not be published",
     "parameters": {"n": 65, "tile_len": 64, "dataset": "dyadic"}},
    {"id": "n5121_tile5120", "seed": 103709, "block_dim": 1,
     "purpose": "The maximum tile plus one element: the largest allocation with the smallest "
                "possible second tile, which is where a scratch computation that overran the "
                "5,120-element buffer would be caught",
     "parameters": {"n": 5121, "tile_len": 5120, "dataset": "dyadic"}},
]


def check_domain(inputs, expected):
    """The declared domain, and the property the accumulator check rests on: `c` is not zero, so a
    kernel that computed `a * b` without initializing the accumulator disagrees."""
    tile = inputs["tile_len"]
    if not 64 <= tile <= 5120 or tile % 64:
        raise ValueError("tile_len is a multiple of 64 in [64, 5120]")
    for name, (source, accumulator) in TYPES.items():
        for arg, dtype in (("a", source), ("b", source), ("c", accumulator)):
            value = inputs[f"{name}_{arg}"]
            if value.dtype != dtype or value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(f"{name}_{arg} must be a [1, n] {dtype}")
            if not bool(torch.isfinite(value).all()) or bool((value.abs() > 2).any()):
                raise ValueError("the declared domain is finite values in [-2, 2]")
        if not bool((inputs[f"{name}_c"] != 0).any()):
            raise ValueError(f"{name}_c is all zero, so a missing accumulator initialization "
                             f"would be invisible")
    if (expected["f32"] == POISON).any():
        raise ValueError("the reference contains the poison value")


def execute(case, inputs, launcher, backend, device):
    """Three launches, one per precision path, each with its own kernel from the device's facade.
    Every destination arrives filled with -777 and seeded in, so a tail element no tile published
    reads back as the fill rather than as a plausible small value."""
    outputs = {}
    for name, entry in make_kernels(device).items():
        a, b, c = (inputs[f"{name}_{arg}"] for arg in ("a", "b", "c"))
        op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                    block_dim=case["block_dim"],
                    out_dir=f"tmp/{launcher}/{case['id']}/{name}", seed_outputs=True)
        outputs[name] = op(a, b, c, torch.full_like(c, POISON), c.numel(), inputs["tile_len"])
    return outputs


def compare(name, got, want):
    """Element bound and relative-norm bound, at this path's own tolerance."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:6s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    rule = TOLERANCE[name]
    bounds = {"rtol": rule["rtol"], "atol": rule["atol"]}
    room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
    margin = ((got.double() - want.double()).abs() / room).max().item()
    norm = torch.linalg.vector_norm(want.double().flatten())
    residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
    relative = (residual / norm).item() if norm > 0 else residual.item()
    ok = bool(torch.allclose(got, want, **bounds)) and relative <= rule["max_relative_l2"]
    print(f"    {name:6s} {'ok  ' if ok else 'FAIL'}  allclose={margin:.2f}x  "
          f"rel_l2={relative:.2e}/{rule['max_relative_l2']:g}  over {got.numel()} elements")
    if not ok:
        outside = ~torch.isclose(got, want, **bounds)
        index = outside.nonzero()
        unwritten = int((got[outside] == POISON).sum())
        # A whole repeat's worth of wrong lanes at the end points at the scratch tail.
        last = int(index[-1][1]) if len(index) else 0
        print(f"      {len(index)}/{got.numel()} elements outside the bound; first "
              f"{[tuple(i.tolist()) for i in index[:3]]}, last column {last}"
              + (f"; {unwritten} still hold the {POISON} fill (never published)"
                 if unwritten else "")
              + (f"; all within the final {LANES_PER_REPEAT}-lane repeat"
                 if len(index) and int(index[0][1]) >= got.numel() - LANES_PER_REPEAT else ""))
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (n={p['n']}, tile={p['tile_len']}, {p['dataset']} data, "
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
