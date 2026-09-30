# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the six A2/A3 block32-causal attention schedules through OpExec and check them.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same sources through the a3 facade
    python main.py --list
    python main.py --case v5_ring_wrap --launcher pipesim
    python main.py --launcher aclnn --case v6_source_smoke

The mask is `key // 32 <= query // 32`, top-left aligned: a block staircase, so every query
in one 32-token block sees the same key-block prefix. Scores, running statistics and output
are FP32; the probabilities are rounded to FP16 before the cube PV.

Six variants implement that one algorithm with six schedules -- v1 a two-slot lag-1
partition, v2 a next-query prefetch, v3 a lag-3 snake, v4 four-key groups, v5 those groups
plus an eight-tile V preload ring, v6 one continuous cross-M-tile stream. Every case names
one variant and changes nothing else, so the four shapes below (aligned, tail, idle cores,
ring wrap) mean the same thing for all six and the results are directly comparable.

The kernel publishes the row statistics padded to a multiple of eight FP32 values per head,
to keep private heads off each other's 32-byte GM lines; `execute` slices them back to the
compact public layout on the host.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# `out` is not bitwise: the probabilities are rounded to FP16 before PV, and the FP32 online
# state is rescaled once per key tile, so the kernel's summation order is its tiling and the
# reference sums the whole row at once. The variants do not even agree with each other to the
# last bit -- v4, v5 and v6 take several key tiles under a single maximum, and rounding P
# against a different maximum is a different rounding -- which is the mathematical formula
# preserved and the finite-precision result changed. The residual bound is what rejects a
# schedule that silently dropped a key group and still landed inside the elementwise bounds.
#
# rowmax and rowsum are the FP32 row state captured before any probability rounding, so they
# are held an order of magnitude tighter -- they are the checkpoint that tells you whether a
# wrong `out` came from the softmax or from PV.
TOLERANCE = {"default": {"atol": 0.005, "rtol": 0.005, "max_relative_l2": 0.005},
             "rowmax": {"atol": 1e-05, "rtol": 1e-05, "max_relative_l2": 1e-05},
             "rowsum": {"atol": 1e-05, "rtol": 1e-05, "max_relative_l2": 1e-05}}

# The same four shapes run against every variant -- 128x128 aligned, a 129x133 double tail,
# two 65-query heads on three cores, and 769x769 so the workspace ring wraps -- plus the
# original 257-token boundary case for all six and the 2177-token sweep for v5 and v6.
CASES = [
    {"id": "v1_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full 128x128 tile on one core: v1's two-slot lag-1 partition with no tail",
     "parameters": {"variant": "v1", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v1_tail", "seed": 43, "block_dim": 2,
     "purpose": "A one-row M tail against a five-column N tail, both masked by v1",
     "parameters": {"variant": "v1", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v1_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two 65-query heads over three cores: the third core owns nothing",
     "parameters": {"variant": "v1", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v1_ring_wrap", "seed": 45, "block_dim": 1,
     "purpose": "769 tokens on one core: v1's two workspace slots are reused many times over",
     "parameters": {"variant": "v1", "BH": 1, "S1": 769, "S2": 769, "D": 128}},

    {"id": "v2_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full tile: v2 prefetches a next query that does not exist yet",
     "parameters": {"variant": "v2", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v2_tail", "seed": 43, "block_dim": 2,
     "purpose": "Double tail with a next-query prefetch in flight across the boundary",
     "parameters": {"variant": "v2", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v2_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two heads over three cores: the prefetch must not cross a head boundary",
     "parameters": {"variant": "v2", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v2_ring_wrap", "seed": 45, "block_dim": 2,
     "purpose": "769 tokens over two cores: split workspace and independent producer/consumer counters",
     "parameters": {"variant": "v2", "BH": 1, "S1": 769, "S2": 769, "D": 128}},

    {"id": "v3_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full tile: v3's lag-3 snake has nothing to run ahead of",
     "parameters": {"variant": "v3", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v3_tail", "seed": 43, "block_dim": 2,
     "purpose": "Double tail under the deeper lag: three tiles are in flight at the boundary",
     "parameters": {"variant": "v3", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v3_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two heads over three cores, one idle, with a lag-3 fill and drain each",
     "parameters": {"variant": "v3", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v3_ring_wrap", "seed": 45, "block_dim": 1,
     "purpose": "769 tokens on one core: the four-slot snake wraps repeatedly at lag 3",
     "parameters": {"variant": "v3", "BH": 1, "S1": 769, "S2": 769, "D": 128}},

    {"id": "v4_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full tile: a single four-key group, so v4's grouping is degenerate here",
     "parameters": {"variant": "v4", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v4_tail", "seed": 43, "block_dim": 2,
     "purpose": "Double tail inside a four-key group: the last group is partial",
     "parameters": {"variant": "v4", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v4_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two heads over three cores with grouped keys and one idle core",
     "parameters": {"variant": "v4", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v4_ring_wrap", "seed": 45, "block_dim": 2,
     "purpose": "769 tokens: the four-slot ring wraps at lookahead two, its tightest margin",
     "parameters": {"variant": "v4", "BH": 1, "S1": 769, "S2": 769, "D": 128}},

    {"id": "v5_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full tile on one core -- the launch width the source recorded as unsupported",
     "parameters": {"variant": "v5", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v5_tail", "seed": 43, "block_dim": 2,
     "purpose": "Double tail with the disjoint V backfill reading 128 - valid_n head rows",
     "parameters": {"variant": "v5", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v5_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two heads over three cores: S2=73 is the smallest backfill this variant allows",
     "parameters": {"variant": "v5", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v5_ring_wrap", "seed": 45, "block_dim": 2,
     "purpose": "769 tokens: the five-slot workspace ring and the eight-tile V preload ring both wrap",
     "parameters": {"variant": "v5", "BH": 1, "S1": 769, "S2": 769, "D": 128}},
    {"id": "v5_source_smoke", "seed": 42, "block_dim": 20,
     "purpose": "The original 2177-token sweep: v5 fills and drains its pipeline once per M tile",
     "parameters": {"variant": "v5", "BH": 1, "S1": 2177, "S2": 2177, "D": 128}},

    {"id": "v6_aligned", "seed": 42, "block_dim": 1,
     "purpose": "One full tile: the continuous stream has a single M tile to flatten",
     "parameters": {"variant": "v6", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "v6_tail", "seed": 43, "block_dim": 2,
     "purpose": "Double tail inside one continuous group stream rather than a per-tile loop",
     "parameters": {"variant": "v6", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "v6_heads_idle", "seed": 44, "block_dim": 3,
     "purpose": "Two heads over three cores: each core's stream is its own, one is empty",
     "parameters": {"variant": "v6", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "v6_ring_wrap", "seed": 45, "block_dim": 2,
     "purpose": "769 tokens: the workspace, V preload and five-slot rowmax/rowsum M-tile rings all wrap",
     "parameters": {"variant": "v6", "BH": 1, "S1": 769, "S2": 769, "D": 128}},
    {"id": "v6_source_smoke", "seed": 42, "block_dim": 20,
     "purpose": "The original 2177-token sweep: v6 fills once at the start and drains once at the end",
     "parameters": {"variant": "v6", "BH": 1, "S1": 2177, "S2": 2177, "D": 128}},

    {"id": "v1_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The original 257-token boundary on 20 cores: one query row past two full tiles",
     "parameters": {"variant": "v1", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "v2_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The 257-token boundary under the next-query prefetch",
     "parameters": {"variant": "v2", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "v3_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The 257-token boundary under the lag-3 snake",
     "parameters": {"variant": "v3", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "v4_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The 257-token boundary under four-key groups at lookahead two",
     "parameters": {"variant": "v4", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "v5_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The 257-token boundary with the V preload ring, at the source's own launch width",
     "parameters": {"variant": "v5", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "v6_source_257", "seed": 42, "block_dim": 20,
     "purpose": "The 257-token boundary under the continuous cross-M-tile stream",
     "parameters": {"variant": "v6", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
]

OUTPUTS = ("out", "rowmax", "rowsum")


def execute(case, inputs, launcher, device):
    """Launch one variant. All three destinations are handed in poisoned with NaN and seeded
    into the launch, so a row no core claimed reads back as NaN rather than as a zero.

    The row statistics are published per head at a stride padded up to a multiple of eight
    FP32 values, so two heads can never share a 32-byte GM line. Slicing them back to the
    compact [BH * S1] public layout is host work and stays here."""
    p = case["parameters"]
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    out = torch.full(q.shape, float("nan"))
    stride = (p["S1"] + 7) // 8 * 8
    rowmax = torch.full((p["BH"] * stride,), float("nan"))
    rowsum = torch.full_like(rowmax, float("nan"))
    op = OpExec(build_kernel(p["variant"], device), launcher=launcher, backend="cce",
                device=device, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}/{p['variant']}", seed_outputs=True)
    values = op(q, k, v, out, rowmax, rowsum, p["S1"], p["S2"], 128, p["BH"], 128 ** -0.5)
    compact = {name: value.reshape(p["BH"], stride)[:, :p["S1"]].contiguous().reshape(-1)
               for name, value in zip(("rowmax", "rowsum"), values[1:], strict=True)}
    return {"out": values[0], **compact}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail, outside = torch.equal(got, want), "  bitwise", got != want
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        atol, rtol = bounds.get("atol", 0.0), bounds.get("rtol", 0.0)
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for each
        # element. Print the worst element as a fraction of its own allowance, so the number has a
        # bound of 1 and a passing line cannot read as a failing one.
        room = (atol + rtol * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        outside = ~torch.isclose(got.float(), want.float(), **bounds)
        ok = not outside.any().item()
        detail = f"  allclose={margin:.2f}x (atol={atol:g} rtol={rtol:g})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok and outside.any():
        # The destinations arrive NaN-poisoned, so an element that is still NaN was never written.
        # Saying which, and where, is the difference between "something is nan" and "row 127 is".
        idx, poison = outside.nonzero(), int((outside & torch.isnan(got.float())).sum())
        note = f", {poison} still NaN-poisoned (never written)" if poison else ""
        print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
              f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--device", default="a2", choices=("a2", "a3"))
    parser.add_argument("--case", default="all", help="a case id, a variant name, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"], case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case or variant named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (BH={p['BH']} S1={p['S1']} S2={p['S2']}, device={args.device}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        for name in OUTPUTS:
            rule = TOLERANCE.get(name, TOLERANCE["default"])
            if not compare(name, actual[name], expected[name], rule):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
