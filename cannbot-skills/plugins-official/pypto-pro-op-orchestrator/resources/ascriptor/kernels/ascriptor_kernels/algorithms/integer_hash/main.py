# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the MurmurHash3 fmix32 finalizer through OpExec and check every bit of every payload.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case bit_boundaries

INT32 tensors here are carriers, not numbers: they hold raw UINT32 payloads. The kernel
reinterprets them to uint32 for the logical right shift and to uint16 for the XOR halves,
and materializes each odd constant as a full INT32 tensor before multiplying, so the product
wraps modulo 2^32 with every low bit intact. Read the reinterprets in kernel.py before
changing anything: an INT32 shift would be arithmetic and would smear the sign bit.

The partition is decided at run time by `GetVecNum()`/`GetVecIdx()`, so the same body runs on
one owner or two and `block_dim` is the only thing the cases change.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import make_inputs, reference

# Bitwise. The finalizer is exact integer arithmetic: two modular multiplies and three
# logical XOR-shifts, with no rounding anywhere to spend a tolerance on. The reference
# recomputes it in Python's arbitrary-precision integers, so a signed shift, a constant
# truncated to a float mantissa, a dropped stage or an unwritten tail all appear as a changed
# bit rather than as a small error.
TOLERANCE = None

CASES = [
    {"id": "source_0", "seed": 0, "block_dim": 1,
     "purpose": "The source's own geometry: 1024 values, one tile, one owner",
     "parameters": {"shape": [1, 1024], "mode": "random", "tile_len": 1024}},
    {"id": "source_1", "seed": 1, "block_dim": 1,
     "purpose": "One tile at the full 8192-value UB capacity",
     "parameters": {"shape": [1, 8192], "mode": "random", "tile_len": 8192}},
    {"id": "source_2", "seed": 2, "block_dim": 1,
     "purpose": "1023 values in a 1023-long tile: neither is a power of two",
     "parameters": {"shape": [1, 1023], "mode": "random", "tile_len": 1023}},
    {"id": "source_3", "seed": 3, "block_dim": 1,
     "purpose": "517 values: an odd length whose byte extent is not 32-byte aligned",
     "parameters": {"shape": [1, 517], "mode": "random", "tile_len": 517}},
    {"id": "source_4", "seed": 4, "block_dim": 1,
     "purpose": "A single value: the smallest tile the loop can be given",
     "parameters": {"shape": [1, 1], "mode": "random", "tile_len": 1}},
    {"id": "source_5", "seed": 5, "block_dim": 1,
     "purpose": "24653 values: three full 8192 tiles and a 77-value tail on one owner",
     "parameters": {"shape": [1, 24653], "mode": "random", "tile_len": 8192}},
    {"id": "source_6", "seed": 6, "block_dim": 2,
     "purpose": "A 2-D [128, 128] input flattened to 16384: exactly one tile per owner",
     "parameters": {"shape": [128, 128], "mode": "random", "tile_len": 8192}},
    {"id": "source_7", "seed": 7, "block_dim": 1,
     "purpose": "8256 values: two tiles on one owner, so the DBuff slot is reused",
     "parameters": {"shape": [64, 129], "mode": "random", "tile_len": 8192}},
    {"id": "source_counter", "seed": 0, "block_dim": 1,
     "purpose": "seed 0x9E3779B9 xor counter: 4096 distinct inputs, a permutation not a hash",
     "parameters": {"shape": [4096], "mode": "counter", "counter_seed": 2654435769,
                    "tile_len": 4096}},
    {"id": "bit_boundaries", "seed": 13600, "block_dim": 1,
     "purpose": "0x80000000, 0xFFFFFFFF, 0xAAAAAAAA and friends: where a signed shift diverges",
     "parameters": {"shape": [1, 73], "mode": "boundaries", "tile_len": 32}},
    {"id": "two_cores_tail", "seed": 13601, "block_dim": 2,
     "purpose": "1285 values in six 256-tiles, three per owner, the last one only 5 long",
     "parameters": {"shape": [5, 257], "mode": "random", "tile_len": 256}},
]


def execute(case, inputs, launcher, backend):
    """Launch the finalizer over the flattened input. INT32 cannot carry a NaN, so the
    destination is filled with the 0x13579BDF sentinel instead and seeded into the launch: a
    value the kernel never writes reads back as that sentinel rather than as a plausible
    hash."""
    y = torch.full((1, inputs["n"]), 0x13579BDF, dtype=torch.int32)
    op = OpExec(kernel_for("a2"), launcher=launcher, backend=backend, device="a2",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(inputs["x"].reshape(1, -1), y, inputs["n"], inputs["tile_len"])
    return {"output": actual.reshape(inputs["shape"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), "  bitwise"
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got.float(), want.float(), **bounds), f"  allclose={margin:.2f}x ({bounds})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **bounds)
        if outside.any():
            idx = outside.nonzero()
            poison = int((outside & torch.isnan(got.float())).sum())
            note = f", {poison} still NaN-poisoned (never written)" if poison else ""
            print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
                  f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
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
        p = case["parameters"]
        print(f"{case['id']}  (shape={p['shape']}, tile_len={p['tile_len']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
