# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the FP32 HiFloat8 quantize-dequantize through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case edge   # cce backend, on this machine's card

The kernel builds the HiFloat8 step out of the FP32 exponent bits and rounds half away
from zero, with no dtype conversion anywhere: the tile that arrives is the tile that is
quantized. Overflow to signed infinity, the 2^-23 underflow floor, and incoming
infinities and NaN are output classes the kernel must reproduce, not inputs it may
reject, so the comparison is exact rather than a tolerance.

Read this before `hif8_bf16`, which wraps this identical body in two casts.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# The contract declares this one vector body on both A2 and A3, and `build_kernel` binds it
# to one facade; change this to "a3" to run the identical source against the A3 target.
DEVICE = "a2"

# Bitwise, with NaN equal to NaN. Every finite HiFloat8 reconstruction is a power-of-two step
# times an integer, which FP32 holds exactly, so there is no rounding left for a tolerance to
# absorb. NaN payload identity is deliberately outside this: the kernel must return a NaN
# where a NaN went in, not the same NaN bits.
TOLERANCE = None

CASES = [
    {"id": "edge", "seed": 0, "block_dim": 2,
     "purpose": "The complete source edge table: signed zeros, both sides of the 2^-23 floor "
                "and of the 40960 overflow threshold, FP32 max, infinities and NaN",
     "parameters": {"total": 87, "scale": 1.0, "pattern": "edge"}},
    {"id": "source_aligned", "seed": 5, "block_dim": 2,
     "purpose": "Exactly one 512-element tile: the aligned control with no tail",
     "parameters": {"total": 512, "scale": 1.0, "pattern": "source"}},
    {"id": "source_reuse", "seed": 6, "block_dim": 2,
     "purpose": "Ten tiles over two cores: the double buffer wraps and the last tile is a "
                "37-element tail",
     "parameters": {"total": 4645, "scale": 32.0, "pattern": "source"}},
    {"id": "idle", "seed": 7, "block_dim": 3,
     "purpose": "Less than one tile on three cores: two cores are handed an empty range and "
                "must publish nothing",
     "parameters": {"total": 31, "scale": 0.5, "pattern": "source"}},
    {"id": "original_257", "seed": 0, "block_dim": 20,
     "purpose": "The source's 257-element distribution on twenty cores: one tile, nineteen idle",
     "parameters": {"total": 257, "scale": 1.0, "pattern": "source"}},
    {"id": "original_small_scale", "seed": 1, "block_dim": 20,
     "purpose": "The source's 1e-7 distribution straddles the 2^-23 underflow floor, so both "
                "flush-to-zero and the smallest kept step appear in one launch",
     "parameters": {"total": 4096, "scale": 1e-07, "pattern": "source"}},
    {"id": "original_reuse", "seed": 2, "block_dim": 20,
     "purpose": "The source's 1e4 distribution: 42 tiles over twenty cores, a 37-element tail, "
                "and finite inputs that cross the overflow threshold into signed infinity",
     "parameters": {"total": 21029, "scale": 10000.0, "pattern": "source"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. The destination is handed in poisoned with NaN and seeded into the
    launch, so an element no core writes reads back as NaN instead of as the zero this kernel
    also produces legitimately for an underflowing input."""
    op = OpExec(build_kernel(DEVICE), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    y = torch.full_like(inputs["x"], float("nan"))
    return {"y": op(inputs["x"], y, inputs["total"])}


def compare(name, got, want, tolerance):
    """NaN counts as a value here, so the exact comparison accepts NaN against NaN.
    max_abs_diff is measured over the lanes where both sides are finite: inf minus inf is
    NaN, and a single such lane would otherwise swallow every real difference."""
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok = bool((torch.eq(got, want) | (got.isnan() & want.isnan())).all())
        detail = ""
    else:
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (tolerance.get("atol", 0.0) + tolerance.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok = torch.allclose(got.float(), want.float(), **tolerance)
        detail = f"  allclose={margin:.2f}x ({tolerance})"
    delta = (got.float() - want.float()).abs()
    finite = got.float().isfinite() & want.float().isfinite()
    worst = delta[finite].max().item() if bool(finite.any()) else 0.0
    print(f"    {name:4s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e} (finite lanes)"
          f"{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (~(torch.eq(got, want) | (got.isnan() & want.isnan())) if tolerance is None
                   else ~torch.isclose(got.float(), want.float(), **tolerance))
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
            print(f"{case['id']:22s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (total={p['total']}, scale={p['scale']}, pattern={p['pattern']}, "
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
