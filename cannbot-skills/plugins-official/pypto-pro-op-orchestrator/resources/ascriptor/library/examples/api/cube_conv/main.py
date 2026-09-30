# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One 16x16 convolution output tile at three geometries, physical M tail included.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case stride_tail    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`conv2d(dst, feature_map, weights, Conv2D(...), h=, w=, c=, cout=, m0=, tile_k=)` over a 4x4 image
with three input and five output channels. The host owns the packing: an NC1HWC0 feature map with
C1=1 and C0=16, and weights as [Cout_p, K] row-major with K = kh*kw*c0 = 144. Channels past the real
3 and 5 are zero.

The physical tile is 16 rows whatever the geometry, and only the `same_padding` and `dilation` cases
fill all of them with logical output. At stride 2 the logical output is 4 rows -- and the other
twelve are not padding: `load3d` keeps producing raster windows past Ho*Wo (D-223) and they can hold
real values. The reference reaches them without imitating the loader at all: it extends the *bottom*
zero pad until ordinary `conv2d` produces those same windows. Every physical row is compared.

Small integer FP16 inputs make the FP32 convolution exact, so the comparison is bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_conv
from reference import make_inputs, reference

DEVICE = "a5"

TILE = (16, 16)
COUT = 5   # the real output channels; columns 5..15 of the tile are carrier padding

OUTPUTS = ("o",)

CASES = [
    {"id": "same_padding", "seed": 8931, "block_dim": 1,
     "purpose": "stride 1, pad 1: a 4x4 image gives 4x4 output, so all 16 physical rows are "
                "logical output rows and there is no tail to reason about",
     "parameters": {"stride": 1, "dilation": 1, "pad": 1, "output_rows": 16}},
    {"id": "stride_tail", "seed": 8932, "block_dim": 1,
     "purpose": "stride 2: only 4 of the 16 physical rows are logical output. The remaining twelve "
                "are raster windows load3d continues past Ho*Wo and they carry real values, which "
                "the reference reproduces by extending the bottom zero pad rather than by "
                "decoding what the loader did",
     "parameters": {"stride": 2, "dilation": 1, "pad": 1, "output_rows": 4}},
    {"id": "dilation", "seed": 8933, "block_dim": 1,
     "purpose": "dilation 2 with pad 2: the 3x3 kernel spans five input positions, so every window "
                "touches the padding and the output is 4x4 again -- the geometry where a dilation "
                "applied to the pad instead of to the kernel would still produce the right shape",
     "parameters": {"stride": 1, "dilation": 2, "pad": 2, "output_rows": 16}},
]


def check_domain(inputs, expected):
    """What makes the comparison exact and the packing checkable: small integer values, zero carrier
    padding in both operands, and a reference whose columns past the real Cout are zero."""
    image, weights = inputs["image"], inputs["weights"]
    if image.shape != (1, 3, 4, 4) or weights.shape != (COUT, 3, 3, 3):
        raise ValueError("this example declares a 4x4x3 image and 5 3x3x3 filters")
    for name, tensor in (("image", image), ("weights", weights)):
        if not torch.equal(tensor.float(), tensor.float().round()):
            raise ValueError(f"{name} must be integral, or the FP32 convolution is not exact")
    if inputs["packed_image"][:, 3:].count_nonzero() or inputs["packed_weights"].reshape(16, 3, 3, 16)[COUT:].count_nonzero():
        raise ValueError("the carrier padding must be zero in both packed operands")
    if expected["o"][:, COUT:].count_nonzero():
        raise ValueError(f"the reference's columns past Cout={COUT} must be zero")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so a physical
    row the fixpipe never wrote is distinguishable from a tail row that legitimately came out zero --
    which is the distinction this unit's tail case is about."""
    p = inputs["geometry"]
    op = OpExec(make_conv(p["stride"], p["dilation"], p["pad"]), launcher=launcher, backend=backend,
                device=DEVICE, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}/{case['id']}", seed_outputs=True)
    return {"o": op(inputs["packed_image"], inputs["packed_weights"],
                    torch.full(TILE, float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want, rows=TILE[0]):
    """Bitwise over all 16 physical rows, with the logical and tail rows reported separately."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {TILE[0]} physical rows "
          f"({rows} logical + {TILE[0] - rows} raster tail)")
    if not ok:
        got, want = got.cpu(), want.cpu()
        differing = (got != want).any(dim=1).nonzero().flatten().tolist()
        logical = [r for r in differing if r < rows]
        tail = [r for r in differing if r >= rows]
        poisoned = [r for r in differing if bool(torch.isnan(got[r]).any())]
        print(f"      rows {differing[:8]} differ: {len(logical)} logical, {len(tail)} in the "
              f"raster tail" + (f"; rows {poisoned[:4]} still hold NaN (never written)"
                                if poisoned else ""))
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
        p = case["parameters"]
        print(f"{case['id']}  (stride={p['stride']}, dilation={p['dilation']}, pad={p['pad']}, "
              f"{p['output_rows']} logical rows, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], p["output_rows"]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
