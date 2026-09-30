# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A contiguous bias row applied once, across five A2/A3 matmul paths and ten geometries.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --case f32_splitk_m16_n64_k128  # one of them
    python main.py --device a3                  # the same source against the A3 facade
    python main.py --launcher pipesim           # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn             # the cce backend, on this machine's card

    o = x @ y.T + bias

Five kernels: FP32 with no split, with `splitk=16`, and with `splitn=32`; INT8 into INT32 with no
split and with `splitn=32`. The bias is an L1 tensor declared `layout=Layout.ND`, which selects the
A2 family's contiguous GM-to-L1 transfer rather than a fractalized one.

Two details decide whether a bias lands once and in the right column. `is_init=True` is what makes
split-K add it on the initialization tile only -- the `k=128` case has eight passes, so a bias added
on each would be eight times too large. And split-N takes 32-element slices of the row: the INT32
bias already has the column addressing for that, while the FP32 bias reaches it through an
accepted same-width reinterpret to INT32 inside `matmul`, with no value conversion anywhere.

The bias is `3, 6, 9, ...` so every column is distinct and monotone, and the operands are dyadic
eighths or INT8 values small enough that no product overflows. Every output bit is compared.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernels
from reference import make_inputs, reference

DEVICES = ("a2", "a3")

POISON = -777
STATIC_N = 64     # no-split and split-K declare this statically for the full bias row
SPLIT_N = 32      # the N tile width, and therefore the bias slice width

OUTPUTS = ("o",)

CASES = [
    {"id": "f32_none_m32_n64_k32", "seed": 103600, "block_dim": 1,
     "purpose": "FP32, no split, at the static N=64 the no-split and split-K paths require: the "
                "baseline where the whole bias row is added once by construction",
     "parameters": {"mode": "f32_none", "M": 32, "N": 64, "K": 32}},
    {"id": "f32_splitk_m32_n64_k48", "seed": 103601, "block_dim": 1,
     "purpose": "split-K over K=48, three 16-element passes: the bias belongs on the "
                "initialization tile alone, so a bias added per pass is three times too large",
     "parameters": {"mode": "f32_splitk", "M": 32, "N": 64, "K": 48}},
    {"id": "f32_splitn_m32_n64_k32", "seed": 103602, "block_dim": 1,
     "purpose": "split-N in two 32-column halves, each taking its own slice of the bias row -- the "
                "smallest case where the wrong slice is a wrong answer rather than the same one",
     "parameters": {"mode": "f32_splitn", "M": 32, "N": 64, "K": 32}},
    {"id": "f32_splitn_m16_n96_k48", "seed": 103603, "block_dim": 1,
     "purpose": "N=96: three N tiles, so the third slice's columns differ from both earlier ones "
                "and a tile that reused the first slice is visible in two places",
     "parameters": {"mode": "f32_splitn", "M": 16, "N": 96, "K": 48}},
    {"id": "i8_none_m32_n64_k96", "seed": 103604, "block_dim": 1,
     "purpose": "INT8 into INT32 with no split, K=96 -- a multiple of 32, which is the INT8 "
                "alignment the compact L1 allocation requires",
     "parameters": {"mode": "i8_none", "M": 32, "N": 64, "K": 96}},
    {"id": "i8_splitn_m32_n64_k96", "seed": 103605, "block_dim": 1,
     "purpose": "INT8 split-N: the INT32 bias row already has the contiguous column addressing the "
                "slices need, where the FP32 one has to be reinterpreted to reach it",
     "parameters": {"mode": "i8_splitn", "M": 32, "N": 64, "K": 96}},
    {"id": "i8_splitn_m16_n96_k32", "seed": 103606, "block_dim": 1,
     "purpose": "INT8 at the minimum M and K with three N tiles: the smallest INT8 geometry that "
                "still slices the bias three ways",
     "parameters": {"mode": "i8_splitn", "M": 16, "N": 96, "K": 32}},
    {"id": "f32_splitn_m64_n32_k16", "seed": 103607, "block_dim": 1,
     "purpose": "The maximum M against the minimum N and K: split-N with exactly one 32-column "
                "tile, where the slicing is degenerate and must still be right",
     "parameters": {"mode": "f32_splitn", "M": 64, "N": 32, "K": 16}},
    {"id": "i8_splitn_m64_n32_k128", "seed": 103608, "block_dim": 1,
     "purpose": "The maximum M and K in INT8 with a single N tile: the largest compact L1 "
                "allocation this example declares",
     "parameters": {"mode": "i8_splitn", "M": 64, "N": 32, "K": 128}},
    {"id": "f32_splitk_m16_n64_k128", "seed": 103609, "block_dim": 1,
     "purpose": "The minimum M with the maximum K: eight split-K passes, which is the most chances "
                "a kernel has to add the bias more than once",
     "parameters": {"mode": "f32_splitk", "M": 16, "N": 64, "K": 128}},
]


def check_domain(inputs, expected):
    """The geometry the compact L1 allocation requires, and the two properties of the bias row the
    slicing checks depend on: distinct columns, and a monotone order so a swapped slice is visible."""
    mode, x, y, bias = (inputs[key] for key in ("mode", "x", "y", "bias"))
    integer = mode.startswith("i8")
    m, k = x.shape
    n = y.shape[0]
    if not 16 <= m <= 64 or m % 16 or n not in (32, 64, 96):
        raise ValueError("M is a multiple of 16 in [16, 64]; N is 32, 64 or 96")
    if k % (32 if integer else 16) or not 16 <= k <= 128:
        raise ValueError(f"K must be a multiple of {32 if integer else 16} in [16, 128]")
    if mode.endswith(("none", "splitk")) and n != STATIC_N:
        raise ValueError(f"the no-split and split-K paths declare a static N={STATIC_N}")
    row = bias.flatten()
    if len(set(row.tolist())) != n or not bool((row[1:] > row[:-1]).all()):
        raise ValueError("the bias columns must be distinct and increasing, or a wrong N slice "
                         "could not be told from the right one")
    if (expected["o"] == POISON).any():
        raise ValueError("the reference contains the poison value")


def execute(case, inputs, launcher, backend, device):
    """One launch. The destination arrives filled with -777 in the output's own dtype and seeded in,
    so an element no fixpipe store reached reads back as the fill."""
    mode, x, y, bias = (inputs[key] for key in ("mode", "x", "y", "bias"))
    m, k, n = x.shape[0], x.shape[1], y.shape[0]
    entry = make_kernels(device)[mode]
    op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    scalars = (m, n, k) if mode.endswith("splitn") else (m, k)
    return {"o": op(x, y, bias, torch.full((m, n), POISON, dtype=bias.dtype), *scalars)}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: bounded dyadic FP32 arithmetic and non-overflowing integer products are exact."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]}x{got.shape[1]} "
          f"{str(got.dtype).removeprefix('torch.')} elements")
    if not ok:
        got, want = got.cpu(), want.cpu()
        outside = got != want
        columns = sorted({int(c) for _, c in outside.nonzero().tolist()})
        # A bias fault is constant down a column; a product fault is not.
        delta = (got - want)
        per_column = {c: sorted({float(v) for v in delta[:, c][outside[:, c]]}) for c in columns[:3]}
        unwritten = int((got[outside] == POISON).sum())
        print(f"      {int(outside.sum())} elements differ in columns {columns[:8]}; per-column "
              f"deltas {per_column}"
              + ("  (one delta per column points at the bias, not the product)"
                 if all(len(v) == 1 for v in per_column.values()) else "")
              + (f"; {unwritten} still hold the {POISON} fill (never written)"
                 if unwritten else ""))
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
            print(f"{case['id']:26s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['mode']}, {p['M']}x{p['N']}x{p['K']}, device={args.device}, "
              f"launcher={args.launcher})")
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
