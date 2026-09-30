# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""An FP16 row normalized in FP32: unpack, widen, reduce, broadcast, narrow, compact.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case zero_row       # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card
    python main.py --backend pto_isa --launcher board     # the PTO ISA backend on a card
    python main.py --backend pypto_pro --launcher pypto   # the PyPTO Pro toolchain's board

    out[r, c] = f16(f32(x[r, c]) * rsqrt(mean(f32(x[r, :])^2) + 2^-10) * f32(w[c]))

This is the round trip every normalization opens with, and the only example that walks the
reduction's return path: `cadd` leaves the sum in lane 0, `reg_to_ub_single` parks that lane in one
UB cell and `ub_to_reg_single` brings it back across every lane. The FP32 work happens on the even
lanes of an FP16 register -- `ub_to_reg_unpack` in, `cast` with `RegLayout.ZERO` to widen,
`reg_to_ub_downsample` out -- and the A5 register vocabulary has no `rsqrt`, so the scale is `sqrt`
then `div`.

The comparison is bitwise against an FP64 reference: the kernel's FP32 arithmetic and its single
round-to-even narrow reproduce it exactly on every declared case, so a drift of one ulp is a result
rather than noise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import EPS, WIDTH, row_norm
from reference import make_inputs, reference

DEVICE = "a5"

ROWS = 4

OUTPUTS = ("o",)

CASES = [
    {"id": "ordinary", "seed": 4401, "block_dim": 1,
     "purpose": "Four ordinary rows with no boundary in them: the whole round trip, end to end, "
                "against an FP64 reference",
     "parameters": {}},
    {"id": "zero_row", "seed": 4402, "block_dim": 1,
     "purpose": "The last row is all zero, and it is the only case in this matrix where eps is "
                "observable at all: with the bias the row normalizes to a finite zero, without it "
                "the row divides by zero and arrives NaN. In every other input domain eps is far "
                "below one FP16 ulp, so a matrix without this row cannot tell an implementation "
                "that applies eps from one that drops it",
     "parameters": {"zero_row": True}},
    {"id": "tiny_row", "seed": 4403, "block_dim": 1,
     "purpose": "Row 0 is 2^-12 in every lane and the last row is still zero. The mean of squares "
                "is 2^-24, orders below eps, so the scale is eps-dominated and the result is "
                "decided by the narrowing rather than by the reduction",
     "parameters": {"tiny_row": True, "zero_row": True}},
    {"id": "second_seed", "seed": 4404, "block_dim": 1,
     "purpose": "A second ordinary draw, so a bitwise pass is not a property of one input",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """What the comparison rests on: the shapes the kernel declares, and a finite reference. The
    zero row is finite only because of eps, so a non-finite reference here would mean the reference
    itself had dropped the bias -- and the two would then agree for the wrong reason."""
    x, w = inputs["x"], inputs["w"]
    if x.shape != (ROWS, WIDTH) or x.dtype != torch.float16:
        raise ValueError(f"the input must be float16[{ROWS}, {WIDTH}]")
    if w.shape != (1, WIDTH) or w.dtype != torch.float16:
        raise ValueError(f"the weights must be float16[1, {WIDTH}]")
    if not bool(torch.isfinite(expected["o"]).all()):
        raise ValueError(f"the reference is not finite; with eps={EPS} it should be, even for an "
                         f"all-zero row")


def execute(case, inputs, launcher, backend):
    """One launch for all four rows. The destination arrives NaN-poisoned and is seeded into the
    launch, so a row the vector function never reached reads back as NaN rather than as a plausible
    normalized row -- which matters here because a correct all-zero row is also all zeros."""
    op = OpExec(row_norm, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    produced = op(inputs["x"], inputs["w"], torch.full((ROWS, WIDTH), float("nan"),
                                                       dtype=torch.float16), ROWS)
    return {"o": produced[0] if isinstance(produced, (tuple, list)) else produced}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise. The kernel computes in FP32 and narrows once; the reference computes in FP64 and
    narrows once. On these cases they agree exactly, and a one-ulp disagreement is a finding."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {ROWS} rows x {WIDTH} FP16 lanes")
    if not ok:
        got, want = got.cpu(), want.cpu()
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            lanes = (got[row] != want[row]).nonzero().flatten().tolist()
            # One ulp apart is a rounding disagreement; NaN is a row nothing normalized.
            ulps = [int((got[row, c].view(torch.int16) - want[row, c].view(torch.int16)).abs())
                    for c in lanes[:4]]
            poisoned = bool(torch.isnan(got[row]).all())
            print(f"      row {row}: {len(lanes)} lanes differ at {lanes[:6]}, "
                  f"{ulps} ulps apart"
                  + ("  (the whole row is still NaN-poisoned: never written)" if poisoned else ""))
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
            print(f"{case['id']:12s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (backend={args.backend}, launcher={args.launcher}, "
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
