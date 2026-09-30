# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-register sum, maximum and minimum at three widths, with the zero lanes compared too.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case second_seed    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`cadd`, `cmax` and `cmin` reduce a whole register and leave the result in lane 0. The kernel runs
all three over three register widths -- FP32 x64, INT32 x64, INT64 x32 -- and writes each result a
full register apart, addressed by element offset (`out[0]`, `out[cols]`, `out[2 * cols]`).

Five outputs are compared, and `upper` is the interesting one: it is every lane except lane 0 of
the float destination, which the model leaves zero. Comparing it is how the lane behaviour is
stated rather than assumed -- but see metadata.json: a model result is not a statement about the
device.

Only the float sum carries a tolerance, for FP32 reduction order. The extrema, both integer
widths and every zero lane are bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import reduce_family
from reference import make_inputs, reference

DEVICE = "a5"

# Per output. An output not named here is compared byte for byte. The float sum is the only
# result whose value depends on the order the lanes were added in, and these are the bounds the
# unit has always used (Torch's isclose defaults).
TOLERANCE = {"sum": {"rtol": 1e-05, "atol": 1e-08}}

OUTPUTS = ("sum", "extrema", "upper", "int32", "int64")

# The three destinations the kernel writes, with the poison each arrives holding.
DESTINATIONS = (((3, 64), torch.float32, float("nan")),
                ((3, 64), torch.int32, 99),
                ((3, 32), torch.int64, 99))

CASES = [
    {"id": "signed_widths", "seed": 8761, "block_dim": 1,
     "purpose": "Three register widths in one launch, with the INT64 input carrying -2^62 and "
                "2^62 + 12345 -- values that differ only in their upper word, so the 64-bit "
                "extrema cannot come out right from the low halves alone",
     "parameters": {}},
    {"id": "second_seed", "seed": 8762, "block_dim": 1,
     "purpose": "The same nine reductions over a different input: lane 0 holding the result and "
                "every other lane holding zero are properties of the instruction rather than of "
                "these particular values",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """Two things the comparison rests on. The integer sums must not overflow their own width, or
    the reference and the device would be comparing two different defined behaviours; and the
    reference's `upper` must actually be the zero lanes, which is the claim being checked."""
    for name, width in (("xi", torch.int32), ("xl", torch.int64)):
        total = inputs[name].to(torch.float64).sum().item()
        limit = torch.iinfo(width).max
        if not -limit <= total <= limit:
            raise ValueError(f"{name} sums to {total}, outside {width}")
    if expected["upper"].count_nonzero():
        raise ValueError("the reference's non-lane-0 columns must be zero")


def execute(case, inputs, launcher, backend):
    """One launch for all three destinations, each pre-poisoned and seeded in -- so a lane the
    reduction never wrote reads back as NaN or 99 rather than as a plausible zero, which matters
    here because zero is the expected value for 189 of the 192 float lanes."""
    op = OpExec(reduce_family, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full(shape, poison, dtype=dtype) for shape, dtype, poison in DESTINATIONS]
    of, oi, ol = op(inputs["xf"], inputs["xi"], inputs["xl"], *poisoned)
    # The float destination carries three separate results: the sum in lane 0 of row 0, the two
    # extrema in lane 0 of rows 1 and 2, and the lanes that should have stayed zero.
    return {"sum": of[0, 0:1], "extrema": of[1:, 0], "upper": of[:, 1:], "int32": oi, "int64": ol}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise unless TOLERANCE names this output."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:8s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    rule = TOLERANCE.get(name)
    if rule is None:
        ok, detail = torch.equal(raw(got), raw(want)), "bitwise"
    else:
        # `atol` alone is not a bound on the worst element: allclose allows atol + rtol*|want|
        # each. Print the worst one as a fraction of its own allowance, so a passing line cannot
        # read as a failing one.
        room = (rule["atol"] + rule["rtol"] * want.double().abs()).clamp(min=1e-30)
        margin = ((got.double() - want.double()).abs() / room).max().item()
        ok = bool(torch.allclose(got, want, **rule))
        detail = f"allclose={margin:.2f}x (atol={rule['atol']:g} rtol={rule['rtol']:g})"
    print(f"    {name:8s} {'ok  ' if ok else 'FAIL'}  {detail} over {got.numel()} lanes")
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        outside = (raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
                   if rule is None else
                   ~torch.isclose(got.reshape(-1), want.reshape(-1), **rule))
        index = outside.nonzero().flatten().tolist()
        poison = {float("nan"), 99}
        stale = [i for i in index if flat_got[i].item() in poison or flat_got[i] != flat_got[i]]
        note = f"; lanes {stale[:4]} still hold the poison (never written)" if stale else ""
        print(f"      {len(index)}/{got.numel()} lanes differ: got "
              f"{[flat_got[i].item() for i in index[:4]]} want "
              f"{[flat_want[i].item() for i in index[:4]]}{note}")
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
            print(f"{case['id']:15s} {case['purpose']}")
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
