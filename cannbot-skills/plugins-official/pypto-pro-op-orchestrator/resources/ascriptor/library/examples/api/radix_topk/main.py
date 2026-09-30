# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One `radix_topk` call, and the two different kinds of check its two outputs need.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --case ties          # one of them
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher board     # the cce backend, on this machine's card

The source tile is always 4096 lanes with `count` live and the rest at -infinity; the two
destinations are 512 lanes with `k` filled. One instruction does the selection.

The two outputs cannot be checked the same way, and that is the thing to take from this example.
The **values** have a reference: the k largest live values are a well-defined multiset, so they are
compared after sorting -- the device's order is not part of the contract. The **indices** have no
reference at all, because which index is returned for a tie is not specified. They are checked by
their properties instead: in range, distinct, and each one pointing at the value that was returned
beside it, compared as bits rather than as numbers.

`ties` exists because that distinction is only visible when several lanes hold the same value.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import select_largest
from reference import LANES, SLOTS, make_inputs, reference

DEVICE = "a5"

VALUE_POISON = float("nan")     # an unfilled value slot
INDEX_POISON = -1               # an unfilled index slot; no valid index is negative

OUTPUTS = ("values",)           # the indices are checked by property, not against a reference

CASES = [
    {"id": "source", "seed": 31010, "block_dim": 1,
     "parameters": {"count": 65, "k": 7, "dataset": "random"},
     "purpose": "The original selection: 65 live lanes of 4096, seven of them chosen"},
    {"id": "k_one", "seed": 31011, "block_dim": 1,
     "parameters": {"count": 65, "k": 1, "dataset": "random"},
     "purpose": "k = 1: the single largest. The smallest selection there is, and the one where an "
                "off-by-one in the rank is unmistakable"},
    {"id": "k_equals_count", "seed": 31012, "block_dim": 1,
     "parameters": {"count": 32, "k": 32, "dataset": "random"},
     "purpose": "k = count: everything live is selected, so the answer is the whole prefix and "
                "every index must appear exactly once"},
    {"id": "ties", "seed": 31013, "block_dim": 1,
     "parameters": {"count": 64, "k": 8, "dataset": "ties"},
     "purpose": "Values drawn from a pool of seven, so many lanes are equal and several compete for "
                "the same rank. This is the case where 'which index wins a tie is not specified' "
                "stops being an abstraction"},
    {"id": "full_source", "seed": 31014, "block_dim": 1,
     "parameters": {"count": LANES, "k": SLOTS, "dataset": "random"},
     "purpose": "Every lane live and every slot filled: the largest declared geometry, with no "
                "padding anywhere to make the selection easier"},
]


def check_domain(inputs, expected):
    """That the case can distinguish a correct selection from a plausible one."""
    count, k = inputs["count"], inputs["k"]
    live = inputs["src"].flatten()[:count]
    if expected["values"].numel() != k:
        raise ValueError("the reference holds exactly k values")
    if k < count and float(expected["values"].min()) <= float(live.min()):
        raise ValueError("the selection must exclude something, or it is not a selection")
    if not bool(torch.equal(expected["values"], torch.sort(expected["values"],
                                                           descending=True).values)):
        raise ValueError("the reference is in descending order, which is what the sort compares to")


def execute(case, inputs, launcher, backend):
    """One launch. Both destinations arrive poisoned and seeded in: NaN for the values and -1 for the
    indices, so an unfilled slot cannot be mistaken for a selection."""
    op = OpExec(select_largest, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    before = inputs["src"].view(torch.int32).clone()
    values, indices = op(inputs["src"], torch.full((1, SLOTS), VALUE_POISON),
                         torch.full((1, SLOTS), INDEX_POISON, dtype=torch.int32),
                         inputs["count"], inputs["k"])
    if not torch.equal(inputs["src"].view(torch.int32), before):
        raise ValueError("the launch must not modify its source")
    return {"values": values.flatten()[:inputs["k"]], "indices": indices, "raw_values": values}


def check_selection(inputs, actual):
    """The index side, which has no reference. Returns what failed."""
    count, k = inputs["count"], inputs["k"]
    src = inputs["src"].flatten()
    ids = actual["indices"].flatten()[:k].to(torch.int64)
    problems = []
    if int(ids.numel()) != k:
        problems.append(f"expected {k} indices, got {int(ids.numel())}")
    if bool((ids < 0).any()) or bool((ids >= count).any()):
        unfilled = int((ids == INDEX_POISON).sum())
        problems.append(f"an index is outside [0, {count})"
                        + (f"; {unfilled} still hold the {INDEX_POISON} fill" if unfilled else ""))
    elif int(torch.unique(ids).numel()) != k:
        problems.append(f"only {int(torch.unique(ids).numel())} of {k} indices are distinct")
    elif not torch.equal(actual["values"].view(torch.int32), src[ids].view(torch.int32)):
        # Compared as bits: a value that merely equals its source numerically is not the same lane.
        problems.append("a returned value is not the bits of the lane its index names")
    print(f"    {'indices':7s} {'ok  ' if not problems else 'FAIL'}  {k} in range, distinct, and "
          f"each naming its value's lane")
    for line in problems:
        print(f"      {line}")
    return problems


def compare(name, got, want):
    """The selected multiset, compared after sorting: the device's order is not the contract."""
    if got.dtype != want.dtype or got.numel() != want.numel():
        print(f"    {name:7s} FAIL  {got.dtype}[{got.numel()}] != {want.dtype}[{want.numel()}]")
        return False
    got, want = got.cpu().flatten(), want.cpu().flatten()
    ordered = torch.sort(got, descending=True).values
    ok = torch.equal(ordered.view(torch.int32), want.view(torch.int32))
    print(f"    {name:7s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} values, sorted")
    if not ok:
        differ = (ordered != want).nonzero().flatten().tolist()
        unfilled = int(torch.isnan(got).sum())
        print(f"      {len(differ)}/{got.numel()} ranks differ, first at {differ[:6]}: got "
              f"{ordered[differ[:3]].tolist()} against {want[differ[:3]].tolist()}"
              + (f"; {unfilled} are still the NaN fill (never selected)" if unfilled else ""))
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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (k={p['k']} of {p['count']} live in {LANES}, {p['dataset']} values, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
        if check_selection(inputs, actual):
            failed.append(f"{case['id']}/indices")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
