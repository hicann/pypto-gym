# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sort score/ID records, merge them, and validate every invariant before normalising a legal tie.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case tied_scores    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

A record is two 32-bit words: an FP32 score and a UINT32 ID. `sort32` sorts four lists of 32
records each, `mergesort4` merges those four into one list of 128, and `mergesort_2seq` merges the
first two into 64.

The order inside an equal-score group is legitimately undetermined, and this example does not
handle that by loosening the comparison. `canonical_records` first checks the things that *are*
determined -- the scores descend, every selected ID appears exactly once, and each ID is paired with
its own score -- and only then reorders tie groups by ID so the remaining comparison can be bitwise.
A duplicated or missing ID fails at that first step even when every gathered score ties.

IDs start at 1000, so an implementation that returned positions instead of IDs cannot pass by
coincidence.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import sort_family
from reference import make_inputs, reference

DEVICE = "a5"

LISTS, PER_LIST = 4, 32

OUTPUTS = ("sort", "merge4", "merge2")

CASES = [
    {"id": "distinct_records", "seed": 8801, "block_dim": 1,
     "purpose": "128 distinct scores from a permutation, so the order is fully determined and the "
                "comparison is over exact record bits with no normalisation doing any work",
     "parameters": {}},
    {"id": "different_order", "seed": 8802, "block_dim": 1,
     "purpose": "A second permutation of the same score set: a pass is then not a property of one "
                "input order, which for a sort is the thing most worth ruling out",
     "parameters": {}},
    {"id": "tied_scores", "seed": 8803, "block_dim": 1,
     "purpose": "Scores floored to multiples of 3, which puts several records in each equal-score "
                "group. Their internal order is undetermined, so this is the case where the "
                "invariant checks carry the weight: descending scores, each ID paired with its own "
                "score, and a complete permutation of the input IDs -- all verified before the tie "
                "order is normalised away",
     "parameters": {"ties": True}},
]


def canonical_records(records, source_scores, source_indices):
    """Validate one output list, then normalise its legal tie order.

    Returned as the record words, so the caller's comparison stays bitwise. Every check here is a
    property the sort must have regardless of how ties fell; the reordering at the end is the only
    freedom the primitive is allowed, and it happens after the checks rather than instead of them.
    """
    if records.dtype != torch.float32 or records.numel() != 2 * source_scores.numel():
        raise ValueError("records must contain exactly one FP32 score / UINT32 ID pair per input")
    words = records.contiguous().view(torch.int32).reshape(-1)
    scores, ids = records.reshape(-1)[::2], words[1::2].tolist()
    expected_ids = source_indices.contiguous().view(torch.int32).reshape(-1).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("a duplicated ID loses a selected input, even when its score ties")
    if set(ids) != set(expected_ids):
        raise ValueError("a missing or out-of-range input ID")
    if not bool(torch.all(scores[:-1] >= scores[1:])):
        raise ValueError("record scores must be in descending order")
    expected_bits = dict(zip(expected_ids,
                             source_scores.contiguous().view(torch.int32).reshape(-1).tolist(),
                             strict=True))
    if any(expected_bits[index] != bits
           for index, bits in zip(ids, words[::2].tolist(), strict=True)):
        raise ValueError("an ID does not select its accompanying score")
    order = sorted(range(len(ids)), key=lambda index: (-scores[index].item(), ids[index]))
    return words.reshape(-1, 2)[order].reshape(-1)


def check_domain(inputs, expected):
    """Two properties the ID checks depend on: the IDs are outside the range of positions, and they
    are unique -- without either, a sort that returned positions or repeated a record could pass."""
    ids = inputs["indices"].contiguous().view(torch.int32).reshape(-1).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("the input IDs must be unique")
    if min(ids) < inputs["x"].numel():
        raise ValueError("the IDs must start above the largest position, or returning positions "
                         "instead of IDs could pass")
    for name in OUTPUTS:
        if expected[name].dtype != torch.int32:
            raise ValueError(f"the {name} reference must be record words, not values")


def execute(case, inputs, launcher, backend):
    """One launch producing all three lists. Destinations arrive NaN-poisoned and are seeded in: an
    unwritten record would be a NaN score, which `canonical_records` rejects at its descending-order
    check rather than silently sorting to one end."""
    op = OpExec(sort_family, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full(shape, float("nan"))
                for shape in ((LISTS, 2 * PER_LIST), (LISTS, 2 * PER_LIST), (2, 2 * PER_LIST))]
    sort, merge4, merge2 = op(inputs["x"], inputs["indices"], *poisoned)
    scores, ids = inputs["x"], inputs["indices"]
    return {
        "sort": torch.stack([canonical_records(sort[row], scores[row], ids[row])
                             for row in range(LISTS)]),
        "merge4": canonical_records(merge4, scores, ids).reshape(LISTS, 2 * PER_LIST),
        "merge2": canonical_records(merge2, scores[:2], ids[:2]).reshape(2, 2 * PER_LIST),
    }


def compare(name, got, want):
    """Bitwise over the record words, after the tie order has been normalised on both sides."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:7s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:7s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel() // 2} records")
    if not ok:
        flat_got, flat_want = got.reshape(-1), want.reshape(-1)
        index = (flat_got != flat_want).nonzero().flatten().tolist()
        records = sorted({i // 2 for i in index})
        detail = ", ".join(
            f"record {r}: score bits 0x{flat_got[2 * r] & 0xFFFFFFFF:08x}/"
            f"0x{flat_want[2 * r] & 0xFFFFFFFF:08x} id {flat_got[2 * r + 1]}/{flat_want[2 * r + 1]}"
            for r in records[:3])
        print(f"      {len(records)} records differ; {detail}")
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        ties = case["parameters"].get("ties", False)
        print(f"{case['id']}  ({LISTS}x{PER_LIST} records, "
              f"{'with tie groups' if ties else 'all distinct'}, launcher={args.launcher})")
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
