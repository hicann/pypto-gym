# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Nine SIMT atomic families, each with an order-independent answer, after an all-vector rendezvous.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case contributor_set   # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

A `mode="mix"` launch with two vector participants, 256 SIMT threads each. 512 input values map
onto 64 columns, so every column has eight contributors, and each of the nine output rows is one
atomic family: add, sub, max, min, exchange, and, or, xor, compare-and-swap.

All nine are order-independent, and that is what makes them comparable at all. The first four and
the three bitwise ones are commutative; `exch` is only well defined here because every contributor
writes the same value (its column index); `cas` is only well defined because exactly one candidate
can ever match -- threads below the column count compare against `i`, which the row was seeded
with, and every other thread compares against -7, which nothing is.

The rows are seeded by a separate SIMT launch, and the seeding has to be visible to *both*
participants before any atomic runs: `allvec_ready` / `allvec_wait` on the V pipe is that
rendezvous. It is not a core-local barrier.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import N, ROWS, simt_atomic_family
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 123456

OUTPUTS = ("o",)

FAMILIES = ("add", "sub", "max", "min", "exch", "and", "or", "xor", "cas")

CASES = [
    {"id": "contributor_set", "seed": 8811, "block_dim": 1,
     "purpose": "Eight contributors per column across two vector participants of 256 threads. The "
                "reference counts contributors rather than modelling a thread order, which is "
                "only legitimate because all nine families here are order-independent",
     "parameters": {}},
    {"id": "nonzero_initial_rows", "seed": 8812, "block_dim": 1,
     "purpose": "A different seed, so the add and sub rows start from non-zero values "
                "(seed % 13 and seed % 17). An implementation that skipped the seeding launch "
                "would agree with the reference only when those happened to be zero",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """What makes each family's answer order-independent, asserted where it can be. The two rows
    whose independence is a property of the *kernel* rather than of the operation -- exchange and
    CAS -- are the ones checked here: exchange must be the column index, and CAS must be the
    successful candidate's value."""
    x, initial = inputs["x"], inputs["initial"]
    if x.shape != (1, 8 * N) or initial.shape != (ROWS, N):
        raise ValueError(f"the input must be int32[1, {8 * N}] with an int32[{ROWS}, {N}] seed")
    if (initial == POISON).any() or (expected["o"] == POISON).any():
        raise ValueError("the poison value must not be a legitimate answer")
    if not torch.equal(expected["o"][4], torch.arange(N, dtype=torch.int32)):
        raise ValueError("every exchange contributor writes the column index, so row 4 is arange")
    if not torch.equal(expected["o"][8], 1000 + torch.arange(N, dtype=torch.int32)):
        raise ValueError("exactly one CAS candidate matches, so row 8 is 1000 + the column index")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with 123456 and seeded in, but note what that
    catches here: the kernel seeds the rows itself from `initial`, so a poisoned lane in the output
    means neither the seeding launch nor any atomic reached it."""
    op = OpExec(simt_atomic_family, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((ROWS, N), POISON, dtype=torch.int32)
    return {"o": op(inputs["x"], inputs["initial"], destination)}


def compare(name, got, want):
    """Bitwise, and a failure names the atomic family whose row disagreed."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {ROWS} families x {N} columns")
    if not ok:
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            columns = (got[row] != want[row]).nonzero().flatten().tolist()
            unwritten = bool((got[row] == POISON).all())
            print(f"      {FAMILIES[row]:5s} {len(columns)} columns differ at {columns[:6]}: got "
                  f"{got[row][columns[:3]].tolist()} want {want[row][columns[:3]].tolist()}"
                  + ("  (the whole row still holds the poison: neither seeded nor contributed to)"
                     if unwritten else ""))
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
        print(f"{case['id']}  (seed={case['seed']}, 8 contributors per column, "
              f"launcher={args.launcher})")
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
