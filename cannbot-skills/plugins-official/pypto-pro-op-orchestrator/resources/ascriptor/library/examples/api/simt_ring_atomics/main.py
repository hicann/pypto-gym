# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A wrapping SIMT increment and decrement, counted by contributor rather than by order.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case wrap_counts    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`simt_atomic_inc(cell, LIMIT)` increments and wraps to zero past `LIMIT`; `simt_atomic_dec` goes the
other way and wraps to `LIMIT` past zero. With `LIMIT = 5` the cycle is six values, 0 through 5.

256 threads over 64 columns is four contributors per column, so the answers are arithmetic: the
increment row starts at 0 and ends at 4, the decrement row starts at 5 and ends at 1. The reference
counts contributors and models no thread order at all -- which is the only honest reference for an
atomic, and is possible here because both operations commute over a cycle.

`simt_threadfence()` sits between the two atomics and is preserved as what it is: a thread fence,
not a global barrier.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import LIMIT, N, T, simt_atomic_incdec
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 0xDEADBEEF
CONTRIBUTORS = T // N  # 256 threads over 64 columns

OUTPUTS = ("o",)

CASES = [
    {"id": "wrap_counts", "seed": 8821, "block_dim": 1,
     "purpose": f"{CONTRIBUTORS} contributors per column over a cycle of {LIMIT + 1} values: the "
                f"increment row starts at 0 and ends at {CONTRIBUTORS % (LIMIT + 1)}, the "
                f"decrement row starts at {LIMIT} and ends at "
                f"{(LIMIT - CONTRIBUTORS) % (LIMIT + 1)}",
     "parameters": {}},
    {"id": "different_dummy", "seed": 8822, "block_dim": 1,
     "purpose": "The same kernel with a different dummy input, and identical output is the point. "
                "The dummy exists only to satisfy the input-tensor requirement -- it is copied to "
                "UB and never read -- so the counts come from the thread geometry rather than from "
                "any data, and this case is what states that",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """The arithmetic the reference claims, restated from the launch geometry rather than copied
    from it: four contributors, a six-value cycle, and a poison that is not a legitimate count."""
    if tuple(inputs["dummy"].shape) != (1, 8):
        raise ValueError("the dummy input must be uint32[1, 8]")
    wrapped_up = CONTRIBUTORS % (LIMIT + 1)
    wrapped_down = (LIMIT - CONTRIBUTORS) % (LIMIT + 1)
    if not bool((expected["o"][0] == wrapped_up).all()):
        raise ValueError(f"the increment row must be {wrapped_up} in every column")
    if not bool((expected["o"][1] == wrapped_down).all()):
        raise ValueError(f"the decrement row must be {wrapped_down} in every column")
    if (expected["o"] == POISON).any():
        raise ValueError("the poison value must not be a legitimate count")


def execute(case, inputs, launcher, backend):
    """One launch: a seeding SIMT body writes 0 and LIMIT, then 256 threads contribute. The
    destination arrives filled with 0xDEADBEEF and seeded in, so a column neither launch touched is
    distinguishable from one that wrapped back to a small number."""
    op = OpExec(simt_atomic_incdec, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((2, N), POISON, dtype=torch.uint32)
    return {"o": op(inputs["dummy"], destination)}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise over the two uint32 rows."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over 2 rows x {N} columns")
    if not ok:
        got, want = got.cpu(), want.cpu()
        for row, label in ((0, "inc"), (1, "dec")):
            columns = (got[row] != want[row]).nonzero().flatten().tolist()
            if columns:
                unwritten = [c for c in columns if int(got[row, c]) == POISON]
                print(f"      {label} {len(columns)} columns differ at {columns[:6]}: got "
                      f"{got[row][columns[:3]].tolist()} want {want[row][columns[:3]].tolist()}"
                      + (f"; {len(unwritten)} still hold the poison (never seeded)"
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({CONTRIBUTORS} contributors, cycle 0..{LIMIT}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
