# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar control flow on the device: a static unroll, a runtime loop, branches, break, continue.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --case past_break    # one of them
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

One `Var`, one INT32 output, no vector arithmetic and no transfers to align. The point is which parts
of the control flow are decided when:

    `for _ in unroll(3)`   at compile time -- three copies of the body, always
    `for index in range(n)` at run time -- `n` is a scalar argument
    `break` at index 9      at run time, so a larger n changes nothing
    `continue` on even index at run time, per iteration
    `if index > 4`          at run time, choosing which term to add

Five values of `n` cover the boundaries that matter: no iterations at all, one, the value where the
`> 4` branch has not yet been taken, exactly the break point, and past it. The two largest must give
the same answer, which is the whole content of the `break`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import scalar_control
from reference import LIMIT, UNROLLED, make_inputs, reference, total

DEVICE = "a2"

POISON = -999     # an output the kernel never wrote; no total here can be negative

OUTPUTS = ("o",)

CASES = [
    {"id": "no_iterations", "seed": 0, "block_dim": 1, "parameters": {"n": 0},
     "purpose": "n = 0: the runtime loop never runs, so the answer is the static unroll's 3 alone. "
                "This is the case that separates the two kinds of loop"},
    {"id": "one_iteration", "seed": 0, "block_dim": 1, "parameters": {"n": 1},
     "purpose": "n = 1: index 0 is even, so `continue` skips it and the answer is still 3. A kernel "
                "that counted the skipped iteration disagrees here"},
    {"id": "before_the_branch", "seed": 0, "block_dim": 1, "parameters": {"n": 4},
     "purpose": "n = 4: indices 1 and 3 contribute, both at the single rate, because neither is past "
                "4. The `> 4` branch has not been taken yet"},
    {"id": "at_the_break", "seed": 0, "block_dim": 1, "parameters": {"n": LIMIT},
     "purpose": "n = 9: exactly the break point, so every contributing index is included and the "
                "doubled ones are too"},
    {"id": "past_the_break", "seed": 0, "block_dim": 1, "parameters": {"n": 12},
     "purpose": "n = 12: the loop bound is past the break, and the answer must equal the case above. "
                "That equality is the only thing that shows the `break` happens at all"},
]


def check_domain(inputs, expected):
    """The answer must be positive, distinct from the fill, and -- past the break -- unchanged."""
    n = inputs["n"]
    if int(expected["o"].item()) <= 0 or int(expected["o"].item()) == POISON:
        raise ValueError("every total here is positive and unlike the fill value")
    if n == 0 and int(expected["o"].item()) != UNROLLED:
        raise ValueError("with no iterations the answer is the unroll's contribution alone")
    if n > LIMIT and total(n) != total(LIMIT):
        raise ValueError("past the break the answer must stop changing, or there is no break")


def execute(case, inputs, launcher, backend):
    """One launch. The output arrives at -999 and is seeded in, so an unwritten total is negative."""
    op = OpExec(scalar_control, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(torch.full((1,), POISON, dtype=torch.int32), inputs["n"])}


def compare(name, got, want):
    """Exact: one integer, computed by the same branches."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  total {int(got.item())}"
          + ("" if ok else f", expected {int(want.item())}"
                           + (" (still the fill: never written)"
                              if int(got.item()) == POISON else "")))
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed, answers = [], {}
    for case in selected:
        print(f"{case['id']}  (n={case['parameters']['n']}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        answers[case["id"]] = actual["o"]
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    if "at_the_break" in answers and "past_the_break" in answers:
        # Neither case alone says the break happened; their equality is what does.
        same = torch.equal(answers["at_the_break"], answers["past_the_break"])
        print(f"\nbreak at {LIMIT}: n=12 gives {'the same' if same else 'a DIFFERENT'} total as n=9")
        if not same:
            failed.append("past_the_break/o (differs from at_the_break)")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
