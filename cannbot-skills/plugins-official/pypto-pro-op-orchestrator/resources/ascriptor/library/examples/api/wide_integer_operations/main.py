# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Wide integer arithmetic: 64-bit signed and unsigned lanes, and per-lane variable shifts.

    python main.py                        # every case, functional simulator
    python main.py --list                 # the case ids, with their purpose
    python main.py --case edges_one_row   # one of them
    python main.py --only unsigned        # one family: native, extended, unsigned or shifts
    python main.py --launcher pipesim     # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn       # the cce backend, on this machine's card

Forty-four named outputs over four families, each with its own destination so a missing operation
fails as itself:

    native      one 64-bit signed register per lane: add, sub, mul, and, or, xor, not, two fixed
                shifts, dup of a literal, two compare-selects, abssub and square-plus-b
    extended    the element-wise and reducing operations, including cadd / cmax / cmin, whose
                result is one lane per row rather than a full row
    unsigned    the same core over UINT64 lanes, which is where the ordering differs
    shifts      a *per-lane* shift count from a second register, at 32 and 64 bits

The unsigned family is the reason the signed one is not enough. `select_reg` picks by `x > y`, and
an unsigned comparison disagrees with a signed one on exactly the pairs whose high bits differ;
`shift_right` is logical there and arithmetic in the signed family. `check_domain` counts those
disagreeing lanes and refuses a case that has none -- without them the unsigned outputs could be
produced by signed instructions and still match.

The reference is Python's unbounded integers, wrapped to 64 bits only where the carrier does
(`% 2**64` for the unsigned family, `signed(..., bits)` for a left shift). Nothing here has a
tolerance: a floating comparison would excuse exactly the lost low bits this example is about.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse
from importlib import import_module

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from reference import MOD, PLANS, make_inputs, output_name, reference, signed

DEVICE = "a5"

POISON = -777                          # for a signed destination
U64_POISON = -6510615555426900571      # 0xA5A5A5A5A5A5A5A5 read as a signed 64-bit value
REDUCED = ("cadd", "cmax", "cmin")     # one lane per row, not a full row
FAMILIES = ("native", "extended", "unsigned", "shifts")

OUTPUTS = tuple(output_name(module, dtype, name)
                for module, _, names, _, dtype in PLANS for name in names)

CASES = [
    {"id": "edges_one_row", "seed": 103500, "block_dim": 1,
     "parameters": {"rows": 1, "dataset": "edges"},
     "purpose": "One row of hand-picked extremes: 0, 1, 2**63 - 1, 2**63, 2**64 - 1 for the "
                "unsigned lanes and the signed boundaries for the others. This is the case where "
                "an unsigned comparison and a signed one disagree the most often"},
    {"id": "source_64_rows", "seed": 0, "block_dim": 1,
     "parameters": {"rows": 64, "dataset": "source"},
     "purpose": "The original 64-row arithmetic composition, with values spread over the "
                "no-overflow domain -- the full row count, so every lane of the vector is live"},
    {"id": "random_three_rows", "seed": 103501, "block_dim": 1,
     "parameters": {"rows": 3, "dataset": "random"},
     "purpose": "Three rows of random values, including 64 random bits per unsigned lane, so no "
                "structure in the data can accidentally satisfy a wrong instruction"},
]


def check_domain(inputs, expected):
    """The literals the dup outputs must carry, the shape the reductions must have, and -- the one
    that matters -- proof that this case actually distinguishes unsigned from signed."""
    if not bool((expected["native_dup"] == (1 << 40) + 12345).all()):
        raise ValueError("native dup must be its 64-bit literal in every lane")
    if not bool((expected["unsigned_dup"].view(torch.int64)
                 == signed((1 << 63) + (1 << 40) + 7)).all()):
        raise ValueError("unsigned dup must be its high-bit literal in every lane")
    for name in REDUCED:
        if expected["extended_" + name].shape[1] != 1:
            raise ValueError(f"{name} reduces a row to one lane")
    ua = [v % MOD for v in inputs["unsigned_a"].view(torch.int64).flatten().tolist()]
    ub = [v % MOD for v in inputs["unsigned_b"].view(torch.int64).flatten().tolist()]
    ordering = sum((x > y) != (signed(x) > signed(y)) for x, y in zip(ua, ub))
    if not ordering:
        raise ValueError("no lane pair orders differently unsigned than signed; this case cannot "
                         "tell an unsigned compare-select from a signed one")
    if not any(x >> 63 for x in ua):
        raise ValueError("no unsigned lane has its high bit set; a logical shift right would be "
                         "indistinguishable from an arithmetic one here")
    counts = inputs["shift32_count"].flatten().tolist() + inputs["shift64_count"].flatten().tolist()
    if not all(0 <= c <= 30 for c in counts) or len(set(counts)) < 2:
        raise ValueError("the per-lane shift counts are in 0..30 and must not all be equal")


def execute(case, inputs, launcher, backend, only):
    """One launch per entry. A signed destination arrives filled with -777 and an unsigned one with
    0xA5A5A5A5A5A5A5A5, both seeded in, so an unwritten lane is a value the arithmetic never makes."""
    outputs = {}
    for module, entry_name, names, input_names, dtype in PLANS:
        if only not in (None, module):
            continue
        entry = getattr(import_module("kernel." + module), entry_name)
        arguments = [inputs[key] for key in input_names]
        source = arguments[0]
        if dtype == "u64":
            poisoned = [torch.full(source.shape, U64_POISON, dtype=torch.int64).view(torch.uint64)
                        for _ in names]
        else:
            poisoned = [torch.full_like(source, POISON) for _ in names]
        op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                    block_dim=case["block_dim"],
                    out_dir=f"tmp/{launcher}/{case['id']}/{entry_name}", seed_outputs=True)
        produced = op(*arguments, *poisoned, source.shape[0])
        values = produced if isinstance(produced, (tuple, list)) else [produced]
        for name, value in zip(names, values, strict=True):
            if module == "extended" and name in REDUCED:
                # The kernel writes a full row and only its first lane is defined.
                value = value[:, :1].contiguous()
            outputs[output_name(module, dtype, name)] = value
    return outputs


def compare(name, got, want):
    """Bitwise. Python's unbounded integers give the exact answer, wrapped only where the carrier
    wraps, so every one of the 64 bits is accounted for on both sides."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:23s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:23s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} "
          f"{str(got.dtype).removeprefix('torch.')} lanes")
    if not ok:
        raw = (got.reshape(-1).view(torch.int64) if got.dtype == torch.uint64 else got.reshape(-1),
               want.reshape(-1).view(torch.int64) if want.dtype == torch.uint64 else want.reshape(-1))
        differ = raw[0] != raw[1]
        index = differ.nonzero().flatten().tolist()
        fill = U64_POISON if got.dtype == torch.uint64 else POISON
        unwritten = int((raw[0][differ] == fill).sum())
        # A lost low word and a lost sign look different; printing the XOR says which.
        xor = [f"{(int(a) ^ int(b)) & (MOD - 1):#x}" for a, b in
               zip(raw[0][index[:3]].tolist(), raw[1][index[:3]].tolist())]
        print(f"      {len(index)}/{got.numel()} lanes differ, first at {index[:6]}; got "
              f"{raw[0][index[:3]].tolist()} against {raw[1][index[:3]].tolist()}; xor {xor}"
              + (f"; {unwritten} still hold the fill (never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--only", default=None, choices=FAMILIES,
                        help="run one family instead of all four")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:20s} {case['purpose']}")
        print(f"\n{len(OUTPUTS)} outputs over the families: " + ", ".join(FAMILIES))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    wanted = {output_name(module, dtype, name)
              for module, _, names, _, dtype in PLANS for name in names
              if args.only in (None, module)}
    names = [name for name in OUTPUTS if name in wanted]
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['rows']} row(s) of 32 lanes, {p['dataset']} data, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.only)
        for name in names:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
