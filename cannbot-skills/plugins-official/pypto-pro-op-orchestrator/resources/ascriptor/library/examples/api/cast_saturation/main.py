# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""All four CTRL saturation modes against a per-instruction CastConfig, and the restoration.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case specials_initial_2  # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

Two flags decide whether a narrowing cast saturates or wraps, and the raw bits mean this:

    global=0            defer to the instruction's CastConfig(saturate=...)
    global=1, cast=0    force saturation, whatever CastConfig says
    global=1, cast=1    force wrapping, whatever CastConfig says

Every case walks all four (global, cast) pairs in one launch, and for each pair runs four
conversions to int16: an integer one with `saturate=True`, the same with `saturate=False`, then the
float pair. So row `m` column block `r` of the output answers "what did CTRL mode `m` do to
conversion `r`", and the reference states the same rule as `mode == 2 or (mode < 2 and r even)`.

`RegLayout.ZERO` puts the results in even lanes and zeros the odd ones, and the odd lanes are
compared too. Row 4 publishes whether the case's initial flag pair came back, after which the
kernel also restores the launch's own pair. A full barrier retires outstanding work before every
flag change, because the flags are mode state and not an operand.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import cast_saturation
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 12345
ROWS, LANES = 5, 512     # four CTRL modes plus the restoration row; four 128-lane registers each
MODE_NAMES = ("global=0 cast=0 (CastConfig decides)", "global=0 cast=1 (CastConfig decides)",
              "global=1 cast=0 (forced saturation)", "global=1 cast=1 (forced wrapping)")

OUTPUTS = ("o",)

FAMILIES = {
    False: "Ties at every boundary: +/-0.5, +/-32767.5, +/-32768.5, +/-40000.5 against integer "
           "inputs -40000, -32769, -32768, -1, 0, 32767, 32768, 40000. Ties-to-even and the "
           "wrapping arithmetic are both decided at those exact points",
    True: "The edges: -inf, +inf, NaN, +/-2^40 and -32768.0 exactly. A saturating conversion has "
          "to answer the type's limits for the infinities and zero for the NaN, and a wrapping one "
          "has to agree with the reference's modular arithmetic on 2^40",
}

CASES = [
    {"id": f"{'specials' if edges else 'ties'}_initial_{mode}", "seed": 8851 + mode,
     "block_dim": 1,
     "purpose": f"{FAMILIES[edges]}. The launch arrives at {MODE_NAMES[mode]}, which is what row "
                f"4 has to restore -- every case still walks all four modes inside the launch",
     "parameters": {"initial_mode": mode, "edges": edges}}
    for edges in (False, True)
    for mode in (0, 1, 2, 3)
]


def check_domain(inputs, expected):
    """What the rows mean. Row 4 must be expected zero (the restoration succeeded), the odd lanes
    must be expected zero (RegLayout.ZERO), and the poison must not be a legitimate answer."""
    values = expected["o"]
    if values.shape != (ROWS, LANES):
        raise ValueError(f"the reference must be int16[{ROWS}, {LANES}]")
    if values[4].count_nonzero():
        raise ValueError("row 4 is the restoration status and must be expected zero")
    if values[:, 1::2].count_nonzero():
        raise ValueError("the odd lanes must be expected zero under RegLayout.ZERO")
    if (values == POISON).any():
        raise ValueError("the reference contains the poison value")
    if inputs["initial_mode"] not in range(4):
        raise ValueError("initial_mode is a raw two-bit CTRL pair")


def execute(case, inputs, launcher, backend):
    """One launch covering all four modes. The destination arrives filled with 12345 and seeded in,
    so a row nothing published -- including the restoration row, whose expected value is zero --
    fails instead of passing."""
    op = OpExec(cast_saturation, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((ROWS, LANES), POISON, dtype=torch.int16)
    return {"o": op(inputs["x"], inputs["xf"], destination, inputs["initial_mode"])}


def compare(name, got, want):
    """Bitwise, and a failure names the CTRL mode and which of the four conversions disagreed."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over 4 CTRL modes x 4 conversions "
          f"+ the restoration row")
    if not ok:
        conversions = ("int sat", "int nosat", "float sat", "float nosat")
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            where = "restoration" if row == 4 else MODE_NAMES[row]
            if bool((got[row] == POISON).all()):
                print(f"      {where}: the whole row still holds the poison (never published)")
                continue
            for block in range(4):
                segment = slice(block * 128, (block + 1) * 128)
                lanes = (got[row, segment] != want[row, segment]).nonzero().flatten().tolist()
                if lanes:
                    label = conversions[block] if row < 4 else f"block {block}"
                    print(f"      {where} / {label}: {len(lanes)} lanes differ, got "
                          f"{got[row, segment][lanes[:4]].tolist()} want "
                          f"{want[row, segment][lanes[:4]].tolist()}")
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
        p = case["parameters"]
        print(f"{case['id']}  (launch {MODE_NAMES[p['initial_mode']]}, "
              f"{'edge' if p['edges'] else 'tie'} inputs, launcher={args.launcher})")
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
