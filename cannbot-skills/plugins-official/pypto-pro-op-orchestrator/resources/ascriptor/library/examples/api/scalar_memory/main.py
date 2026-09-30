# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load a UB element into a scalar cell, copy the cell, and store both to one lane each.

    python main.py                        # every case, functional simulator, a5
    python main.py --list                 # the case ids, with their purpose
    python main.py --case operators       # one of them
    python main.py --device a2            # the same source against another facade
    python main.py --launcher pipesim     # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn       # the cce backend, on this machine's card
    python main.py --backend pto_isa --launcher board     # the PTO ISA backend on a card
    python main.py --backend pypto_pro --launcher pypto   # the PyPTO Pro toolchain's board

One 1x8 FP32 row is staged in UB. `Var(-999.0, f32)` declares a cell with a sentinel the kernel
must overwrite: column 2 is read into it, the cell is copied, the original is incremented, and the
two cells are stored to columns 3 and 4. Every other lane stays the copied input.

Two cases spell the same two instructions differently -- `GetValueFrom` / `SetValueTo` against
`<<=` / `>>=` -- and must produce identical bytes. Four facades (`a2`, `a3`, `a5`, `a5pr`) share
one kernel source, selected by `--device`, and one reference.

The comparison is bitwise. All eight lanes are defined: six are copies, one is a single FP32
addition, one is the copied cell, and -999 is nowhere in the reference -- so a load that never
happened shows up as a sentinel in the output rather than as a small numerical difference.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_scalar
from reference import make_inputs, reference

DEVICES = ("a5", "a2", "a3", "a5pr")

SENTINEL = -999.0  # the cell's constructor value, which a correct run never stores

OUTPUTS = ("o",)

CASES = [
    {"id": "methods", "seed": 8941, "block_dim": 1,
     "purpose": "The explicit spelling: `value.GetValueFrom(local[:, 2:3])` and "
                "`value.SetValueTo(local[:, 3:4])`. Both take a view and address only its first "
                "element, which is why every view here is narrowed to a single column",
     "parameters": {"spelling": "methods"}},
    {"id": "operators", "seed": 8941, "block_dim": 1,
     "purpose": "The same body with `<<=` and `>>=` in place of the two method calls, on the "
                "same seed: identical bytes are what establishes the operators as spellings of "
                "those instructions rather than a second path with its own rounding",
     "parameters": {"spelling": "operators"}},
]


def check_domain(inputs, expected):
    """Three statements the cases rest on: the shape the kernel declares, that the reference
    computes the two stored lanes from column 2 rather than from anywhere else, and that the
    sentinel is absent from the expected output -- the last one is what makes a missing load
    visible instead of merely wrong."""
    x = inputs["x"]
    if x.shape != (1, 8) or x.dtype != torch.float32:
        raise ValueError("the input must be float32[1, 8]")
    out = expected["o"]
    if out[0, 3] != x[0, 2] + 1.0 or out[0, 4] != x[0, 2]:
        raise ValueError("the reference must store column 2 incremented and copied")
    if (out == SENTINEL).any():
        raise ValueError(f"the reference must not contain the cell's sentinel {SENTINEL}")


def execute(case, inputs, launcher, backend, device):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so a lane
    the final `o <<= local` never covered reads back as NaN."""
    entry = make_scalar(device, inputs["spelling"])
    op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full_like(inputs["x"], float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: one FP32 addition and seven copies, with no operation that could legitimately
    round differently from the host's."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} lanes")
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        outside = raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
        index = outside.nonzero().flatten().tolist()
        sentinels = [i for i in index if flat_got[i].item() in (SENTINEL, SENTINEL + 1.0)]
        note = (f"; lanes {sentinels} still hold the cell's constructor value, so the UB load "
                f"never replaced it" if sentinels else "")
        print(f"      lanes {index} differ: got "
              f"{[round(flat_got[i].item(), 4) for i in index[:4]]} want "
              f"{[round(flat_want[i].item(), 4) for i in index[:4]]}{note}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:10s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (spelling={case['parameters']['spelling']}, device={args.device}, "
              f"backend={args.backend}, launcher={args.launcher})")
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
