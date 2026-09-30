# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read the saturation flags back through a DMA slice index, then restore every level of state.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case initial_3      # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`get_saturation_flag` / `set_saturation_flag` read and write two pieces of mode state, `"cast"`
and `"float"`. The kernel makes the reads observable rather than trusted: it sets `cast` true and
`float` false, then uses the values it reads back as the *index* of the DMA source slice, so the
output is `x[8:16]` and `x[16:24]` exactly when the two reads answered 1 and 0.

Three levels of state are then unwound. The launch's own flags are saved before anything else; the
case's `initial_mode` pair is installed and saved; the kernel's own pair is set, used and restored
to the case's; and finally the launch's pair is put back. The `restored` output publishes the XOR
difference between what was read back and what was saved -- zero when the restoration worked.

`restored` arrives filled with 12345, not zero. A missing publication therefore fails instead of
passing because the reference happens to be zero, which is the failure mode this output is for.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import sat_flags
from reference import make_inputs, reference

DEVICE = "a5"

POISON = {"o": float("nan"), "restored": 12345}

OUTPUTS = ("o", "restored")

CASES = [
    {"id": f"initial_{mode}", "seed": 8861 + mode, "block_dim": 1,
     "purpose": f"The launch arrives with cast={bool(mode & 1)} and float={bool(mode & 2)}. "
                + ("Both flags start clear, so the kernel's restore has to put back a false it "
                   "never had to change." if mode == 0 else
                   "The cast flag starts set and the kernel sets it again, so a restore that "
                   "simply cleared both flags would still look right in the slice index and wrong "
                   "here." if mode == 1 else
                   "The float flag starts set and the kernel clears it, which is the one "
                   "combination where the kernel's own value differs from what it must restore."
                   if mode == 2 else
                   "Both flags start set, so both of the kernel's writes change state and both "
                   "have to be undone."),
     "parameters": {"initial_mode": mode}}
    for mode in (0, 1, 2, 3)
]


def check_domain(inputs, expected):
    """What the two outputs mean. The slices come from the flag reads, so the reference has to be
    those exact slices; and `restored` must be all zero, or the unit would be asserting that a
    failed restoration is the expected answer."""
    x = inputs["x"]
    if x.shape != (1, 32) or x.dtype != torch.float32:
        raise ValueError("the input must be float32[1, 32]")
    if not torch.equal(expected["o"],
                       torch.cat((x[:, 8:16], x[:, 16:24], x[:, 16:32]), dim=1)):
        raise ValueError("the reference must be the three slices the flag reads select")
    if expected["restored"].count_nonzero():
        raise ValueError("the restoration status must be expected zero in every lane")
    if POISON["restored"] in expected["restored"]:
        raise ValueError("the status poison must not be a legitimate answer")


def execute(case, inputs, launcher, backend):
    """One launch. Both destinations arrive poisoned and are seeded in: NaN for the data, 12345 for
    the status, so an unpublished status lane cannot read as a successful restoration."""
    op = OpExec(sat_flags, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    out = torch.full((1, 32), POISON["o"])
    status = torch.full((1, 8), POISON["restored"], dtype=torch.int32)
    produced = op(inputs["x"], out, status, inputs["initial_mode"])
    return dict(zip(OUTPUTS, produced, strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: three DMA slices, and eight status lanes that are independently expected zero."""
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
        if name == "restored":
            stale = [i for i in index if flat_got[i].item() == POISON["restored"]]
            note = ("; those lanes still hold the 12345 fill, so the status was never published"
                    if stale else "; a non-zero difference means a flag was not restored")
        else:
            poisoned = [i for i in index if flat_got[i] != flat_got[i]]
            note = ("; those lanes are still NaN-poisoned (never written)" if poisoned else
                    "; a wrong slice means a flag read back wrong")
        print(f"      lanes {index[:8]} differ: got {[flat_got[i].item() for i in index[:4]]} "
              f"want {[flat_want[i].item() for i in index[:4]]}{note}")
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
            print(f"{case['id']:11s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        mode = case["parameters"]["initial_mode"]
        print(f"{case['id']}  (launch cast={bool(mode & 1)} float={bool(mode & 2)}, "
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
