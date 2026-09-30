# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vector-to-cube-to-vector ownership, compact NZ publication, and FIX destination pitch.

    python main.py                               # every case, functional simulator
    python main.py --list                        # the case ids, with their purpose
    python main.py --case fix_pitched_columns    # one of them
    python main.py --launcher pipesim            # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn              # the cce backend, on this machine's card
    python main.py --fault wrong_pitch           # a deliberately broken kernel; it must fail

This is an address and ownership example. It is not an attention algorithm and it makes no timing
claim. Three families share one folder because they share one ownership skeleton -- `VcMutex` over
the L1 producer ring, `CvMutex` over each vector consumer ring, both participants running every
beat:

  `three_beats`, `second_seed`  each participant scales 16 FP16 rows, publishes to L1, and reads
                               back `abs(product) + 1` from its own 16 result rows.
  `*_single8`, `*_dual8`       exactly representable b16 labels travel through a register and a
                               nine-row-pitch staging tile into four NZ columns, multiplied by a
                               64x64 identity so the output *is* the address readback. The complete
                               staging allocation is captured and compared byte for byte, so the
                               padding row and the four guard columns are checked too.
  `fix_*`                      two labelled FIX writes per beat into UB, with every destination byte
                               read back after each one: the first write must leave the rest
                               poisoned, and the second must preserve the first.

`source[row, :]` reads all 128 allocated lanes; `MaskReg(dtype, LOWHALF)` limits the *store* to the
first 64. Shrinking the row allocation to 64 is a different program.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_fix_views, make_narrow, roundtrip
from reference import (FIX_CASES, NARROW_CASES, STAGING_POISON, fix_output_shape,
                       fix_parameters, make_inputs, narrow_parameters, packed_reference,
                       reference)

DEVICE = "a5"

POISON = float("nan")     # every FP32 destination; an unwritten lane is a NaN, never a plausible one
CAPTURE_POISON = -29.0    # the staging capture's own fill, distinct from the tile's -13
FAULTS = ("missing_mask", "wrong_pitch", "missing_ready", "compact_destination")

# pto_isa: `fix_pitched_columns` is refused at source emission, at kernel.py:200:24 --
# `dma.l0c_to_ub: the transfer needs 4096 f32 elements from local, which holds 3968 past this
# window`. Keeping the parent's pitch means the view declares the parent's whole 16x256 extent while
# its origin sits 128 elements in, and pto_isa's window arithmetic wants origin + declared size to
# fit the allocation. The other two FIX cases emit on pto_isa, and pypto_pro emits all three -- its
# adapter narrows a native Tile's valid extent instead of re-declaring the storage. The refusal is
# recorded rather than worked around: a compact alias would satisfy the check and write the wrong
# rows, which is what `--fault compact_destination` shows. Reproduce without a card with
# `compile_kernel(make_fix_views("pitched_columns"), backend="pto_isa")`.
REFUSED = {("pto_isa", "fix_pitched_columns"):
           "the pitched column view declares the parent's 4096 FP32 elements from an origin with "
           "3968 left past it; keeping the parent row pitch is the point of the case"}

CASES = [
    {"id": "three_beats", "seed": 8991, "block_dim": 1, "parameters": {},
     "purpose": "The original roundtrip: three beats, both participants owning 16 rows each, "
                "staging at UB NZ pitch 17 with only 16 rows live. Bounded integer inputs keep the "
                "product exact through the FP16 preprocessing pass"},
    {"id": "second_seed", "seed": 8992, "block_dim": 1, "parameters": {},
     "purpose": "The same program on different data, so an answer that happens to be right for one "
                "input alignment is not the whole evidence"},
    {"id": "f16_single8", "seed": 9011, "block_dim": 1,
     "parameters": narrow_parameters("f16_single8"),
     "purpose": "Eight live FP16 rows out of sixteen executed: participant 1 fills its register "
                "with zero and still publishes all eight of its rows. Rows 8-15 of the input hold "
                "non-zero labels, so a missing zero fill is visible in the identity readback"},
    {"id": "bf16_single8", "seed": 9012, "block_dim": 1,
     "parameters": narrow_parameters("bf16_single8"),
     "purpose": "The same eight-live-row program in BF16. Both formats are 2-byte, so LOWHALF "
                "activates 64 lanes either way -- the storage format changes no address here, and "
                "this case is what says so"},
    {"id": "f16_dual8", "seed": 9013, "block_dim": 1,
     "parameters": narrow_parameters("f16_dual8"),
     "purpose": "Sixteen live FP16 rows: both participants publish eight data rows into disjoint L1 "
                "rows, so a participant that computed the wrong base row overwrites the other's"},
    {"id": "bf16_dual8", "seed": 9014, "block_dim": 1,
     "parameters": narrow_parameters("bf16_dual8"),
     "purpose": "The sixteen-live-row program in BF16"},
    {"id": "fix_contiguous", "seed": 9061, "block_dim": 1,
     "parameters": fix_parameters("fix_contiguous"),
     "purpose": "The FIX destination is a complete contiguous [16,128] allocation -- the control "
                "the two pitched arrangements are read against"},
    {"id": "fix_pitched_columns", "seed": 9062, "block_dim": 1,
     "parameters": fix_parameters("fix_pitched_columns"),
     "purpose": "Two aligned column halves of one [16,256] allocation. Physical rows start every "
                "1024 bytes and the two views at bytes 0 and 512; both starts are 32-byte aligned "
                "and neither view is contiguous. The subview keeps the parent pitch, and the "
                "readback after the first write proves the right half is still poisoned"},
    {"id": "fix_independent_ubs", "seed": 9063, "block_dim": 1,
     "parameters": fix_parameters("fix_independent_ubs"),
     "purpose": "The same two products into two separate [16,128] allocations, assembled only in "
                "GM. Compared against the pitched case it separates 'two destinations' from 'two "
                "views of one destination'"},
]


def family(case_id):
    return "fix" if case_id in FIX_CASES else "narrow" if case_id in NARROW_CASES else "roundtrip"


def outputs_of(case_id):
    """The staging capture is a compared output, not a side record: without it the identity readback
    alone would accept a tile whose padding and guard columns had been written over."""
    return ("o", "capture") if family(case_id) == "narrow" else ("o",)


def expected_of(inputs, case_id):
    expected = reference(inputs)
    if family(case_id) == "narrow":
        expected["capture"] = packed_reference(inputs)
    return expected


def check_domain(inputs, expected, case):
    """The declared geometry, and -- for the two cases whose subject is what was *not* written -- the
    property that makes their comparison mean anything."""
    kind = family(case["id"])
    shape = (fix_output_shape(case["id"]) if kind == "fix"
             else (5, 16, 64) if kind == "narrow" else (3, 32, 16))
    if set(expected) != set(outputs_of(case["id"])):
        raise ValueError("the reference must publish exactly this case's named outputs")
    if expected["o"].dtype != torch.float32 or tuple(expected["o"].shape) != shape:
        raise ValueError("the reference must retain the declared FP32 output geometry")
    if kind == "narrow":
        capture = expected["capture"]
        if capture.dtype != inputs["x"].dtype or tuple(capture.shape) != (5, 2, 9, 128):
            raise ValueError("the capture reference must match the staging allocation exactly")
        # A live element sits at `16 * row + 144 * column // 16 + column % 16` in the allocation --
        # the 144 is the nine-row pitch times sixteen lanes -- so the eight live rows of NZ column q
        # fill offsets [144q, 144q + 128) and its padding row is the sixteen after that. NZ columns
        # four to seven are never stored into at all. Neither region has a row of its own in the
        # [9, 128] view, which is exactly why the capture is compared as bytes.
        flat = capture.reshape(5, 2, -1)
        untouched = [flat[:, :, 144 * q + 128:144 * q + 144] for q in range(4)] + [flat[:, :, 576:]]
        if not all(bool((region == STAGING_POISON).all()) for region in untouched):
            raise ValueError("the padding rows and the guard columns must still hold the tile poison")
    if kind == "fix" and case["parameters"]["mode"] != "contiguous":
        # After the first of the two writes the right half is still the caller's poison. If the
        # reference did not say so, the readback could not tell a preserved half from a rewritten one.
        if not torch.equal(expected["o"][:, 0, :, :, 128:],
                           inputs["initial"][:, 128:].expand(5, 2, 16, 128)):
            raise ValueError("the first write's reference must leave the right half poisoned")


def execute(case, inputs, launcher, backend, fault):
    """One launch per case. Every destination and the staging capture arrive filled and seeded in, so
    a lane the kernel never wrote is a poison value rather than a leftover."""
    kind = family(case["id"])
    if kind == "fix":
        entry = make_fix_views(FIX_CASES[case["id"]], fault=fault)
    elif kind == "narrow":
        entry = make_narrow(*NARROW_CASES[case["id"]], fault=fault)
    else:
        entry = roundtrip
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    if kind == "fix":
        o = op(inputs["x"], inputs["y"], inputs["initial"],
               torch.full(fix_output_shape(case["id"]), POISON))
        return {"o": o}
    if kind == "narrow":
        o, capture = op(inputs["x"], inputs["y"], inputs["initial"],
                        torch.full((5, 16, 64), POISON),
                        torch.full((5, 2, 9, 128), CAPTURE_POISON, dtype=inputs["x"].dtype))
        return {"o": o, "capture": capture}
    return {"o": op(inputs["x"], inputs["y"], torch.full((3, 32, 16), POISON))}


def compare(name, got, want):
    """Bitwise. Small-integer scaling and an identity product are exact, the FIX labels are one FP64
    product rounded once, and a retained poison byte has to be the same byte."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:8s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    raw = (got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    ok = torch.equal(*raw)
    print(f"    {name:8s} {'ok  ' if ok else 'FAIL'}  bitwise over {raw[0].numel()} bytes "
          f"({got.numel()} {str(got.dtype).removeprefix('torch.')} elements)")
    if not ok:
        differ = (raw[0] != raw[1]).reshape(got.shape + (-1,)).any(-1)
        index = differ.nonzero()
        # A destination lane the kernel never wrote still holds the fill it was seeded with.
        fill = CAPTURE_POISON if name == "capture" else POISON
        unwritten = int((torch.isnan(got) if fill != fill else got == fill).sum())
        print(f"      {int(differ.sum())}/{got.numel()} elements differ, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}"
              + (f"; {unwritten} still hold the {fill} fill (never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--fault", default=None, choices=FAULTS,
                        help="build a deliberately broken kernel and report what catches it; "
                             "never run one of these on hardware")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:20s} {family(case['id']):9s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    if args.fault:
        return fault_probe(selected, args)
    failed, skipped = [], []
    for case in selected:
        print(f"{case['id']}  ({family(case['id'])}, launcher={args.launcher}, "
              f"backend={args.backend})")
        reason = REFUSED.get((args.backend, case["id"]))
        if reason:
            print(f"    skipped on {args.backend}: {reason}")
            skipped.append(case["id"])
            continue
        inputs = make_inputs(case)
        expected = expected_of(inputs, case["id"])
        check_domain(inputs, expected, case)
        actual = execute(case, inputs, args.launcher, args.backend, None)
        for name in outputs_of(case["id"]):
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    ran = len(selected) - len(skipped)
    print(f"\n{ran - len({f.split('/')[0] for f in failed})}/{ran} cases passed"
          + (f", {len(skipped)} skipped on {args.backend}" if skipped else ""))
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


def fault_probe(selected, args):
    """A deliberately broken kernel, so the checks above can be seen to have teeth. Each mutation
    must be refused by the launcher or produce a failing output; a fault that passes means the
    comparison it was aimed at is not doing its job, and that is what this reports."""
    wanted = "fix" if args.fault == "compact_destination" else "narrow"
    cases = [case for case in selected if family(case["id"]) == wanted
             and (args.fault != "compact_destination"
                  or case["parameters"]["mode"] == "pitched_columns")]
    if not cases:
        print(f"--fault {args.fault} belongs to the {wanted} family; none of the selected cases is "
              f"one. --list prints them")
        return 1
    caught = 0
    for case in cases:
        print(f"{case['id']}  --fault {args.fault}  (launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = expected_of(inputs, case["id"])
        try:
            actual = execute(case, inputs, args.launcher, args.backend, args.fault)
        except Exception as error:      # the launcher refusing the mutation is a catch
            print(f"    refused by {args.launcher}: {type(error).__name__}: "
                  f"{str(error).splitlines()[0]}")
            caught += 1
            continue
        wrong = [name for name in outputs_of(case["id"])
                 if not compare(name, actual[name], expected[name])]
        if wrong:
            print("    caught by " + ", ".join(wrong))
        else:
            print("    NOT CAUGHT -- every output still matches, so this mutation is invisible to "
                  "the comparisons above")
        caught += bool(wrong)
    print(f"\n{caught}/{len(cases)} faulted runs were caught")
    return 0 if caught == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
