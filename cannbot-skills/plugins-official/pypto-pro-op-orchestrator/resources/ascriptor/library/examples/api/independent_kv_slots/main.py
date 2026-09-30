# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent K and V slot lifetimes across two cube products and one vector cast.

    python main.py                        # every case, functional simulator
    python main.py --list                 # the case ids, with their purpose
    python main.py --case large_k1v2p2    # one of them
    python main.py --launcher pipesim     # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn       # the cce backend, on this machine's card
    python main.py --fault reuse_v_early  # a deliberately broken kernel; it must be caught

For each item and each of five beats: `S = float32(Q @ K.T)`, `P = cast_b16(S)`, `O = float32(P @ V)`.
There is no softmax, no scale, no causal mask and no accumulation between beats. This is a storage
ownership example, not an attention kernel and not a benchmark.

What the example is about is who reads a buffer last:

    K in L1     written by GM-to-L1 MTE2, last read by QK's final L1-to-L0B MTE1 fragment
    V in L1     written by GM-to-L1 MTE2, last read by the *delayed* PV's final MTE1 fragment
    P in UB     written by the vector cast, last read by the MTE3 publication into L1
    P in L1     written by both vector publications, last read by PV's final L1-to-L0A fragment

K retires after QK; V stays alive until the PV that runs a beat later. That difference is the whole
subject, and it is why one slot is enough for K and two are not enough for V under a lookahead
policy. The K load's readiness and availability events are written out explicitly, outside
`auto_sync()`, so that a missing edge stays missing instead of being repaired for us.

A sibling contract may permit a numerically equal K and V sharing one allocation, which adds PV to
that allocation's reader set. Nothing here establishes that: K and V are separately stored *and*
numerically different, and `check_domain` asserts both.

`--fault` runs the three ownership and capacity negatives and checks that each is caught where it is
supposed to be -- including the one the functional simulator cannot see.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec, compile_kernel

from kernel import make_kernel
from reference import CASES as CASE_TABLE
from reference import make_inputs, output_seed, parameters, reference

DEVICE = "a5"
L1_CAPACITY = 524288      # bytes, on the A5 profile

OUTPUTS = ("o",)

# Each fault, and the earliest stage that can see it. `missing_k_available` is the interesting one:
# the functional simulator executes source order and its outputs stay bit-exact, so only the pipe
# model's hazard check finds it. A probe that only ever ran on sim would call that kernel correct.
FAULTS = {
    "missing_k_available": {
        "seen_by": "pipesim",
        "what": "the reverse availability event for the K load is removed, so nothing orders the "
                "next GM-to-L1 write against the previous QK's last read of that slot",
    },
    "reuse_v_early": {
        "seen_by": "outputs",
        "what": "V drops to one slot while the lookahead policy still needs the previous beat's V "
                "for its delayed PV -- premature reuse, not insufficient L1",
    },
    "oversized_k2v2p2": {
        "seen_by": "build",
        "what": "the K2/V2/P2 policy at width 256, whose L1 footprint does not fit the device",
    },
}

CASES_BY_ID = {
    "small_k2v2p2": "Two K slots, two V slots, two P slots at N=D=64 over three items on one core. "
                    "Three items on one core also wrap the slots across item boundaries, which a "
                    "single-item case cannot show",
    "small_k1v2p2": "One K slot with two V and two P. K retires after QK, so one slot is enough for "
                    "it while the delayed PV still needs the previous V -- the pair with the case "
                    "above is what separates the two lifetimes",
    "small_sequential": "One slot each: the current PV is consumed before any next-beat load. The "
                        "control that says the lookahead policies' extra slots are about lookahead "
                        "and not about the arithmetic",
    "bf16_idle": "BF16 on three mixed cores for one item, so two cores execute no beats at all. An "
                 "idle participant must still leave every guard row intact",
    "large_k1v2p2": "N=D=256, where each K/V slot is 131,072 bytes and the lookahead policy's L1 "
                    "footprint is 417,792 of the 524,288 available. The largest geometry that fits",
    "large_sequential": "N=D=256 with one slot each, 278,528 bytes. Its pair with the case above is "
                        "the whole capacity argument, measured rather than asserted",
}

CASES = [{"id": case_id, "seed": 9608, "block_dim": CASE_TABLE[case_id][4],
          "parameters": parameters(case_id), "purpose": purpose}
         for case_id, purpose in CASES_BY_ID.items()]


def check_domain(inputs, expected):
    """Independence of K and V -- the premise of the whole example -- and the guard rows that make an
    unpublished result visible."""
    p = inputs["parameters"]
    if inputs["k"].untyped_storage().data_ptr() == inputs["v"].untyped_storage().data_ptr():
        raise ValueError("K and V must be separately stored; a shared allocation is another contract")
    if torch.equal(inputs["k"], inputs["v"]):
        raise ValueError("K and V must differ numerically, or PV could not tell them apart")
    for name in ("q", "k", "v"):
        if not bool(torch.isfinite(inputs[name]).all()) or bool((inputs[name].abs() > 1).any()):
            raise ValueError(f"{name} must be finite and in [-1, 1]")
    seed = output_seed(p)
    if not torch.equal(expected["o"][:, :, 16:, :], seed[:, :, 16:, :]):
        raise ValueError("the reference must leave every guard row at its label")
    if seed[:, :, 16:, :].unique().numel() != seed[:, :, 16:, :].numel():
        raise ValueError("each guard label must be distinct, or a misdirected store could hide")
    if bool(torch.isnan(expected["o"][:, :, :16, :]).any()):
        raise ValueError("every result row is defined; a NaN there would mean the poison survived")


def execute(case, inputs, launcher, backend, fault=None):
    """One launch per case, at the case's own mixed-core count. The output arrives with NaN over the
    sixteen result rows and a distinct label in each of the sixteen guard rows, seeded in."""
    p = inputs["parameters"]
    entry = make_kernel(p["policy"], width=p["d"], items=p["items"], beats=p["beats"],
                        dtype_name=p["dtype"], fault=fault)
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["q"], inputs["k"], inputs["v"], output_seed(p))}


def compare(name, got, want):
    """Bitwise. The dyadic inputs make the FP32 sums exact, the reference rounds the first result to
    b16 explicitly, and result and guard rows are compared together."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    results, guards = (slice(None, 16), slice(16, None))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} FP32 elements "
          f"({got[:, :, results].numel()} result + {got[:, :, guards].numel()} guard)")
    if not ok:
        wrong = got != want
        print(f"      {int(wrong.sum())} differ: {int(wrong[:, :, results].sum())} result and "
              f"{int(wrong[:, :, guards].sum())} guard elements, max |error| "
              f"{(got - want).abs().max().item()}"
              + (f"; {int(torch.isnan(got[:, :, results]).sum())} result elements are still NaN "
                 f"(never published)" if bool(torch.isnan(got[:, :, results]).any()) else "")
              + ("; the final beat is still exact, which localizes this to a reused slot rather "
                 "than a broken product" if torch.equal(got[:, -1], want[:, -1]) else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--fault", default=None, choices=tuple(FAULTS),
                        help="run one ownership or capacity negative instead of the cases, and "
                             "check it is caught where it should be; never run one on hardware")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            p = case["parameters"]
            print(f"{case['id']:18s} {p['policy']:10s} N=D={p['d']:<4d} {p['items']} item(s) "
                  f"{p['dtype']:9s} block_dim={case['block_dim']}  {case['purpose']}")
        return 0
    if args.fault:
        return fault_probe(args)

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['policy']}, N=D={p['d']}, {p['items']} item(s) x {p['beats']} "
              f"beats, {p['dtype']}, {case['block_dim']} core(s), launcher={args.launcher})")
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


def fault_probe(args):
    """Each negative must be caught by the stage it declares, and by no earlier one.

    That second half matters as much as the first. `missing_k_available` produces bit-exact output on
    the functional simulator, so this probe asserts that it does -- a run that quietly started
    failing there would mean the fault had become something else, and the lesson would be lost.
    """
    fault = args.fault
    seen_by, what = FAULTS[fault]["seen_by"], FAULTS[fault]["what"]
    print(f"--fault {fault}: {what}\n  caught by: {seen_by}")
    if seen_by == "build":
        entry = make_kernel("k2v2p2", width=256, items=1)
        try:
            compile_kernel(entry, backend=args.backend, block_dim=1)
        except Exception as error:
            first = str(error).splitlines()[0]
            print(f"  refused while lowering: {type(error).__name__}: {first}")
            located = all(token in str(error) for token in ("addr_alloc", "kernel.py:"))
            print(f"  the refusal {'names a source location' if located else 'IS NOT LOCATED'} and "
                  f"the footprint through V is 532480 bytes against {L1_CAPACITY} available")
            return 0 if located else 1
        print("  NOT CAUGHT -- the oversized policy lowered, so the capacity check is gone")
        return 1

    case = next(c for c in CASES if c["id"] == "small_k1v2p2")
    inputs = make_inputs(case)
    expected = reference(inputs)
    outcomes = {}
    for launcher in ("sim", "pipesim"):
        try:
            actual = execute(case, inputs, launcher, args.backend, fault)
        except Exception as error:
            print(f"  {launcher:8s} refused: {type(error).__name__}: "
                  f"{str(error).splitlines()[0][:150]}")
            outcomes[launcher] = "refused"
            continue
        ok = compare("o", actual["o"], expected["o"])
        outcomes[launcher] = "exact" if ok else "wrong outputs"
        print(f"  {launcher:8s} {outcomes[launcher]}")
    wanted = {"pipesim": {"sim": "exact", "pipesim": "refused"},
              "outputs": {"sim": "wrong outputs", "pipesim": "wrong outputs"}}[seen_by]
    off = {k: (outcomes[k], v) for k, v in wanted.items() if outcomes[k] != v}
    if off:
        for launcher, (was, should) in off.items():
            print(f"  MISMATCH on {launcher}: {was}, expected {should}")
        return 1
    print("  as declared" + (" -- and the functional model, run alone, would have called this "
                             "kernel correct" if seen_by == "pipesim" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
