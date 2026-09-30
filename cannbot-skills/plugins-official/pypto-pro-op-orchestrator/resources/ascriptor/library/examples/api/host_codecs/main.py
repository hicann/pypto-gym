# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The library's host codecs, against bit references built from the formats themselves.

    python main.py                      # the codec comparisons, then the transport case
    python main.py --list               # the case ids, with their purpose
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

`ascriptor.dtypes` converts between host tensors and the packed carrier formats: signed int4 eight to
an INT32 word, FP4 two nibbles to a byte low first, E8M0, and HiFloat8. This folder checks those
conversions against tables reference.py builds from each format's definition -- an exponent and
fraction enumeration for HiFloat8, a literal lattice for FP4, integer shifts for int4 -- and imports
no codec on the reference side, which is what makes the comparison mean anything.

Six comparisons run before the cases, each on its own line. All of them are exhaustive over the
one-byte code space, and the HiFloat8 encoder is checked at every interval midpoint and both of its
neighbours, across two source dtypes and all four combinations of its two special-value modes.

The device part is deliberately small: one kernel moves all 256 carriers through UB unchanged. The
codecs are host arithmetic, and nothing here claims otherwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.dtypes import (e8m0_to_fp32, fp4_e1m2_to_fp32, fp4_e2m1_to_fp32, fp16_to_hif8,
                              fp32_to_e8m0, fp32_to_fp4_e1m2, fp32_to_fp4_e2m1, fp32_to_hif8,
                              hif8_to_fp32, pack_signed_int4_to_int32,
                              unpack_int32_to_signed_int4)
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import carrier_transport
from reference import (CARRIERS, INT4_VALUES, hif8_boundary_inputs, hif8_reference_encode,
                       make_inputs, reference, tables)

DEVICE = "a5"

POISON = 77       # a byte the transport never wrote

OUTPUTS = ("o",)
CODECS = ("int4", "fp4_e2m1", "fp4_e1m2", "e8m0", "hif8_decode", "hif8_encode")

CASES = [
    {"id": "every_carrier", "seed": 0, "block_dim": 1, "parameters": {"pattern": "ascending"},
     "purpose": "All 256 byte values through UB in order, so a transport that dropped or duplicated "
                "one is visible as a missing value rather than as a plausible byte"},
    {"id": "reversed_carriers", "seed": 0, "block_dim": 1, "parameters": {"pattern": "descending"},
     "purpose": "The same 256 values in the opposite order: the transport must not depend on them "
                "being sorted"},
]


def check_domain(inputs, expected):
    """A transport's reference is its input, and the input must cover the whole byte space."""
    if not torch.equal(expected["o"], inputs["x"]):
        raise ValueError("the reference for a transport is its input")
    if expected["o"].data_ptr() == inputs["x"].data_ptr():
        raise ValueError("the reference must not alias the input")
    if int(torch.unique(inputs["x"]).numel()) != CARRIERS:
        raise ValueError("the case must carry every byte value exactly once")


def codecs():
    """Six exhaustive comparisons against reference.py's independent tables. Returns what failed."""
    expected, failed = tables(), []
    carriers = torch.arange(CARRIERS, dtype=torch.int32).to(torch.uint8)

    def verdict(name, ok, detail):
        print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  {detail}")
        if not ok:
            failed.append(name)

    original = torch.tensor(INT4_VALUES, dtype=torch.int32)
    packed = pack_signed_int4_to_int32(original)
    verdict("int4", torch.equal(packed, expected["int4"])
            and torch.equal(unpack_int32_to_signed_int4(packed, len(INT4_VALUES)), original),
            f"{len(INT4_VALUES)} values pack into {packed.numel()} INT32 word(s) and unpack back")

    for kind, decoder, encoder in (("e2m1", fp4_e2m1_to_fp32, fp32_to_fp4_e2m1),
                                   ("e1m2", fp4_e1m2_to_fp32, fp32_to_fp4_e1m2)):
        want = expected["fp4_" + kind]
        decoded = decoder(carriers)
        # Both zero signs are preserved, so every physical carrier round-trips exactly.
        verdict("fp4_" + kind,
                torch.equal(decoded.view(torch.int32), want.view(torch.int32))
                and torch.equal(encoder(want), carriers),
                f"{CARRIERS} carriers decode to {want.numel()} values and re-encode, both zero signs")

    verdict("e8m0", torch.equal(e8m0_to_fp32(carriers), expected["e8m0"])
            and torch.equal(fp32_to_e8m0(expected["e8m0"]), carriers),
            f"{CARRIERS} powers of two, byte 255 being infinity in this legacy host helper")

    decoded, want = hif8_to_fp32(carriers), expected["hif8"]
    nan = want.isnan()
    verdict("hif8_decode", torch.equal(decoded.isnan(), nan)
            and torch.equal(decoded[~nan].view(torch.int32), want[~nan].view(torch.int32)),
            f"{CARRIERS} codes, the NaN one matched as NaN because its payload is unspecified")

    boundary = hif8_boundary_inputs()
    mismatched = []
    for saturate in (False, True):
        for nan_to_zero in (False, True):
            for dtype, encoder in ((torch.float32, fp32_to_hif8), (torch.float16, fp16_to_hif8)):
                source = boundary.to(dtype)
                wanted = torch.tensor(hif8_reference_encode(source.float().tolist(),
                                                            saturate=saturate,
                                                            nan_to_zero=nan_to_zero),
                                      dtype=torch.uint8)
                if not torch.equal(encoder(source, saturate=saturate, nan_to_zero=nan_to_zero),
                                   wanted):
                    mismatched.append(f"{dtype}/saturate={saturate}/nan_to_zero={nan_to_zero}")
    verdict("hif8_encode", not mismatched,
            f"{boundary.numel()} interval boundaries x 2 source dtypes x 4 special modes"
            + (f"; failed on {mismatched}" if mismatched else ""))
    return failed


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with 77 and seeded in."""
    op = OpExec(carrier_transport, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full((1, CARRIERS), POISON, dtype=torch.uint8))}


def compare(name, got, want):
    """Bitwise: a transport changes no byte."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:12s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} carrier bytes")
    if not ok:
        index = (got != want).nonzero()
        unwritten = int((got == POISON).sum())
        print(f"      {len(index)}/{got.numel()} bytes differ, first "
              f"{[tuple(i.tolist()) for i in index[:4]]}"
              + (f"; {unwritten} still hold the {POISON} fill (never moved)" if unwritten else ""))
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
            print(f"{case['id']:20s} {case['purpose']}")
        print("\nhost codec comparisons: " + ", ".join(CODECS))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    print("host codecs  (against reference.py's independent tables; nothing runs on the device)")
    failed = list(codecs())
    for case in selected:
        print(f"{case['id']}  ({case['parameters']['pattern']} carriers, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    cases_failed = {f.split("/")[0] for f in failed if "/" in f}
    print(f"\n{len(selected) - len(cases_failed)}/{len(selected)} cases passed, "
          f"{len(CODECS) - len([f for f in failed if '/' not in f])}/{len(CODECS)} "
          f"codec comparisons passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
