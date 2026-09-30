# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast layouts: UINT2 and FP4 packing, dual FP8 encoding, and HiFloat8 both ways.

    python main.py                            # every case, functional simulator
    python main.py --list                     # the case ids, with their purpose
    python main.py --case f16_hif8_ta_all_bits  # one of them
    python main.py --mode uint2_unpack        # every case of one mode
    python main.py --launcher pipesim         # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn           # the cce backend, on this machine's card

Ten entries, one per cast, over eighteen cases. Every output is compared against a reference built
from the format's bit layout -- a code table and shift arithmetic -- and not from any library codec,
so agreement is evidence about the instruction rather than about the library agreeing with itself.

What each family is for:

  uint2_pack / uint2_unpack   four 2-bit values per carrier byte. The unpack entry takes an explicit
                              padding row, and three cases give it 0, 165 and random bytes: the
                              decoded result must be identical in all three, which running them
                              together is what shows.
  fp4_pack                    two E1M2 codes per byte, low nibble first. `fp4_all_finite_bf16`
                              enumerates every finite BF16 bit pattern -- 65024 of them.
  fp8_dual                    one source row encoded to E5M2 *and* E4M3 in a single launch, so the
                              two lattices are compared against the same values.
  hif8_to_f32 / hif8_to_f16   all 256 HiFloat8 carriers decoded, including its infinities and its
                              one NaN code.
  f32_to_hif8_* / f16_to_hif8_*  encoding with the two rounding policies, `ta` and `hybrid`. The two
                              `all_bits` cases enumerate all 65536 half-precision words.

The comparison is byte equality, with one exemption: when both sides are NaN they agree whatever
their payloads are, because a NaN payload is not specified by the format. HiFloat8 has a single zero
code, so the sign of zero never arises here -- and everywhere else a flipped zero sign would be a
different byte and a failure.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel.fp4 import bf16_to_fp4_e1m2_kernel
from kernel.fp8 import micro_cast_fp8_pack4_dual_kernel
from kernel.hif8 import (float_to_hif8_carrier_kernel, float_to_hif8_hybrid_carrier_kernel,
                         half_to_hif8_carrier_kernel, half_to_hif8_hybrid_carrier_kernel,
                         hif8_carrier_to_float_kernel, hif8_carrier_to_half_kernel)
from kernel.uint2_decode import uint2_to_bf16_kernel
from kernel.uint2_encode import bf16_to_uint2_kernel
from reference import MODES, make_inputs, reference

DEVICE = "a5"

BYTE_POISON = 0xA5        # for a carrier destination; 0xA5 is not a code any case produces
VALUE_POISON = -777       # for a decoded destination

OUTPUTS = ("o",)

ENTRIES = {
    "uint2_pack": bf16_to_uint2_kernel,
    "uint2_unpack": uint2_to_bf16_kernel,
    "fp4_pack": bf16_to_fp4_e1m2_kernel,
    "fp8_dual": micro_cast_fp8_pack4_dual_kernel,
    "hif8_to_f32": hif8_carrier_to_float_kernel,
    "hif8_to_f16": hif8_carrier_to_half_kernel,
    "f32_to_hif8_ta": float_to_hif8_carrier_kernel,
    "f32_to_hif8_hybrid": float_to_hif8_hybrid_carrier_kernel,
    "f16_to_hif8_ta": half_to_hif8_carrier_kernel,
    "f16_to_hif8_hybrid": half_to_hif8_hybrid_carrier_kernel,
}
# The three cases that must agree with each other: same carriers, different padding bytes.
PADDING_GROUP = ("uint2_unpack_zero", "uint2_unpack_canary", "uint2_unpack_varying")

CASES = [
    {"id": "uint2_all_carriers", "seed": 103300, "block_dim": 1,
     "parameters": {"mode": "uint2_pack", "dataset": "all_carriers", "rows": 8},
     "purpose": "Eight rows whose carriers walk all 256 byte values, so every one of the four "
                "2-bit positions is packed from every pattern"},
    {"id": "uint2_one_row", "seed": 103301, "block_dim": 1,
     "parameters": {"mode": "uint2_pack", "dataset": "random", "rows": 1},
     "purpose": "One row: the minimum, where a kernel that assumed more than one row fails"},
    {"id": "uint2_source_rows", "seed": 103302, "block_dim": 1,
     "parameters": {"mode": "uint2_pack", "dataset": "random", "rows": 64},
     "purpose": "The original 64-row composition on random values"},
    {"id": "uint2_unpack_zero", "seed": 103303, "block_dim": 1,
     "parameters": {"mode": "uint2_unpack", "dataset": "all_carriers", "rows": 8, "padding": 0},
     "purpose": "Unpacking with an all-zero padding row. One of the three that must agree"},
    {"id": "uint2_unpack_canary", "seed": 103304, "block_dim": 1,
     "parameters": {"mode": "uint2_unpack", "dataset": "all_carriers", "rows": 8, "padding": 165},
     "purpose": "The same carriers with 165 (0xA5) in every padding byte -- a value whose 2-bit "
                "fields are all non-zero, so padding that leaked into the result is unmistakable"},
    {"id": "uint2_unpack_varying", "seed": 103305, "block_dim": 1,
     "parameters": {"mode": "uint2_unpack", "dataset": "all_carriers", "rows": 8, "padding": -1},
     "purpose": "The same carriers with random padding bytes, which no single constant could be "
                "confused with"},
    {"id": "fp4_source_values", "seed": 103306, "block_dim": 1,
     "parameters": {"mode": "fp4_pack", "dataset": "source", "rows": 64},
     "purpose": "The 32 representable E1M2 values and the two saturating ones, repeated over 64 "
                "rows: every code, in both nibbles"},
    {"id": "fp4_all_finite_bf16", "seed": 103307, "block_dim": 1,
     "parameters": {"mode": "fp4_pack", "dataset": "finite_bits", "rows": 510},
     "purpose": "Every finite BF16 bit pattern -- 65024 of them -- so the rounding and saturation "
                "boundaries are enumerated rather than sampled"},
    {"id": "fp8_three_rows", "seed": 103308, "block_dim": 1,
     "parameters": {"mode": "fp8_dual", "dataset": "source", "rows": 3},
     "purpose": "Three rows of small dyadic values through both FP8 encodings at once"},
    {"id": "fp8_boundaries", "seed": 103309, "block_dim": 1,
     "parameters": {"mode": "fp8_dual", "dataset": "boundaries", "rows": 35},
     "purpose": "Both lattices' representable values, their midpoints, the neighbours of each, and "
                "the infinities and NaN. This is where a rounding rule that ties the wrong way, or "
                "saturates instead of overflowing, shows up"},
    {"id": "hif8_decode_f32", "seed": 103310, "block_dim": 1,
     "parameters": {"mode": "hif8_to_f32", "dataset": "all_carriers", "total": 256},
     "purpose": "All 256 HiFloat8 carriers decoded to FP32, including both infinities and the one "
                "NaN code -- the complete decode table, not a sample of it"},
    {"id": "f32_hif8_ta_boundaries", "seed": 103311, "block_dim": 1,
     "parameters": {"mode": "f32_to_hif8_ta", "dataset": "boundaries", "total": 1920},
     "purpose": "FP32 encoded with the ta rounding policy at every boundary word"},
    {"id": "f32_hif8_hybrid_boundaries", "seed": 103312, "block_dim": 1,
     "parameters": {"mode": "f32_to_hif8_hybrid", "dataset": "boundaries", "total": 1920},
     "purpose": "The same words with the hybrid policy. The pair is the point: one source, two "
                "policies, and the codes differ where the policies do"},
    {"id": "hif8_decode_f16", "seed": 103313, "block_dim": 1,
     "parameters": {"mode": "hif8_to_f16", "dataset": "all_carriers", "total": 256},
     "purpose": "The same 256 carriers decoded to FP16, where the narrower destination cannot hold "
                "every decoded magnitude"},
    {"id": "f16_hif8_ta_boundaries", "seed": 103314, "block_dim": 1,
     "parameters": {"mode": "f16_to_hif8_ta", "dataset": "boundaries", "total": 512},
     "purpose": "FP16 encoded with ta rounding at every boundary word"},
    {"id": "f16_hif8_ta_all_bits", "seed": 103315, "block_dim": 1,
     "parameters": {"mode": "f16_to_hif8_ta", "dataset": "all_half_bits", "total": 65536},
     "purpose": "Every one of the 65536 half-precision words through ta rounding: exhaustive, so "
                "no boundary can be missed by construction"},
    {"id": "f16_hif8_hybrid_boundaries", "seed": 103316, "block_dim": 1,
     "parameters": {"mode": "f16_to_hif8_hybrid", "dataset": "boundaries", "total": 512},
     "purpose": "The hybrid policy at the same boundary words"},
    {"id": "f16_hif8_hybrid_all_bits", "seed": 103317, "block_dim": 1,
     "parameters": {"mode": "f16_to_hif8_hybrid", "dataset": "all_half_bits", "total": 65536},
     "purpose": "And exhaustively, which together with the case above enumerates the whole "
                "difference between the two policies"},
]


def check_domain(inputs, expected):
    """What each mode's reference must look like for its comparison to mean anything."""
    mode, o = inputs["mode"], expected["o"]
    if mode not in MODES or o.numel() == 0:
        raise ValueError("every case declares a known mode and a non-empty reference")
    if o.dtype == torch.uint8 and bool((o == BYTE_POISON).all()):
        raise ValueError("the whole reference is the poison byte, so nothing could be observed")
    if mode == "uint2_unpack":
        # Four values per carrier byte and nothing else; the padding row must not reach the result.
        if set(o.flatten().tolist()) - {0.0, 1.0, 2.0, 3.0}:
            raise ValueError("an unpacked UINT2 value is 0, 1, 2 or 3")
        if not torch.equal(o, reference({"mode": mode, "x": inputs["x"],
                                         "padding": torch.zeros_like(inputs["padding"])})["o"]):
            raise ValueError("the reference must not depend on the padding row")
    if mode.startswith("hif8_to"):
        if inputs["x"].numel() != 256 or not torch.equal(
                inputs["x"], torch.arange(256, dtype=torch.int32).to(torch.uint8)):
            raise ValueError("a decode case covers all 256 carriers exactly once")
        if int(torch.isnan(o).sum()) != 1 or int(torch.isinf(o).sum()) != 2:
            raise ValueError("HiFloat8 decodes to exactly one NaN and two infinities")
    if mode.endswith(("_ta", "_hybrid")) and o.dtype != torch.uint8:
        raise ValueError("an encode case's reference is the carrier byte")


def execute(case, inputs, launcher, backend):
    """One launch. A carrier destination arrives filled with 0xA5 and a decoded one with -777, both
    seeded in, so an element the cast never wrote is a value the format does not produce."""
    mode, x = inputs["mode"], inputs["x"]
    op = OpExec(ENTRIES[mode], launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    if mode in ("uint2_pack", "fp4_pack"):
        columns = 32 if mode == "uint2_pack" else 64
        produced = op(x, torch.full((x.shape[0], columns), BYTE_POISON, dtype=torch.uint8),
                      x.shape[0])
    elif mode == "uint2_unpack":
        produced = op(x, inputs["padding"],
                      torch.full((x.shape[0], 128), VALUE_POISON, dtype=torch.bfloat16), x.shape[0])
    elif mode == "fp8_dual":
        e5 = torch.full_like(x, BYTE_POISON, dtype=torch.uint8).view(torch.float8_e5m2)
        e4 = torch.full_like(x, BYTE_POISON, dtype=torch.uint8).view(torch.float8_e4m3fn)
        # One launch writes both lattices, so the two encodings see the same source values.
        produced = torch.stack([value.view(torch.uint8) for value in op(x, e5, e4, x.shape[0])])
    elif mode.startswith("hif8_to"):
        dtype = torch.float32 if mode.endswith("f32") else torch.float16
        produced = op(x, torch.full(x.shape, VALUE_POISON, dtype=dtype), x.numel())
    else:
        produced = op(x, torch.full(x.shape, BYTE_POISON, dtype=torch.uint8), x.numel())
    return {"o": produced}


def agree(got, want):
    """Byte equality per element, except that two NaNs agree whatever their payloads hold."""
    if got.dtype != want.dtype or got.shape != want.shape:
        return None
    raw = [t.cpu().contiguous().reshape(-1).view(torch.uint8).reshape(got.numel(), -1)
           for t in (got, want)]
    same = (raw[0] == raw[1]).all(-1)
    if got.is_floating_point():
        both_nan = (torch.isnan(got.cpu()) & torch.isnan(want.cpu())).reshape(-1)
        same = same | both_nan
    return same


def compare(name, got, want):
    """The declared rule: exact, with NaN matching NaN and nothing else loosened."""
    same = agree(got, want)
    if same is None:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    ok = bool(same.all())
    exempt = (int((torch.isnan(got.cpu()) & torch.isnan(want.cpu())).sum())
              if got.is_floating_point() else 0)
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  exact over {got.numel()} "
          f"{str(got.dtype).removeprefix('torch.')} elements"
          + (f", {exempt} NaN pair(s) matched by value" if exempt else ""))
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        index = (~same).nonzero().flatten().tolist()
        # The fill is cast to the output's own dtype: -777 is not representable in BF16, and
        # comparing against the Python value would report nothing was left unwritten.
        fill = torch.tensor(BYTE_POISON if got.dtype == torch.uint8 else VALUE_POISON,
                            dtype=got.dtype)
        unwritten = int((flat_got[index] == fill).sum())
        def show(t, i):
            return hex(int(t[i])) if t.dtype == torch.uint8 else f"{float(t[i]):g}"
        print(f"      {len(index)}/{got.numel()} differ, first at {index[:6]}: got "
              f"{[show(flat_got, i) for i in index[:4]]} want "
              f"{[show(flat_want, i) for i in index[:4]]}"
              + (f"; {unwritten} still hold the fill (never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--mode", default=None, choices=sorted(ENTRIES),
                        help="run every case of one mode")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:28s} {case['parameters']['mode']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])
                and args.mode in (None, case["parameters"]["mode"])]
    if not selected:
        parser.error(f"nothing selected by --case {args.case} --mode {args.mode}; --list prints them")
    failed, produced = [], {}
    for case in selected:
        p = case["parameters"]
        size = p.get("rows") or p.get("total")
        print(f"{case['id']}  ({p['mode']}, {p['dataset']}, {size} "
              f"{'row(s)' if 'rows' in p else 'element(s)'}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        produced[case["id"]] = actual["o"]
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    if all(case_id in produced for case_id in PADDING_GROUP):
        # No single case can say this: the carriers are the same and only the padding differs.
        base = produced[PADDING_GROUP[0]]
        disagreed = [c for c in PADDING_GROUP[1:] if not torch.equal(produced[c], base)]
        print(f"\npadding row: {len(PADDING_GROUP) - len(disagreed)}/{len(PADDING_GROUP)} unpack "
              f"cases decode identically whatever the padding bytes hold")
        failed += [f"{c}/o (differs from {PADDING_GROUP[0]})" for c in disagreed]
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
