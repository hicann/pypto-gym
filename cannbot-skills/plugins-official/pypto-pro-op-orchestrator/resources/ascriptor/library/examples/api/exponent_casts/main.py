# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Move a BF16 exponent field to an 8-bit code and back, with an instruction sequence.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case all_codes_to_bf16 # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

These are synthesized sequences, not native BF16 <-> E8M0 casts -- there is no such instruction
here. Encoding shifts the sign bit off the top and the exponent field into the low byte, narrows
u16 to u8 (one code per even byte), and packs the codes into the low half of the register.
Decoding unpacks them back into even bytes, widens, shifts the code into the BF16 exponent bits,
and patches the one value the shift gets wrong: code 255 would land on +inf, and the operation
this stands in for yields the canonical BF16 NaN, so a compare-and-select replaces it.

Encoding discards the sign and the mantissa, so it is not reversible; the two cases are two
separate conversions, not a round trip.

Both outputs are compared as INT32 -- an 8-bit code or a 16-bit storage word losslessly widened --
so the comparison is over exact carrier and storage bits.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import make_inputs, reference

DEVICE = "a5"

ROWS, COLS = 4, 128
CANONICAL_NAN = 0x7FC0

OUTPUTS = ("o",)

CASES = [
    {"id": "bf16_exponent_bits", "seed": 8781, "block_dim": 1,
     "purpose": "BF16 in, codes out. Four rows: the exact powers of two from 2^-40 to 2^87, 1.5x "
                "each of them, their negatives, and a row whose first six lanes are +0, +inf, "
                "NaN, -0, the smallest normal and a subnormal -- so the sign and mantissa the "
                "extraction is supposed to discard are actually there to be discarded",
     "parameters": {"variant": "encode"}},
    {"id": "all_codes_to_bf16", "seed": 8782, "block_dim": 1,
     "purpose": "Every one of the 256 codes, decoded to a BF16 storage word. Rows 0 and 1 cover "
                "the range once each; row 3 repeats the eight boundary codes (0, 1, 126, 127, "
                "128, 253, 254, 255) sixteen times, which is where code 0 has to give +0 and code "
                "255 the canonical NaN rather than the infinity the shift alone produces",
     "parameters": {"variant": "decode"}},
]


def check_domain(inputs, expected):
    """The two mappings that are decisions rather than arithmetic, asserted on the reference: code
    0 decodes to +0 and code 255 to the canonical BF16 NaN. Both differ from the legacy host
    MX-scale helper, where 0 is 2^-127 and 255 is +inf."""
    x = inputs["x"]
    if tuple(x.shape) != (ROWS, COLS):
        raise ValueError(f"the input must be [{ROWS}, {COLS}]")
    if inputs["variant"] == "encode":
        if x.dtype != torch.bfloat16:
            raise ValueError("the encode case takes bfloat16")
        if not bool((expected["o"] <= 255).all()) or not bool((expected["o"] >= 0).all()):
            raise ValueError("an encoded code is a byte")
    else:
        if x.dtype != torch.uint8:
            raise ValueError("the decode case takes uint8 codes")
        codes = x.int()
        if (codes == 255).any() and not bool((expected["o"][codes == 255] == CANONICAL_NAN).all()):
            raise ValueError("code 255 must decode to the canonical BF16 NaN")
        if (codes == 0).any() and not bool((expected["o"][codes == 0] == 0).all()):
            raise ValueError("code 0 must decode to +0")


def execute(case, inputs, launcher, backend):
    """One launch. The destination is the conversion's own narrow type; the comparison widens it
    afterwards, which is why nothing here is poisoned -- an unwritten byte would be a zero code,
    and the `all_codes_to_bf16` case has a legitimate zero in it."""
    variant = inputs["variant"]
    entry = (kernel.e8m0_from_bf16 if variant == "encode" else kernel.bf16_from_e8m0)
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    dtype = torch.uint8 if variant == "encode" else torch.bfloat16
    produced = op(inputs["x"], torch.zeros(ROWS, COLS, dtype=dtype))
    # Widened losslessly for comparison: a code as an integer, a BF16 word as its stored bits.
    return {"o": produced.int() if variant == "encode" else produced.view(torch.uint16).int()}


def compare(name, got, want):
    """Bitwise over the widened carriers."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} carriers")
    if not ok:
        index = (got != want).nonzero()
        print(f"      {len(index)}/{got.numel()} differ; first "
              + ", ".join(f"[{r},{c}] 0x{got[r, c]:04x} != 0x{want[r, c]:04x}"
                          for r, c in index[:4].tolist()))
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
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({case['parameters']['variant']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
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
