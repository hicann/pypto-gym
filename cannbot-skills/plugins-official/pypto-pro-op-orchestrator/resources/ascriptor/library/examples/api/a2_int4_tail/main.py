# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Signed-INT4 cube products where the carrier is longer than the logical K.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case source137_pad15   # one of them
    python main.py --device a3              # the same source against the A3 facade
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

The public tensors are INT32 carriers holding eight low-first signed nibbles each, and the logical K
is a separate scalar argument -- no `reinterpret` packs anything. So `K=9` arrives as two carriers
with fifteen nibbles of padding, and the padding is generated data rather than zeros.

That matters because C220's INT4 MMAD consumes **both** nibbles of the final byte even when the
logical K is odd (M10-099). One cube launch therefore normalizes on the device before any MMAD: it
copies each carrier into a private GM scratch argument, clears the unused high nibbles of the last
word with scalar stores, and cleans the scalar data cache before MTE2. The caller's tensors are
untouched.

`source137` and `source137_pad15` are the same logical input with different padding nibbles, and
their outputs must be bit-identical. That pair is the only check here that the normalization works.

Chunking has two branches: a K chunk of more than 64 logical values uses the main 16-carrier L1
allocation, and one of at most 64 uses a separate 8-carrier one. M and N tile at 64, and only the
valid rows and columns are published.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernel
from reference import make_inputs, reference, unpack

DEVICES = ("a2", "a3")

POISON = -777
NIBBLES_PER_CARRIER = 8
SHORT_TAIL = 64      # a K chunk at or below this uses the separate eight-carrier allocation

OUTPUTS = ("o",)

CASES = [
    {"id": "k1", "seed": 103800, "block_dim": 1,
     "purpose": "K=1: one live nibble in an eight-nibble carrier, so seven eighths of the word is "
                "padding the product has to ignore -- the extreme case for the normalization",
     "parameters": {"M": 1, "N": 1, "K": 1, "KC": 1, "padding": 0, "pattern": "random"}},
    {"id": "k9", "seed": 103801, "block_dim": 1,
     "purpose": "K=9 with padding nibbles of 15, which decode to -1: two carriers, the second "
                "holding one live nibble and seven that must not contribute",
     "parameters": {"M": 17, "N": 19, "K": 9, "KC": 2, "padding": 15, "pattern": "random"}},
    {"id": "k33", "seed": 103802, "block_dim": 1,
     "purpose": "K=33 with padding 7: five carriers, and 33 is one past a 32-value boundary",
     "parameters": {"M": 17, "N": 19, "K": 33, "KC": 5, "padding": 7, "pattern": "random"}},
    {"id": "k64", "seed": 103803, "block_dim": 1,
     "purpose": "K=64 with all sixteen signed nibble values present in the payload: exactly eight "
                "carriers, no padding at all, and the boundary between the short-tail allocation "
                "and the main one",
     "parameters": {"M": 16, "N": 32, "K": 64, "KC": 8, "padding": 0, "pattern": "all_nibbles"}},
    {"id": "k65", "seed": 103804, "block_dim": 1,
     "purpose": "K=65: one logical value past the short-tail branch, so nine carriers go through "
                "the main 16-carrier allocation",
     "parameters": {"M": 17, "N": 19, "K": 65, "KC": 9, "padding": 15, "pattern": "random"}},
    {"id": "k128", "seed": 103805, "block_dim": 1,
     "purpose": "K=128 at 64x64: exactly one main K chunk and one M/N tile, with no tail anywhere",
     "parameters": {"M": 64, "N": 64, "K": 128, "KC": 16, "padding": 0, "pattern": "random"}},
    {"id": "source137", "seed": 103806, "block_dim": 1,
     "purpose": "The original 17x19x137 shape: two K chunks, the second a nine-value tail",
     "parameters": {"M": 17, "N": 19, "K": 137, "KC": 18, "padding": 0, "pattern": "random"}},
    {"id": "source129", "seed": 103807, "block_dim": 1,
     "purpose": "The original 65x70x129 shape with padding 15: two M tiles and two N tiles, so the "
                "valid-extent slicing runs in both dimensions at once",
     "parameters": {"M": 65, "N": 70, "K": 129, "KC": 17, "padding": 15, "pattern": "random"}},
    {"id": "three_k_tiles", "seed": 103808, "block_dim": 1,
     "purpose": "129x65x257: three M tiles and three K chunks, the largest geometry here, where the "
                "operand and accumulator double buffers each rotate several times",
     "parameters": {"M": 129, "N": 65, "K": 257, "KC": 33, "padding": 7, "pattern": "random"}},
    {"id": "source137_pad15", "seed": 103806, "block_dim": 1,
     "purpose": "The same logical input as source137 with 15 in the final carrier's padding "
                "nibbles instead of 0. The output must be bit-identical, and this pair is the only "
                "check in the matrix that the device-side padding normalization (M10-099) works",
     "parameters": {"M": 17, "N": 19, "K": 137, "KC": 18, "padding": 15, "pattern": "random"}},
]


def check_domain(inputs, expected):
    """The carrier contract, and the property the padding pair depends on: the reference decodes only
    the logical prefix, so it is independent of whatever the padding nibbles hold."""
    k = inputs["K"]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 257:
        raise ValueError("the logical K is an integer in [1, 257]")
    for name in ("x", "y"):
        carriers = inputs[name]
        if carriers.dtype != torch.int32 or carriers.ndim != 2 or not carriers.is_contiguous():
            raise ValueError(f"{name} must be contiguous INT32 carrier rows")
        if carriers.shape[1] != -(-k // NIBBLES_PER_CARRIER):
            raise ValueError(f"{name} must hold exactly ceil(K/{NIBBLES_PER_CARRIER}) carriers")
        if not 1 <= carriers.shape[0] <= 129:
            raise ValueError("M and N are in [1, 129]")
    if not torch.equal(expected["o"],
                       (unpack(inputs["x"], k) @ unpack(inputs["y"], k).T).int()):
        raise ValueError("the reference must be the logical-prefix product and nothing else")
    if (expected["o"] == POISON).any():
        raise ValueError("the reference contains the poison value")


def execute(case, inputs, launcher, backend, device):
    """One launch. Two scratch arguments of the carriers' own shapes let the kernel normalize the
    padding on the device without touching the caller's tensors, and all three destinations arrive
    filled with -777 and seeded in."""
    x, y, k = inputs["x"], inputs["y"], inputs["K"]
    m, n, kc = x.shape[0], y.shape[0], x.shape[1]
    op = OpExec(make_kernel(device), launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    before = (x.clone(), y.clone())
    produced = op(x, y, torch.full((m, n), POISON, dtype=torch.int32),
                  torch.full_like(x, POISON), torch.full_like(y, POISON), m, n, k, kc)
    if not torch.equal(x, before[0]) or not torch.equal(y, before[1]):
        raise ValueError("the normalization must not modify the caller's carriers")
    return {"o": produced}


def compare(name, got, want):
    """Bitwise: signed nibble decode and an INT64 logical product give exact non-overflowing INT32."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]}x{got.shape[1]} "
          f"INT32 elements")
    if not ok:
        outside = got != want
        index = outside.nonzero()
        unwritten = int((got[outside] == POISON).sum())
        # Padding that contributed adds a constant per element pair, so the deltas cluster.
        deltas = sorted({int(v) for v in (got - want)[outside][:8]})
        print(f"      {int(outside.sum())}/{got.numel()} elements differ, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}, deltas {deltas[:5]}"
              + (f"; {unwritten} still hold the {POISON} fill (never written)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:17s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['M']}x{p['N']}, K={p['K']} in {p['KC']} carriers, "
              f"padding={p['padding']}, device={args.device}, launcher={args.launcher})")
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
