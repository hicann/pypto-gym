# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fourteen indexed movements: typed element gathers and scatters, and byte-offset block gathers.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case unused_index_high # one of them
    python main.py --only gatherb_b32       # one of the fourteen movements
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

Six typed element gathers, five scatters and three GatherB widths, each its own kernel and its own
named output. Every result is compared as its 256 raw payload bytes, so a signed carrier, an
unsigned high bit and a float bit pattern are all preserved -- no arithmetic cast stands between the
device and the comparison, on either side.

Three things about the index registers are the point of the example:

  a byte gather zero-extends. An INT8 source byte lands in an INT16 destination with a zero high
  byte, not a sign extension. The unsigned and signed byte cases sit side by side for that reason.
  a byte scatter reads source byte `2 * k` for index lane k. The lanes are 16-bit, so the source
  bytes it consumes are the even ones -- a dense byte read is a different program.
  GatherB indices are byte offsets, not element counts. Eight aligned 32-byte blocks are selected
  out of a 512-byte source, and the index names each by its byte address.

Only the defined index lanes are consumed: thirty-two for 64-bit data behind a UINT32 index
register, eight for GatherB. `unused_index_zero` and `unused_index_high` differ only in what the
*undefined* lanes hold -- 0 against 2147483647 -- and running both together checks that every one of
the fourteen outputs is byte-identical across the pair.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import blocks, elements
from reference import PLANS, make_inputs, reference

DEVICE = "a5"

POISON = 0xA5             # every destination byte; 0xA5 is not produced by any payload here
PAYLOAD_BYTES = 256       # every one of the fourteen results is this many raw bytes

OUTPUTS = tuple(plan[0] for plan in PLANS)
# The pair whose outputs must agree: same active indices and payloads, different ignored lanes.
IGNORED_LANE_PAIR = ("unused_index_zero", "unused_index_high")

CASES = [
    {"id": "reverse", "seed": 103900, "block_dim": 1,
     "parameters": {"dataset": "reverse", "padding": 0},
     "purpose": "Indices in descending order, so every lane moves and an implementation that "
                "copied straight through disagrees everywhere rather than at one lane"},
    {"id": "random", "seed": 103901, "block_dim": 1,
     "parameters": {"dataset": "random", "padding": 0},
     "purpose": "A random permutation of the legal indices -- the ordinary case, with no structure "
                "for a wrong address formula to accidentally satisfy"},
    {"id": "repeated_gather", "seed": 103902, "block_dim": 1,
     "parameters": {"dataset": "repeat_gather", "padding": 0},
     "purpose": "Alternating first and last index: a gather may read one source element many "
                "times. Scatter keeps its unique destinations, because duplicate writers have no "
                "declared order here"},
    {"id": "unused_index_zero", "seed": 103903, "block_dim": 1,
     "parameters": {"dataset": "random", "padding": 0},
     "purpose": "The undefined index lanes hold zero. Half of the pair that shows which lanes are "
                "actually consumed"},
    {"id": "unused_index_high", "seed": 103903, "block_dim": 1,
     "parameters": {"dataset": "random", "padding": 2147483647},
     "purpose": "The same active indices and payloads with 2147483647 in the undefined lanes. "
                "Every output must be byte-identical to the case above; a kernel that consumed a "
                "lane it should not have would read far out of range"},
]


def check_domain(inputs, expected):
    """The typed rows each movement declares, and the two properties of the reference that a plain
    'it matches' would not state: a byte gather zero-extends, and an unselected scatter byte is
    zero rather than whatever was there."""
    for name, src_type, src_count, idx_type, idx_count, _, _, width, active, op in PLANS:
        item = inputs["buffers"][name]
        for key, dtype, count in (("src", src_type, src_count), ("idx", idx_type, idx_count)):
            value = item[key]
            if value.dtype != getattr(torch, dtype) or tuple(value.shape) != (1, count):
                raise ValueError(f"{name}.{key} must be one complete {dtype} row of {count}")
        if expected[name].dtype != torch.uint8 or expected[name].numel() != PAYLOAD_BYTES:
            raise ValueError(f"{name} must be compared as {PAYLOAD_BYTES} raw payload bytes")
        if (expected[name] == POISON).all():
            raise ValueError(f"{name}'s reference is entirely the poison byte")
        if op == "gather" and width == 1:
            # The odd byte of each destination lane is the extension byte, and it is zero.
            if not bool((expected[name].flatten()[1::2] == 0).all()):
                raise ValueError("a byte gather zero-extends; a sign extension would fill these")
        if op == "scatter":
            written = set()
            for index in item["idx"].flatten().tolist()[:active]:
                written.update(range(index * width, (index + 1) * width))
            holes = [b for b in range(PAYLOAD_BYTES) if b not in written]
            if holes and not bool((expected[name].flatten()[holes] == 0).all()):
                raise ValueError("an unselected scatter byte is zero, not left at what was there")
            if len(written) != active * width:
                raise ValueError("scatter destinations must be unique; duplicates have no order")


def execute(case, inputs, launcher, backend, only):
    """One launch per movement. Each destination arrives filled with 0xA5 and seeded in, so a byte
    the kernel never wrote is 0xA5 -- a value no payload in this example produces."""
    outputs = {}
    for name, _, _, _, _, dst_type, _, _, _, operation in PLANS:
        if only not in (None, name):
            continue
        entry = getattr(blocks if operation == "blocks" else elements, f"{name}_kernel")
        op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}/{name}",
                    seed_outputs=True)
        poison = torch.full((1, PAYLOAD_BYTES), POISON, dtype=torch.uint8).view(
            getattr(torch, dst_type))
        produced = op(inputs["buffers"][name]["src"], inputs["buffers"][name]["idx"], poison, 1)
        outputs[name] = produced.view(torch.uint8).reshape(1, PAYLOAD_BYTES)
    return outputs


def compare(name, got, want):
    """Bitwise over the raw payload. Nothing here is arithmetic, so nothing here has a tolerance."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:22s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:22s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} payload bytes")
    if not ok:
        differ = (got != want).flatten()
        index = differ.nonzero().flatten().tolist()
        unwritten = int((got.flatten()[differ] == POISON).sum())
        print(f"      {len(index)}/{got.numel()} bytes differ, first at {index[:6]}; got "
              f"{got.flatten()[index[:4]].tolist()} against {want.flatten()[index[:4]].tolist()}"
              + (f"; {unwritten} are still the 0x{POISON:02X} fill (never written)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--only", default=None, choices=OUTPUTS,
                        help="run one of the fourteen movements instead of all of them")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:18s} {case['purpose']}")
        print("\nmovements: " + ", ".join(OUTPUTS))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    names = [name for name in OUTPUTS if args.only in (None, name)]
    failed, produced = [], {}
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['dataset']} indices, ignored lanes hold {p['padding']}, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.only)
        produced[case["id"]] = actual
        for name in names:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    if all(case_id in produced for case_id in IGNORED_LANE_PAIR):
        # Neither case alone can say this: the undefined lanes differ and nothing else does.
        left, right = (produced[case_id] for case_id in IGNORED_LANE_PAIR)
        disagreed = [name for name in names if not torch.equal(left[name], right[name])]
        print(f"\nignored index lanes: {len(names) - len(disagreed)}/{len(names)} movements are "
              f"byte-identical across {' and '.join(IGNORED_LANE_PAIR)}")
        failed += [f"{IGNORED_LANE_PAIR[1]}/{name} (differs from {IGNORED_LANE_PAIR[0]})"
                   for name in disagreed]
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
