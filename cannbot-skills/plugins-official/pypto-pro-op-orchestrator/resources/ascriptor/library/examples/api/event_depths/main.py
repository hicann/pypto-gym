# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A two-, three- or four-slot UB ring with both events written out, and what happens without one.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case depth_4        # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card
    python main.py --missing-event --launcher pipesim   # the same ring with the readiness event
                                                        # removed: pipesim must refuse it

The producer fills a whole batch of UB slots, then the consumer stores them to GM. Two events run
in opposite directions: `ready` (MTE2 to MTE3) publishes a filled slot, and `available`
(MTE3 to MTE2, `preset=True`) starts with one token per slot and keeps the producer from
overwriting a slot whose read has not finished. `depth` selects the buffer and event classes
together -- `DBuff`/`DEvent`, `TBuff`/`TEvent`, `QBuff`/`QEvent` -- and fixes the row count at
`2 * depth + 1`, so the ring wraps twice and the last batch holds a single row.

Row `i` of the input is centred on `10 * i`, so a slot reused too early does not produce a subtly
wrong value: it produces a row from the wrong batch, an order of magnitude away. The comparison is
bitwise, and the destination arrives NaN-poisoned, so a row nothing stored is visible as well.

`--missing-event` is the negative control, and it says something about the launchers as much as
about the kernel: with `ready` removed, pipesim reports a real UB hazard, while `sim` -- which
interprets the kernel sequentially and models no pipes at all -- still passes every case.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_ring
from reference import make_inputs, reference

DEVICE = "a5"

OUTPUTS = ("o",)

CASES = [
    {"id": "depth_2", "seed": 8882, "block_dim": 1,
     "purpose": "Two slots over five rows: the shallowest ring, where every slot is reused twice "
                "and the producer is one row ahead of the consumer for most of the run",
     "parameters": {"depth": 2, "iterations": 5}},
    {"id": "depth_3", "seed": 8883, "block_dim": 1,
     "purpose": "Three slots over seven rows: TBuff/TEvent, and the batch that wraps the ring "
                "twice with one row left over",
     "parameters": {"depth": 3, "iterations": 7}},
    {"id": "depth_4", "seed": 8884, "block_dim": 1,
     "purpose": "Four slots over nine rows: the deepest supported ring, where four MTE2 "
                "transfers are outstanding before the first MTE3 read starts",
     "parameters": {"depth": 4, "iterations": 9}},
]


def check_domain(inputs, expected):
    """What makes a wrong slot reuse legible rather than merely wrong: consecutive rows are an
    order of magnitude apart, so no two rows of the input could be confused for each other."""
    x = inputs["x"]
    iterations = 2 * inputs["depth"] + 1
    if x.shape != (iterations, 64) or x.dtype != torch.float32:
        raise ValueError(f"depth {inputs['depth']} declares a float32[{iterations}, 64] input")
    means = x.mean(dim=1)
    if not bool((means.diff() > 5).all()):
        raise ValueError("consecutive input rows must be far apart, or a wrong slot reuse would "
                         "not be legible in the output")
    if not torch.equal(expected["o"], x):
        raise ValueError("the reference is the input unchanged; the kernel only moves it")


def execute(case, inputs, launcher, backend, synchronize=True):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so a row no
    MTE3 store reached reads back as NaN."""
    entry = make_ring(inputs["depth"], synchronize=synchronize)
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full_like(inputs["x"], float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: every output lane is a copied input lane."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]} rows")
    if not ok:
        rows = raw(got).view(got.shape[0], -1).ne(raw(want).view(got.shape[0], -1)).any(dim=1)
        index = rows.nonzero().flatten().tolist()
        # Rows are ten apart in magnitude, so the mean of a wrong row names the batch it came from.
        arrived = [round(got[r].mean().item() / 10) for r in index[:4]]
        poisoned = [r for r in index if bool(torch.isnan(got[r]).all())]
        print(f"      rows {index} differ; those rows now hold data centred on rows {arrived}"
              + (f"; rows {poisoned[:4]} are still NaN-poisoned (never stored)" if poisoned else ""))
    return ok


def probe_missing_event(selected, launcher, backend):
    """Run each case with the readiness event removed and report what the launcher makes of it.

    pipesim must refuse every case: the consumer's MTE3 read of a slot is no longer ordered after
    the producer's MTE2 write of it, which is a real hazard on overlapping UB bytes. `sim` accepts
    them all, because it interprets the kernel sequentially and models no pipes -- so this is also
    the shortest demonstration of what a green `sim` run does not cover.
    """
    print(f"the readiness event removed, on {launcher}: pipesim must refuse every case\n")
    refused = 0
    for case in selected:
        inputs = make_inputs(case)
        try:
            execute(case, inputs, launcher, backend, synchronize=False)
            print(f"  {case['id']:8s} accepted -- no hazard reported")
        except Exception as error:
            refused += 1
            reason = str(error).splitlines()[0]
            print(f"  {case['id']:8s} refused   {reason[:140]}")
    print(f"\n{refused}/{len(selected)} cases refused")
    if launcher == "pipesim":
        return 0 if refused == len(selected) else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    parser.add_argument("--missing-event", action="store_true",
                        help="remove the MTE2-to-MTE3 readiness event; pipesim must then refuse")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:9s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    if args.missing_event:
        return probe_missing_event(selected, args.launcher, args.backend)
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (depth={p['depth']}, rows={p['iterations']}, "
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
