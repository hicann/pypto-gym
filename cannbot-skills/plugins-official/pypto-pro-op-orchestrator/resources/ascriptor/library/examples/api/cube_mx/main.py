# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MX products with nonuniform per-group scales, through both the shortcut and the explicit path.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case fp4_mixed_explicit  # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

Three payload formats, each computed two ways over the same data:

  shortcut   `matmul_mx(product, a, b, scale_a, scale_b, m=, n=, k=)`
  explicit   `l1_to_l0_mx` for each operand, then `mmad_mx`

The pair must agree. `DT.mx_e4m3` and `DT.mx_e5m2` are compatibility aliases for the ordinary FP8
storage types and **do not attach scales**: the MX semantics come from the `l1_to_l0_mx` and
`mmad_mx` / `matmul_mx` calls, not from the operand's dtype. That is the easiest thing here to get
backwards.

The scales are dense per row and per K32 group -- 16 rows, four groups over K=128 -- and deliberately
nonuniform in both directions, so a wrong packing or a wrong group selection changes the answer.
`gm_to_l1_mx_scale_nd2nz` converts the dense form into packed MX storage.

The reference decodes the payloads itself: arithmetic for FP8 (exponent, fraction, bias, subnormals)
and a table for FP4, importing no ascriptor codec. Restricted exponent bands keep the whole
128-element dyadic dot product exact in FP32, so the comparison is bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_mx
from reference import make_inputs, reference

DEVICE = "a5"

TILE = (16, 16)
GROUPS, GROUP_K = 4, 32   # four K32 scale groups per row, so K = 128

OUTPUTS = ("o",)

FORMATS = {
    "e4m3": (128, "E4M3 payloads, one byte per K value, from a restricted exponent band"),
    "e5m2": (128, "E5M2 payloads: the same shape at a different exponent/mantissa split, so a "
                  "decoder that hard-coded one field width fails here and not there"),
    "fp4_mixed": (64, "FP4 with A as E2M1 and B as E1M2 -- two different four-bit layouts in one "
                      "product, packed low nibble first, so a decoder that used one table for both "
                      "operands disagrees"),
}

CASES = [
    {"id": f"{fmt}_{path}", "seed": 8921 + index, "block_dim": 1,
     "purpose": f"{why}. Through the {path} path"
                + (" -- one call that carries the scales" if path == "shortcut"
                   else ", where l1_to_l0_mx attaches the scales and mmad_mx consumes them; the "
                        "pair with the shortcut case is what establishes the two agree"),
     "parameters": {"format": fmt, "path": path, "payload_cols": cols}}
    for index, (fmt, (cols, why)) in enumerate(FORMATS.items())
    for path in ("shortcut", "explicit")
]


def check_domain(inputs, expected):
    """What the bitwise comparison and the group checks rest on: the scale codes are exact powers of
    two, the scales vary across both rows and groups, and the payload width matches the format."""
    fmt = inputs["format"]
    cols = FORMATS[fmt][0]
    if tuple(inputs["a"].shape) != (TILE[0], cols) or tuple(inputs["b"].shape) != (TILE[0], cols):
        raise ValueError(f"{fmt} payloads must be [{TILE[0]}, {cols}]")
    for name in ("scale_a", "scale_b"):
        scale = inputs[name]
        if tuple(scale.shape) != (TILE[0], GROUPS):
            raise ValueError(f"{name} must be one code per row per K{GROUP_K} group")
        if bool(((scale.int() < 120) | (scale.int() > 134)).any()):
            raise ValueError(f"{name} must stay in the exact power-of-two band this example "
                             f"declares")
        if scale.int().unique(dim=0).shape[0] < 2 or scale.int().unique(dim=1).shape[1] < 2:
            raise ValueError(f"{name} must vary across rows and across groups, or a wrong group "
                             f"selection could not be detected")
    if not bool(torch.isfinite(expected["o"]).all()):
        raise ValueError("the reference must be finite over this restricted band")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives NaN-poisoned and is seeded in."""
    entry = make_mx(inputs["format"], inputs["path"])
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["a"], inputs["b"], inputs["scale_a"], inputs["scale_b"],
                    torch.full(TILE, float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: every product and every 128-element sum is exact in FP32 over this band."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {TILE[0]}x{TILE[1]} FP32 elements")
    if not ok:
        got, want = got.cpu(), want.cpu()
        outside = got != want
        index = outside.nonzero()
        # A wrong scale group scales a whole row or a whole 32-column band by a power of two.
        ratios = (got[outside] / want[outside].clamp(min=1e-30))[:4]
        rows = sorted({int(r) for r, _ in index.tolist()})
        poisoned = int((outside & torch.isnan(got)).sum())
        print(f"      {len(index)}/{got.numel()} elements differ across rows {rows[:6]}; "
              f"got/want ratios {[round(float(r), 4) for r in ratios]}"
              + ("  (a power-of-two ratio points at a scale group, not at a payload)"
                 if all(abs(float(r)) in (0.25, 0.5, 2.0, 4.0) for r in ratios if r == r) else "")
              + (f"; {poisoned} still NaN-poisoned (never written)" if poisoned else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--format", default="all", choices=("all", *FORMATS),
                        help="run only one payload format's cases")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:22s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.format in ("all", case["parameters"]["format"])]
    if not selected:
        parser.error(f"no case matches --case {args.case!r} --format {args.format!r}")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['format']}, {p['path']} path, K={GROUPS * GROUP_K} in "
              f"{GROUPS} scale groups, launcher={args.launcher})")
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
