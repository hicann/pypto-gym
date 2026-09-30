# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load a recorded functional case (RFC-0003) and compare outputs against it.

A case directory holds ``{manifest.json,inputs,outputs}``. ``ascriptor sim --case`` and
:mod:`ascriptor.runtime.opexec` load its arguments and compare every output bit for bit
(bf16 and the 8-bit carriers through their raw bytes), or within a recorded tolerance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# libgomp is not fork-safe once it has spawned threads: keep torch single-threaded in the replaying
# process so the interpreter's forked core groups stay healthy (must precede the first torch import)
os.environ["OMP_NUM_THREADS"] = "1"

import torch  # noqa: E402

TORCH_DTYPES = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "uint8": torch.uint8,
    "int32": torch.int32, "int64": torch.int64, "int8": torch.int8, "int16": torch.int16,
}
for _name in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32", "uint64", "complex64", "complex32"):
    if hasattr(torch, _name):
        TORCH_DTYPES[_name] = getattr(torch, _name)


def load_tensor(case: Path, rec: dict) -> torch.Tensor:
    raw = (case / rec["file"]).read_bytes()
    t = torch.frombuffer(bytearray(raw), dtype=TORCH_DTYPES[rec["dtype"]])
    return t.reshape(rec["shape"]).clone()


def raw_bytes(t: torch.Tensor) -> torch.Tensor:
    """The bytes of a tensor: the comparison is bitwise, so NaN payloads and -0.0 count as well."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def load_args(case: Path, manifest: dict) -> list[Any]:
    args: list[Any] = []
    for a in manifest["args"]:
        if a["kind"] == "tensor":
            args.append(load_tensor(case, a))
        elif a["kind"] == "scalar":
            args.append(a["value"])
        elif a["kind"] == "tensor_list":
            args.append([load_tensor(case, item) for item in a["items"]])
        else:
            raise ValueError(f"argument kind {a['kind']!r} is not supported by the replay")
    return args


def flatten_outputs(result: Any) -> list[Any]:
    """The outputs of a run as the goldens record them (``record_goldens._flatten``): one flat list, a list output
    contributing its members in order."""
    if isinstance(result, (list, tuple)):
        return [t for v in result for t in flatten_outputs(v)]
    return [result]


def unwritten_elements(got: torch.Tensor, want: torch.Tensor, prefill: torch.Tensor | None) -> torch.Tensor | None:
    """Elements the kernel provably wrote on neither side: still poison (0xFF bytes) in the replay and still the
    pre-filled input in the golden. Everything else is compared; None when the output was not a parameter."""
    if prefill is None or tuple(prefill.shape) != tuple(got.shape) or prefill.dtype != got.dtype:
        return None
    es = got.element_size()
    g, w, pf = raw_bytes(got).view(-1, es), raw_bytes(want).view(-1, es), raw_bytes(prefill).view(-1, es)
    return (g == 0xFF).all(dim=1) & (w == pf).all(dim=1)


def within_tolerance(got: torch.Tensor, want: torch.Tensor, tolerance: dict, excused: torch.Tensor | None = None) -> bool:
    """``tolerance = {"rtol": r, "atol": a}`` compares as numbers (NaN == NaN); anything else is bitwise."""
    if got.dtype != want.dtype or tuple(got.shape) != tuple(want.shape):
        return False
    keep = None if excused is None else ~excused
    if "rtol" not in tolerance and "atol" not in tolerance:
        g, w = raw_bytes(got), raw_bytes(want)
        if keep is not None:
            k = keep.repeat_interleave(got.element_size())
            g, w = g[k], w[k]
        return bool(torch.equal(g, w))
    g, w = got.float().reshape(-1), want.float().reshape(-1)
    if keep is not None:
        g, w = g[keep], w[keep]
    rtol = float(tolerance.get("rtol", 0.0))
    # A half / bf16 output flips its last bit whenever the fp32 accumulation order differs, so a
    # relative tolerance below one ULP of the output dtype compares nothing but that flip: floor it.
    if want.dtype == torch.float16:
        rtol = max(rtol, 1e-3)  # 2^-10
    elif want.dtype == torch.bfloat16:
        rtol = max(rtol, 4e-3)  # 2^-8
    return bool(torch.allclose(g, w, rtol=rtol, atol=float(tolerance.get("atol", 0.0)), equal_nan=True))


def index_gather_mismatch(got: torch.Tensor, want: torch.Tensor, spec: dict, case: Path, manifest: dict) -> str | None:
    """Why an INDEX output fails to denote the golden's values, or None when it denotes them exactly.

    Silicon may order equal keys differently from the recording (`sort_rows`: `vbitsort` breaks a tie
    the other way round, D-070/D-213), and a tie shows only in the index output -- the values beside
    it are bit-identical. No rtol/atol can express that: a tolerance compares magnitudes, and index
    209 against index 1128 has none. So the entry says what the output MEANS instead, and the check
    becomes stronger rather than looser:

        "outputs": {"1": {"index_gather": {"input": 0, "values": 0}}}

    output 1 holds indices along the last axis into argument 0, and it passes only if (a) gathering
    that argument by them reproduces golden output ``values`` BIT for bit and (b) each row selects
    the same multiset of positions the golden selected. Together those leave exactly one freedom --
    the order among equal keys -- and nothing else."""
    try:
        src_i, val_i = int(spec["input"]), int(spec["values"])
    except (KeyError, TypeError, ValueError):
        return f"index_gather spec is not {{'input': i, 'values': j}}: {spec!r}"
    if not torch.is_floating_point(got) and got.dtype not in (torch.int32, torch.int64, torch.int16):
        return f"index_gather names a {got.dtype} output; indices must be integers"
    args = manifest.get("args", [])
    if not (0 <= src_i < len(args)) or args[src_i].get("kind") != "tensor":
        return f"index_gather input {src_i} is not a tensor argument"
    outs = manifest.get("outputs", [])
    if not (0 <= val_i < len(outs)):
        return f"index_gather values {val_i} is not an output"
    src = load_tensor(case, args[src_i])
    values = load_tensor(case, outs[val_i])
    if tuple(src.shape) != tuple(got.shape) or tuple(values.shape) != tuple(got.shape):
        return (f"index_gather shapes disagree: indices {tuple(got.shape)}, input {tuple(src.shape)}, "
                f"values {tuple(values.shape)}")
    idx = got.long()
    if bool(((idx < 0) | (idx >= src.shape[-1])).any()):
        return "index_gather: an index is outside the input's last axis"
    picked = torch.take_along_dim(src, idx, dim=-1)
    if not torch.equal(raw_bytes(picked), raw_bytes(values)):
        bad = int((raw_bytes(picked) != raw_bytes(values)).sum())
        return f"index_gather: the indices select {bad} byte(s) that are not the recorded values"
    ref = want.long()
    if not torch.equal(idx.sort(dim=-1).values, ref.sort(dim=-1).values):
        return "index_gather: a row selects positions the golden did not select"
    return None


def compare_outputs(outputs: list[torch.Tensor], case: Path, manifest: dict, tolerance: dict | None = None,
                    prefill: list[torch.Tensor | None] | None = None, notes: list[str] | None = None) -> list[str]:
    """Differences between the replayed outputs and the golden (empty = equal). Output parameters are poisoned
    at launch (D-026); an element is excused only when neither side wrote it (``unwritten_elements``), and
    ``notes`` receives one line per output that has such elements."""
    diffs: list[str] = []
    outputs = flatten_outputs(outputs)  # a list output's members one by one, as the golden records them
    if len(outputs) != len(manifest["outputs"]):
        return [f"{len(outputs)} outputs, golden has {len(manifest['outputs'])}"]
    for i, (got, rec) in enumerate(zip(outputs, manifest["outputs"], strict=True)):
        want = load_tensor(case, rec)
        if got.dtype != want.dtype or tuple(got.shape) != tuple(want.shape):
            diffs.append(f"output {i}: {got.dtype} {tuple(got.shape)} vs golden {want.dtype} {tuple(want.shape)}")
            continue
        excused = unwritten_elements(got, want, prefill[i] if prefill is not None and i < len(prefill) else None)
        if excused is not None:
            n_exc = int(excused.sum())
            if n_exc == 0:
                excused = None
            elif notes is not None:
                notes.append(f"output {i}: {n_exc} element(s) the kernel never writes (poison in the replay, the pre-filled input in the golden)")
        rvp = (tolerance or {}).get("rows_valid_per_period")
        if rvp and want.dim() >= 2:
            # rows r with r % period >= valid hold no comparable data (e.g. the conv M tail, where the
            # hardware and the recording simulator legitimately disagree) — excuse them wholesale
            valid, period = int(rvp[0]), int(rvp[1])
            row_excuse = ((torch.arange(want.shape[0]) % period) >= valid).unsqueeze(-1).expand(want.shape).reshape(-1)
            if bool(row_excuse.any()):
                excused = row_excuse if excused is None else (excused | row_excuse)
                if notes is not None:
                    notes.append(f"output {i}: rows r%{period}>={valid} masked by the corpus board_tolerance entry "
                                 f"({int(row_excuse.sum())} elements)")
        # a kernel whose outputs are of different KINDS (an fp32 result beside a byte carrier of
        # an 8-bit float) needs different bounds per output: one entry may name them under
        # "outputs", keyed by index, and anything not named falls back to the kernel's own
        tol_i = tolerance
        per = (tolerance or {}).get("outputs")
        if per is not None:
            tol_i = per.get(str(i), per.get(i, {k: v for k, v in tolerance.items() if k != "outputs"}))
        spec = (tol_i or {}).get("index_gather")
        if spec is not None:
            why = index_gather_mismatch(got, want, spec, case, manifest)
            if why is None:
                if not torch.equal(raw_bytes(got), raw_bytes(want)) and notes is not None:
                    n = int((got != want).sum())
                    notes.append(f"output {i}: {n} index entr(y/ies) differ from the golden and every one of them "
                                 f"denotes the same recorded value (corpus board_tolerance index_gather)")
                continue
            diffs.append(f"output {i}: {why}")
            continue
        if tol_i and within_tolerance(got, want, tol_i, excused):
            continue
        g, w = raw_bytes(got), raw_bytes(want)
        same = g == w
        if excused is not None:
            same |= excused.repeat_interleave(got.element_size())
        if bool(same.all()):
            continue
        try:
            bad = int((~same).sum())
            mx = (got.float() - want.float()).abs().max().item()
            diffs.append(f"output {i}: {bad} differing bytes, max abs diff {mx:.3e}")
        except Exception:  # noqa: BLE001 - reporting only
            diffs.append(f"output {i}: differs")
    return diffs

