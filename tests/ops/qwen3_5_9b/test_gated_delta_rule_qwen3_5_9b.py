#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Single-op precision test for the Qwen3.5-9B gated_delta_rule fused kernel.

Compares PyPTO `gated_delta_rule_wrapper` against a pure-torch reference at
the real shapes captured from Qwen3.5-9B inference (see ``test_cases.json``).
Only the ``chunk`` path is exercised — the ``recurrent`` decode path is
handled by the upstream kernel.

Run::

    cd <repo_root>
    pytest tests/ops/qwen3_5_9b/test_gated_delta_rule.py
"""
import json
import sys
from pathlib import Path

import pytest
import torch


_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "pypto_gym" / "ops" / "pypto_tile"))

from gated_delta_rule_golden import chunk_gated_delta_rule_golden


def _load_cases():
    with (_HERE / "test_cases.json").open() as f:
        return json.load(f)["test_cases"]


def _mk_tensor(spec, device, seed_offset=0):
    if "shape" not in spec:
        return spec.get("value")
    shape = spec["shape"]
    dt_name = spec["dtype"]
    dt = torch.float32 if dt_name == "float32" else torch.bfloat16
    torch.manual_seed(42 + seed_offset)
    return torch.randn(*shape, dtype=dt, device=device) * 0.1


def _compare(actual, expected, *, name, rtol, atol):
    diff = (actual.float() - expected.float()).abs()
    max_d = diff.max().item()
    tol = atol + rtol * expected.float().abs()
    oob = int((diff > tol).sum().item())
    assert oob == 0, (
        f"{name}: {oob}/{actual.numel()} elements out of tolerance "
        f"(rtol={rtol}, atol={atol}, max_diff={max_d:.3e})"
    )
    return max_d


_CHUNK_CASES = [c for c in _load_cases() if c["path"] == "chunk"]


@pytest.mark.parametrize("case", _CHUNK_CASES, ids=lambda c: c["id"])
def test_gated_delta_rule_chunk(case, npu_device):
    """Chunk-path precision: PyPTO wrapper vs torch reference."""
    from qwen3_5_9b.gated_delta_rule.gated_delta_rule_impl import gated_delta_rule_wrapper

    torch.manual_seed(case["seed"])
    inp = {k: _mk_tensor(v, npu_device, seed_offset=i) for i, (k, v) in enumerate(case["input"].items())}

    out_pto, state_pto = gated_delta_rule_wrapper(
        inp["query"], inp["key"], inp["value"],
        g=inp["g"], beta=inp["beta"],
        initial_state=inp["initial_state"],
        output_final_state=bool(inp.get("output_final_state", True)),
        use_qk_l2norm_in_kernel=bool(inp.get("use_qk_l2norm_in_kernel", True)),
    )
    out_ref, state_ref = chunk_gated_delta_rule_golden(
        inp["query"], inp["key"], inp["value"],
        g=inp["g"], beta=inp["beta"],
        initial_state=inp["initial_state"],
        output_final_state=bool(inp.get("output_final_state", True)),
        use_qk_l2norm_in_kernel=bool(inp.get("use_qk_l2norm_in_kernel", True)),
    )

    rtol = case.get("rtol", 1e-2)
    atol = case.get("atol", 5e-2)
    _compare(out_pto.cpu(), out_ref.cpu(), name="core_attn_out", rtol=rtol, atol=atol)
    if state_pto is not None and state_ref is not None:
        _compare(state_pto.cpu(), state_ref.cpu(), name="last_state", rtol=rtol, atol=atol)
