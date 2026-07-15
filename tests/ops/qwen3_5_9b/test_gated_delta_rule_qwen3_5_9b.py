#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-op precision test for the Qwen3.5-9B gated_delta_rule fused kernel.

Compares PyPTO ``gated_delta_rule_wrapper`` against a pure-torch reference at
the real shapes captured from Qwen3.5-9B inference (see ``test_cases.json``).

Two paths are covered, both driven from ``test_cases.json`` (no silent skips):
  * ``chunk`` (prefill): precision-compared PyPTO wrapper vs torch golden.
  * ``recurrent`` (decode): asserted to raise ``NotImplementedError`` so the
    modeling layer routes it to the upstream kernel — i.e. it is *deliberately*
    not fused, and this test pins that contract instead of hiding it.

Run::

    cd <repo_root>
    pytest tests/ops/qwen3_5_9b/test_gated_delta_rule_qwen3_5_9b.py
    # or, for the [PRECISION_PASS] marker on NPU:
    python3 tests/ops/qwen3_5_9b/test_gated_delta_rule_qwen3_5_9b.py
"""
import json
import logging
import os
import sys
from pathlib import Path

import pytest
import torch


_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "pypto_gym" / "ops" / "pypto_tensor"))

from gated_delta_rule_golden import chunk_gated_delta_rule_golden  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_cases():
    with (_HERE / "test_cases.json").open() as f:
        return json.load(f)["test_cases"]


def _mk_tensor(spec, device, *, name=None, seed_offset=0):
    if "shape" not in spec:
        return spec.get("value")
    shape = spec["shape"]
    dt = torch.float32 if spec["dtype"] == "float32" else torch.bfloat16
    torch.manual_seed(42 + seed_offset)
    if name == "g":
        # The real model gate g = -A_log.exp() * softplus(...) is ALWAYS <= 0.
        # A both-signs randn would make exp(g_cum) blow up vs the real decaying
        # regime and mask precision behavior, so use a realistic non-positive
        # log-gate here (matches modeling_qwen3_5.py).
        return -torch.rand(*shape, dtype=dt, device=device) * 0.5
    return torch.randn(*shape, dtype=dt, device=device) * 0.1


def _build_inputs(case, device):
    return {k: _mk_tensor(v, device, name=k, seed_offset=i)
            for i, (k, v) in enumerate(case["input"].items())}


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


def _run_chunk_case(case, device):
    """Run one chunk-path case: PyPTO wrapper vs torch golden + shape checks.
    Returns the max abs error across out + state."""
    from qwen3_5_9b.gated_delta_rule.gated_delta_rule_impl import gated_delta_rule_wrapper

    torch.manual_seed(case["seed"])
    inp = _build_inputs(case, device)
    kw = dict(g=inp["g"], beta=inp["beta"], initial_state=inp["initial_state"],
              output_final_state=bool(inp.get("output_final_state", True)),
              use_qk_l2norm_in_kernel=bool(inp.get("use_qk_l2norm_in_kernel", True)))

    out_pto, state_pto = gated_delta_rule_wrapper(inp["query"], inp["key"], inp["value"], **kw)
    out_ref, state_ref = chunk_gated_delta_rule_golden(inp["query"], inp["key"], inp["value"], **kw)

    # Enforce the documented output envelope (test_cases.json was previously
    # never checked against the actual op output).
    exp = case["output"]
    assert list(out_pto.shape) == exp["core_attn_out"]["shape"], (
        f"{case['id']}: core_attn_out shape {list(out_pto.shape)} != "
        f"{exp['core_attn_out']['shape']}")
    if state_pto is not None:
        assert list(state_pto.shape) == exp["last_recurrent_state"]["shape"], (
            f"{case['id']}: state shape {list(state_pto.shape)} != "
            f"{exp['last_recurrent_state']['shape']}")

    rtol = case.get("rtol", 1e-2)
    atol = case.get("atol", 5e-2)
    md = _compare(out_pto.cpu(), out_ref.cpu(), name="core_attn_out", rtol=rtol, atol=atol)
    if state_pto is not None and state_ref is not None:
        md = max(md, _compare(state_pto.cpu(), state_ref.cpu(),
                              name="last_recurrent_state", rtol=rtol, atol=atol))
    logging.info(f"[{case['id']}] chunk max_abs_err={md:.3e}")
    return md


def _assert_recurrent_upstream(case, device):
    """The recurrent/decode case is NOT fused by PyPTO (initial_state set / decode);
    assert the wrapper declines it so the modeling layer falls back upstream."""
    from qwen3_5_9b.gated_delta_rule.gated_delta_rule_impl import gated_delta_rule_wrapper

    inp = _build_inputs(case, device)
    with pytest.raises(NotImplementedError):
        gated_delta_rule_wrapper(
            inp["query"], inp["key"], inp["value"], g=inp["g"], beta=inp["beta"],
            initial_state=inp["initial_state"],
            output_final_state=bool(inp.get("output_final_state", True)),
            use_qk_l2norm_in_kernel=bool(inp.get("use_qk_l2norm_in_kernel", True)))
    logging.info(f"[{case['id']}] recurrent -> NotImplementedError -> upstream fallback OK")


_CHUNK_CASES = [c for c in _load_cases() if c["path"] == "chunk"]
_RECURRENT_CASES = [c for c in _load_cases() if c["path"] == "recurrent"]


@pytest.mark.parametrize("case", _CHUNK_CASES, ids=lambda c: c["id"])
def test_gated_delta_rule_chunk(case, npu_device):
    """Chunk-path precision: PyPTO wrapper vs torch reference."""
    _run_chunk_case(case, npu_device)


@pytest.mark.parametrize("case", _RECURRENT_CASES, ids=lambda c: c["id"])
def test_recurrent_routes_upstream(case, npu_device):
    """Decode/recurrent case is intentionally not fused -> upstream fallback."""
    _assert_recurrent_upstream(case, npu_device)


def main():
    import torch_npu  # noqa: F401  enables torch.npu for the direct-run path
    torch.npu.set_device(0)
    os.environ["TILE_FWK_DEVICE_ID"] = "0"
    dev = "npu:0"
    for case in _load_cases():
        if case["path"] == "chunk":
            _run_chunk_case(case, dev)
        else:
            _assert_recurrent_upstream(case, dev)
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
