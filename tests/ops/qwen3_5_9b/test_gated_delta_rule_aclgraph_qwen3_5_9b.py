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
"""ACLGraph / torch.library registration test for the Qwen3.5-9B gated_delta_rule kernel.

Asserts the registered op ``pypto::gated_delta_rule_qwen3_5`` (Meta + NPU) is:
  1. bit-identical to the eager ``gated_delta_rule_wrapper``;
  2. Meta-shape-inferrable under ``FakeTensorMode``;
  3. ``torch.compile(fullgraph=True)`` traceable with no graph break;
  4. real torchair aclgraph (``reduce-overhead``) capture + replay bit-exact.

The kernel has no in-kernel ``torch_npu.npu.synchronize()`` (it is stream-ordered),
so the registered op is aclgraph-capture-clean.

Run::

    pytest tests/ops/qwen3_5_9b/test_gated_delta_rule_aclgraph_qwen3_5_9b.py
    # or, for the [PRECISION_PASS] marker on NPU:
    python3 tests/ops/qwen3_5_9b/test_gated_delta_rule_aclgraph_qwen3_5_9b.py
"""
import logging
import os
import sys
from pathlib import Path

import torch


_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "pypto_gym" / "ops" / "pypto_tensor"))

from qwen3_5_9b.gated_delta_rule.gated_delta_rule_impl import (  # noqa: E402
    gated_delta_rule_wrapper, gated_delta_rule_pypto)

logging.basicConfig(level=logging.INFO, format="%(message)s")

_NV = 32
_D = 128

try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    torch_npu = None
    _HAS_NPU = False


def _mk_inputs(device, s=16):
    torch.manual_seed(0)
    q = (torch.randn(1, s, _NV, _D) * 0.1).to(torch.bfloat16).to(device)
    k = (torch.randn(1, s, _NV, _D) * 0.1).to(torch.bfloat16).to(device)
    v = (torch.randn(1, s, _NV, _D) * 0.1).to(torch.bfloat16).to(device)
    g = (-torch.rand(1, s, _NV) * 0.5).to(torch.float32).to(device)      # log-gate <= 0
    beta = (torch.rand(1, s, _NV) * 0.5).to(torch.bfloat16).to(device)
    return q, k, v, g, beta


def _assert_identical(a, b, name):
    assert torch.equal(a.cpu(), b.cpu()), f"{name}: registered op != wrapper (not bit-identical)"


class _M(torch.nn.Module):
    def forward(self, q, k, v, g, beta):
        return gated_delta_rule_pypto(q, k, v, g=g, beta=beta, initial_state=None,
                                      use_qk_l2norm_in_kernel=True)


def test_registered_op_matches_wrapper(npu_device):
    """Registered NPU op == eager wrapper, bit-for-bit."""
    q, k, v, g, beta = _mk_inputs(npu_device)
    ow, sw = gated_delta_rule_wrapper(q, k, v, g=g, beta=beta, initial_state=None,
                                      output_final_state=True, use_qk_l2norm_in_kernel=True)
    op, sp = gated_delta_rule_pypto(q, k, v, g=g, beta=beta, initial_state=None,
                                    use_qk_l2norm_in_kernel=True)
    _assert_identical(op.float(), ow.float(), "out")
    _assert_identical(sp, sw, "state")
    logging.info("[aclgraph] registered pypto::gated_delta_rule_qwen3_5 == wrapper (bit-identical)")


def test_meta_shape_inference():
    """Meta impl infers output shapes/dtypes under FakeTensorMode (no NPU needed)."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    dev = "npu" if _HAS_NPU else "cpu"
    with FakeTensorMode():
        q = torch.empty(1, 16, _NV, _D, dtype=torch.bfloat16, device=dev)
        k = torch.empty(1, 16, _NV, _D, dtype=torch.bfloat16, device=dev)
        v = torch.empty(1, 16, _NV, _D, dtype=torch.bfloat16, device=dev)
        g = torch.empty(1, 16, _NV, dtype=torch.float32, device=dev)
        beta = torch.empty(1, 16, _NV, dtype=torch.bfloat16, device=dev)
        out, st = torch.ops.pypto.gated_delta_rule_qwen3_5(q, k, v, g, beta, None, True)
        assert tuple(out.shape) == (1, 16, _NV, _D) and out.dtype == torch.bfloat16, "meta out"
        assert tuple(st.shape) == (1, _NV, _D, _D) and st.dtype == torch.float32, "meta state"
    logging.info("[aclgraph meta] Meta shape inference OK for gated_delta_rule_qwen3_5")


def test_torch_compile_fullgraph(npu_device):
    """Registered op composes into a torch.compile fullgraph (no graph break), bit-exact."""
    q, k, v, g, beta = _mk_inputs(npu_device)
    m = _M()
    eo, es = m(q, k, v, g, beta)
    for backend in ("aot_eager", "eager"):
        cm = torch.compile(m, backend=backend, fullgraph=True, dynamic=False)
        co, cs = cm(q, k, v, g, beta)
        _assert_identical(co.float(), eo.float(), f"compile[{backend}] out")
        _assert_identical(cs, es, f"compile[{backend}] state")
        logging.info(f"[aclgraph compile] fullgraph torch.compile({backend}) no graph break (bit-exact)")


def test_aclgraph_capture(npu_device):
    """Real torchair aclgraph (reduce-overhead) capture + replay is bit-exact."""
    try:
        import torchair
    except ImportError:
        logging.info("[aclgraph capture] torchair unavailable; skipped")
        return
    q, k, v, g, beta = _mk_inputs(npu_device)
    m = _M()
    eo, es = m(q, k, v, g, beta)
    cfg = torchair.CompilerConfig()
    cfg.mode = "reduce-overhead"
    cm = torch.compile(m, backend=torchair.get_npu_backend(compiler_config=cfg), dynamic=False)
    for _ in range(3):
        co, cs = cm(q, k, v, g, beta)
    _assert_identical(co.float(), eo.float(), "aclgraph capture out")
    _assert_identical(cs, es, "aclgraph capture state")
    logging.info("[aclgraph capture] torchair reduce-overhead capture+replay bit-exact (diff=0)")


def main():
    test_meta_shape_inference()
    if not _HAS_NPU:
        logging.info("skip NPU checks: no NPU")
        return
    torch_npu.npu.config.allow_internal_format = True
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(dev_id)
    dev = f"npu:{dev_id}"
    test_registered_op_matches_wrapper(dev)
    test_torch_compile_fullgraph(dev)
    test_aclgraph_capture(dev)
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    main()
