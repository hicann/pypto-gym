# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the grouped GEMM MoE kernel (all experts in one call).

Verifies that the single-kernel grouped path matches the per-expert reference
(loop of torch matmuls) within BF16 tolerance.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("torch_npu not available; this test only runs on Ascend NPU.")
    raise RuntimeError("torch_npu not available; this test only runs on Ascend NPU.") from None

import pypto

_IMPL = Path(__file__).resolve().parents[3] / "src" / "pypto_gym" / "ops" / "pypto_tile" / "llada2_moe"
sys.path.insert(0, str(_IMPL))

from llada2_moe_grouped_gemm_impl import (
    llada2_moe_grouped_gemm,
)


def reference_per_expert(sorted_tokens, w13_list, w2_list, expert_cumsum):
    """Per-expert reference using pure PyTorch (FP32 accumulation)."""
    E = len(w13_list)
    N, H = sorted_tokens.shape
    result = torch.zeros_like(sorted_tokens)

    for e in range(E):
        start = expert_cumsum[e].item()
        end = expert_cumsum[e + 1].item()
        if start == end:
            continue
        x = sorted_tokens[start:end].float()
        w13 = w13_list[e].float()
        w2 = w2_list[e].float()

        gate_up = x @ w13                    # [n_e, 2I]
        I = w13.shape[1] // 2
        gate, up = gate_up[..., :I], gate_up[..., I:]
        sw = F.silu(gate) * up               # [n_e, I]
        down = sw @ w2                       # [n_e, H]
        result[start:end] = down.to(sorted_tokens.dtype)

    return result


def _run_grouped_gemm_test(E, total_tokens, H, I, dev):
    """Run a single (E, total_tokens) grouped-GEMM test case."""
    torch.manual_seed(42)
    counts = torch.zeros(E, dtype=torch.int64)
    remaining = total_tokens
    for e in range(E - 1):
        c = torch.randint(0, remaining // (E - e) * 2 + 1, (1,)).item()
        c = min(c, remaining)
        counts[e] = c
        remaining -= c
    counts[E - 1] = remaining
    cumsum = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32).to(dev)
    N = int(counts.sum().item())
    sorted_tokens = torch.randn(N, H, dtype=torch.bfloat16, device=dev) * 0.02
    w13_list = [torch.randn(H, 2 * I, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(E)]
    w2_list = [torch.randn(I, H, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(E)]
    w13_flat = torch.cat(w13_list, dim=0).contiguous()
    w2_flat = torch.cat(w2_list, dim=0).contiguous()
    result = torch.empty(N, H, dtype=torch.bfloat16, device=dev)
    llada2_moe_grouped_gemm(
        sorted_tokens, w13_flat, w2_flat, cumsum, result,
        num_experts=E, hidden_size=H, intermediate_size=I)
    ref = reference_per_expert(sorted_tokens, w13_list, w2_list, cumsum)
    np.testing.assert_allclose(
        result.float().cpu().numpy(), ref.float().cpu().numpy(),
        rtol=8e-3, atol=8e-3,
        err_msg=(f"E={E}, N={N}: grouped GEMM mismatch (counts={counts.tolist()})"))
    print(f"  PASS  E={E:>3d}  N={N:>4d}  counts={counts.tolist()}")


def _run_zero_token_test(H, I, dev):
    """Run zero-token experts test."""
    E = 8
    N = 16
    torch.manual_seed(7)
    sorted_tokens = torch.randn(N, H, dtype=torch.bfloat16, device=dev) * 0.02
    cumsum_zero = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    cumsum_zero[1] = 8
    cumsum_zero[2] = 8
    cumsum_zero[3] = 8
    cumsum_zero[4] = 16
    cumsum_zero[5] = 16
    cumsum_zero[6] = 16
    cumsum_zero[7] = 16
    cumsum_zero[8] = 16
    w13_list = [torch.randn(H, 2 * I, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(E)]
    w2_list = [torch.randn(I, H, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(E)]
    w13_flat = torch.cat(w13_list, dim=0).contiguous()
    w2_flat = torch.cat(w2_list, dim=0).contiguous()
    result = torch.empty(N, H, dtype=torch.bfloat16, device=dev)
    llada2_moe_grouped_gemm(
        sorted_tokens, w13_flat, w2_flat, cumsum_zero, result,
        num_experts=E, hidden_size=H, intermediate_size=I)
    ref = reference_per_expert(sorted_tokens, w13_list, w2_list, cumsum_zero)
    np.testing.assert_allclose(
        result.float().cpu().numpy(), ref.float().cpu().numpy(),
        rtol=8e-3, atol=8e-3,
        err_msg="zero-token experts: grouped GEMM mismatch")
    print("  PASS  zero-token experts test")


def test_llada2_moe_grouped_gemm():
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"
    H = 2048
    I = 512

    for E in [4, 8]:
        for total_tokens in [8, 32, 64, 128]:
            _run_grouped_gemm_test(E, total_tokens, H, I, dev)

    _run_zero_token_test(H, I, dev)


def main():
    pypto.set_host_options(compile_monitor_enable=1, compile_timeout=10,
                            compile_timeout_stage=5,
                            compile_monitor_print_interval=2)
    test_llada2_moe_grouped_gemm()
    print("\nAll grouped GEMM tests passed.")


if __name__ == "__main__":
    main()
