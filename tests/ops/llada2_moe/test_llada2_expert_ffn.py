# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the llada2_expert_ffn PyPTO kernel (BF16 SwiGLU FFN)."""

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
    sys.exit(0)

import pypto

_IMPL = Path(__file__).resolve().parents[3] / "src" / "pypto_gym" / "ops" / "pypto_tile" / "llada2_moe"
sys.path.insert(0, str(_IMPL))

from llada2_expert_ffn_impl import llada2_expert_ffn


def reference(x, w13, w2):
    """y = down(silu(gate(x)) * up(x)) where gate||up = W13."""
    I = w13.shape[-1] // 2
    gate_up = x.float() @ w13.float()
    gate, up = gate_up[..., :I], gate_up[..., I:]
    sw = F.silu(gate) * up
    return (sw @ w2.float()).to(x.dtype)


def test_llada2_expert_ffn():
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    H = 2048
    I = 512
    torch.manual_seed(0)
    for n in [1, 4, 8, 16, 32, 64, 128]:
        x = torch.randn(n, H, dtype=torch.bfloat16, device=dev) * 0.02
        w13 = torch.randn(H, 2 * I, dtype=torch.bfloat16, device=dev) * 0.02
        w2 = torch.randn(I, H, dtype=torch.bfloat16, device=dev) * 0.02
        out = torch.empty(n, H, dtype=torch.bfloat16, device=dev)

        llada2_expert_ffn(x, w13, w2, out)
        ref = reference(x, w13, w2)

        np.testing.assert_allclose(
            out.float().cpu().numpy(), ref.float().cpu().numpy(),
            rtol=8e-3, atol=8e-3,
            err_msg=f"n={n}: BF16 SwiGLU FFN mismatch",
        )


def main():
    pypto.set_host_options(compile_monitor_enable=True, compile_timeout=10,
                            compile_timeout_stage=5,
                            compile_monitor_print_interval=2)
    test_llada2_expert_ffn()


if __name__ == "__main__":
    main()
