# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the llada2_gate_select PyPTO kernel.

Compares the kernel output against a pure-torch reference derived from
LLaDA2MoeGate.forward (modeling_llada2_moe.py:253). Tolerances are tight
(rtol=1e-3, atol=1e-3) because all the heavy math runs in FP32.
"""

import os
import sys
import math
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("torch_npu not available; this test only runs on Ascend NPU.")
    raise SystemExit(0)

import pypto

_IMPL = Path(__file__).resolve().parents[3] / "src" / "pypto_gym" / "ops" / "pypto_tile" / "llada2_moe"
sys.path.insert(0, str(_IMPL))

from llada2_gate_select_impl import llada2_gate_select


def reference_amax(hidden_states, gate_weight, expert_bias, top_k, topk_group,
                   num_expert_group, routed_scaling_factor):
    """Reference using amax group scoring (matches the kernel implementation).

    The kernel uses amax (top-1) instead of sum-of-top-2 for group scoring
    due to Ascend's 32B-aligned reduction constraint. This reference mirrors
    that behavior for exact numerical comparison.
    """
    flat = hidden_states.view(-1, hidden_states.shape[-1])
    logits = torch.nn.functional.linear(flat.float(), gate_weight.float())
    scores = torch.sigmoid(logits)

    scores_aug = scores + expert_bias
    N = scores_aug.shape[0]
    # amax (top-1) group scoring — matches the kernel
    group_scores = scores_aug.view(N, num_expert_group, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(N, num_expert_group, gate_weight.shape[0] // num_expert_group)
        .reshape(N, -1)
    )
    masked = scores_aug.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(masked, k=top_k, dim=-1)

    tw = torch.gather(scores, dim=1, index=topk_ids)
    if top_k > 1:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    tw = tw * routed_scaling_factor
    return topk_ids.to(torch.int32), tw.to(torch.float32)


def test_llada2_gate_select():
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    # LLaDA2.0-mini routing config
    H = 2048
    E = 256
    K = 8
    topk_group = 4
    num_expert_group = 8
    routed_scaling_factor = 2.5

    torch.manual_seed(0)
    for bs in [1, 8, 16, 32, 32]:
        hidden = (torch.randn(bs, H, dtype=torch.bfloat16, device=dev) * 0.02)
        gate_w = (torch.randn(E, H, dtype=torch.float32, device=dev) * 0.05)
        bias = (torch.randn(E, dtype=torch.float32, device=dev) * 0.01)

        tw_out = torch.empty(bs, K, dtype=torch.float32, device=dev)
        ids_out = torch.empty(bs, K, dtype=torch.int32, device=dev)

        llada2_gate_select(
            hidden, gate_w, bias, tw_out, ids_out,
            top_k=K, topk_group=topk_group,
            num_expert_group=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        ref_ids, ref_tw = reference_amax(
            hidden, gate_w, bias, K, topk_group,
            num_expert_group, routed_scaling_factor,
        )

        # Both kernel and reference use the same amax group-scoring, so
        # expert IDs should match exactly (sorted order may differ).
        for i in range(bs):
            kernel_ids = sorted(ids_out[i].cpu().tolist())
            ref_id_list = sorted(ref_ids[i].cpu().tolist())
            assert kernel_ids == ref_id_list, (
                f"row {i}: expert IDs differ. "
                f"kernel={kernel_ids} ref={ref_id_list}"
            )

        # Compare weights for each expert (order may differ between
        # kernel and reference due to topk tie-breaking).
        for i in range(bs):
            k_ids = ids_out[i].cpu().tolist()
            r_ids = ref_ids[i].cpu().tolist()
            for eid in set(k_ids):
                k_pos = k_ids.index(eid)
                r_pos = r_ids.index(eid)
                np.testing.assert_allclose(
                    tw_out[i, k_pos].cpu().numpy(),
                    ref_tw[i, r_pos].cpu().numpy(),
                    rtol=1e-3, atol=1e-3,
                    err_msg=f"row {i}, expert {eid}: weight mismatch",
                )
        print(f"  bs={bs}: {bs}/{bs} rows PASS (IDs + weights)")


def main():
    pypto.set_host_options(compile_monitor_enable=True, compile_timeout=10,
                            compile_timeout_stage=5,
                            compile_monitor_print_interval=2)
    test_llada2_gate_select()


if __name__ == "__main__":
    main()
