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
"""
"""
import os
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format
from glm_ffn_common_interface import symmetric_quantization_per_token, dequant_dynamic, swiglu


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


def powers_of_2(n: int) -> set[int]:
    check_cond(n > 0, "n must be positive")
    result = set()
    power = 0
    while True:
        current = 1 << power
        if current > n:
            break
        result.add(current)
        power += 1
    return result


def check_args(
        gate_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k: int,
        renormalize: bool,
        topk_group: int,
        num_expert_group: int,
        e_score_correction_bias: torch.Tensor,
        w13,
        w13_scale,
        w2,
        w2_scale
) -> None:
    check_cond(gate_weight.dim() == 2, "invalid gate weight dim.")
    check_cond(gate_weight.shape[0] == 160, "invalid gate weight shape.")
    check_cond(gate_weight.shape[1] == 5120, "invalid gate weight shape.")
    check_cond(get_format(gate_weight) == 'ND', "invalid gate weight format.")
    check_cond((gate_weight.dtype == torch.float32), "invalid gate weight dtype.")
    check_cond(hidden_states.dim() == 2, "invalid hidden states dim.")
    check_cond(hidden_states.shape[1] == 5120, "invalid hidden states shape.")
    check_cond(get_format(hidden_states) == 'ND', "invalid hidden states format.")
    check_cond(hidden_states.dtype == torch.bfloat16, "invalid hidden states dtype.")

    check_cond(e_score_correction_bias.dim() == 1, "invalid bias dim.")
    check_cond(e_score_correction_bias.shape[0] == 160, "invalid bias shape.")
    check_cond(get_format(e_score_correction_bias) == 'ND', "invalid bias format.")
    check_cond(e_score_correction_bias.dtype == torch.bfloat16, "invalid bias dtype.")
    check_cond(isinstance(top_k, int), "invalid topk dtype.")
    check_cond(isinstance(renormalize, bool), "invalid renormalize dtype.")
    check_cond(isinstance(topk_group, int), "invalid topk_group dtype.")
    check_cond(isinstance(num_expert_group, int), "invalid num_expert_group dtype.")

    check_cond(w13.dim() == 2, "invalid w13 dim.")
    check_cond(w13.shape[0] == 5120, "invalid w13 shape.")
    check_cond(w13.shape[1] == 384, "invalid w13 shape.")
    check_cond(get_format(w13) == 'NZ', "invalid w13 format.")
    check_cond(w13.dtype == torch.int8, "invalid w13 dtype.")
    check_cond(w13_scale.dim() == 1, "invalid w13_scale dim.")
    check_cond(w13_scale.shape[0] == 384, "invalid w13_scale shape.")
    check_cond(get_format(w13_scale) == 'ND', "invalid w13_scale format.")
    check_cond(w13_scale.dtype == torch.bfloat16, "invalid w13_scale dtype.")
    check_cond(w2.dim() == 2, "invalid w2 dim.")
    check_cond(w2.shape[0] == 192, "invalid w2 shape.")
    check_cond(w2.shape[1] == 5120, "invalid w2 shape.")
    check_cond(get_format(w2) == 'NZ', "invalid w2 format.")
    check_cond(w2.dtype == torch.int8, "invalid w2 dtype.")
    check_cond(w2_scale.dim() == 1, "invalid w2_scale dim.")
    check_cond(w2_scale.shape[0] == 5120, "invalid w2_scale shape.")
    check_cond(get_format(w2_scale) == 'ND', "invalid w2_scale format.")
    check_cond(w2_scale.dtype == torch.bfloat16, "invalid hidden states dtype.")


def gen_quan_per_channel_weight_nz(x):
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    # NZ out
    y_int8_nz = torch_npu.npu_format_cast(y_int8, 29)
    scale_dequant = (1 / scale_quant)
    return y_int8_nz, scale_dequant


ND = pypto.TileOpFormat.TILEOP_ND
NZ = pypto.TileOpFormat.TILEOP_NZ


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_max_num": 128,
                    "stitch_cfgcache_size": 7700000},
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)
def moe_fusion_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    mm_weight: pypto.Tensor([], pypto.DT_FP32, format=ND),
    e_score_bias_input: pypto.Tensor([], pypto.DT_BF16, format=ND),
    w13: pypto.Tensor([], pypto.DT_INT8, format=NZ),
    w13_scale: pypto.Tensor([], pypto.DT_BF16, format=ND),
    w2: pypto.Tensor([], pypto.DT_INT8, format=NZ),
    w2_scale: pypto.Tensor([], pypto.DT_BF16, format=ND),
    weight_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32, format=ND),
    ids_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32, format=ND),
    ffn_res: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    topk_group,
    num_expert_group,
):
    # 3. 得到动态tensor的shape
    bs = hidden_states.shape[0]

    ne = mm_weight.shape[0]
    topk = ids_k.shape[1]

    pypto.experimental.set_operation_options(combine_axis=True)

    # tiling config
    vec_tile_shape = (4, 5120)
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 192, 256)
    hidden_size = hidden_states.shape[1]
    intermediate_size = w2.shape[0]

    # 4. 定义动态函数
    pypto.set_vec_tile_shapes(ne)
    e_score_bias_2d = pypto.reshape(e_score_bias_input, [1, ne], inplace=True)  # (160) -> (1,160)

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx, tile_batch in pypto.loop_unroll(0, bs, 1, name="LOOP_MOE_FUSION_L0", idx_name="bs_idx",
                                                unroll_list=powers_of_2(32)):
        # 5. 通过view得到tile_logits
        tile_hidden_states = hidden_states[bs_idx:bs_idx + tile_batch, :]

        # gate start
        pypto.set_vec_tile_shapes(min(tile_batch, 4), 5120)
        tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, pypto.DT_FP32)
        mm_weight_fp32 = pypto.cast(mm_weight, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([min(tile_batch, 32), min(tile_batch, 32)], [512, 1024], [16, 16])
        res = pypto.matmul(tile_hidden_states_fp32, mm_weight_fp32, tile_hidden_states_fp32.dtype, b_trans=True)
        # gate end

        # select start
        tile_logits = res
        view_first = 1

        # 7. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
        pypto.set_vec_tile_shapes(view_first, ne)
        tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
        e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
        for tmp_idx in range(tile_batch):
            pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
        e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)

        # sigmoid
        topk_weights = pypto.sigmoid(tile_logits_fp32)  # (bs, ne) fp32

        # add
        topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)  # (8, 160) fp32
        # reshape
        group_unit = ne // num_expert_group
        r1 = pypto.reshape(topk_weights_add, [tile_batch, num_expert_group, group_unit])

        # amax
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        max1 = pypto.amax(r1, -1, False)
        group_weight = max1

        # topk
        pypto.set_vec_tile_shapes(view_first, num_expert_group)
        _, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)  # (2, topk_group) int32

        # zeros -> full(0)
        topk_group_mask = pypto.full([tile_batch, num_expert_group], 0.0, group_weight.dtype)  # (16, 1)

        # scatter 尾轴不能切
        topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

        # unsqueeze
        twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)  # (1, 1, 1) fp32

        # expand
        pypto.set_vec_tile_shapes(view_first, num_expert_group, ne)  # ne时 可以切成一块
        twm_expand = pypto.expand_clone(twm_unsqueeze, [tile_batch, num_expert_group, group_unit])

        # reshape
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)  # (1,1,160)
        twm_reshape = pypto.reshape(twm_expand, [tile_batch, ne])

        # logical_not
        pypto.set_vec_tile_shapes(view_first, ne)
        twm_not = pypto.logical_not(twm_reshape)

        # where
        topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

        # topk2
        _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)  # (bs, topk) int32

        # tw_gather
        tw_gather = pypto.gather(topk_weights, 1, topk_ids)  # (bs, 8)

        # sum & div
        pypto.set_vec_tile_shapes(view_first, topk)
        denominator = pypto.sum(tw_gather, -1, True)  # (bs, 1)

        # div for shape (b*s, topk) (b*s, 1)
        topk_weight_out = pypto.div(tw_gather, denominator)  # (bs, topk)

        weight_k[bs_idx:bs_idx + tile_batch, :] = topk_weight_out
        ids_k[bs_idx:bs_idx + tile_batch, :] = topk_ids
        # select end

        # share start
        pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
        hidden_states_offset = [bs_idx, 0]

        # dynamic per_token_quant
        hidden_states_quant, hidden_states_scale = symmetric_quantization_per_token(tile_hidden_states)

        # up_proj的matmul计算
        pypto.set_cube_tile_shapes([tile_batch, tile_batch],
                                [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2],
                                [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True)
        up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)

        # dequant
        w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
        pypto.set_vec_tile_shapes(8, intermediate_size * 2)
        up_proj_dequant = dequant_dynamic(up_proj, w13_scale_2d, hidden_states_scale)
        swiglu_out = swiglu(up_proj_dequant)

        # dynamic per_token_quant
        down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

        # down_proj
        pypto.set_cube_tile_shapes([tile_batch, tile_batch],
                                [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2],
                                [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], False)
        down_proj = pypto.matmul(down_proj_quant, w2, pypto.DT_INT32)

        # dequant
        w2_scale_2d = pypto.unsqueeze(w2_scale, 0)
        pypto.set_vec_tile_shapes(4, hidden_size)
        down_proj_dequant = dequant_dynamic(down_proj, w2_scale_2d, down_proj_scale)
        out = pypto.cast(down_proj_dequant, hidden_states.dtype)
        pypto.assemble(out, hidden_states_offset, ffn_res)


def test_moe_fusion():
    # 1. 设置参数
    enable_graph = False
    ne = 160
    h_num = 5120

    top_k = 8
    topk_group = 1
    num_expert_group = 1
    renormalize = True

    x_dtype = torch.bfloat16
    intermediate_size = 192
    hidden_size = h_num

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    torch.manual_seed(0)
    for bs in [32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]:
        # 3. 准备测试数据
        hidden_states = torch.rand((bs, hidden_size), dtype=x_dtype, device=f'npu:{device_id}') * 0.05
        weight_gate_upper_tensor = torch.rand((hidden_size, intermediate_size * 2),
                                            dtype=x_dtype, device=f'npu:{device_id}') * 0.05
        w13, w13_scale = gen_quan_per_channel_weight_nz(weight_gate_upper_tensor)
        w13_scale = w13_scale.reshape(-1).to(x_dtype)
        weight_down_proj_tensor = torch.rand((intermediate_size, hidden_size),
                                            dtype=x_dtype, device=f'npu:{device_id}') * 0.05
        w2, w2_scale = gen_quan_per_channel_weight_nz(weight_down_proj_tensor)
        w2_scale = w2_scale.reshape(-1).to(x_dtype)
        ffn_res = torch.empty((bs, hidden_size), dtype=x_dtype, device=f'npu:{device_id}')

        mm_weight = torch.rand((ne, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        e_score_bias = torch.rand(
            (ne), dtype=torch.bfloat16, device=f'npu:{device_id}')
        topk_weights = torch.empty(
            (bs, top_k), dtype=torch.float32, device=f'npu:{device_id}')
        topk_ids = torch.empty(
            (bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = [
            mm_weight,
            hidden_states,
            top_k,
            renormalize,
            topk_group,
            num_expert_group,
            e_score_bias,
            w13,
            w13_scale,
            w2,
            w2_scale
        ]
        outputs = [
            topk_weights,
            topk_ids,
            ffn_res
        ]

        if enable_graph:
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                moe_fusion(*inputs, *outputs)
            g.replay()
        else:
            moe_fusion(*inputs, *outputs)

        # 5. 与PyTorch参考实现对比
        result = torch.matmul(hidden_states.to(torch.float32), mm_weight.to(torch.float32).t())
        router_logits_fp32 = result.to(torch.float)
        original_weights = router_logits_fp32.sigmoid()
        bias_2d = e_score_bias.unsqueeze(0)
        topk_weights_g_add = original_weights + bias_2d
        tw_view = topk_weights_g_add.view(bs, num_expert_group, -1)
        grouped_weights = tw_view.max(dim=-1).values

        topk_group_indices_g = torch.topk(grouped_weights.to(torch.float32),
                                          k=topk_group,
                                          dim=-1,
                                          sorted=False)[1]
        topk_group_mask = torch.zeros_like(grouped_weights)

        topk_group_mask.scatter_(1, topk_group_indices_g, 1)
        tgm_unsquee = topk_group_mask.unsqueeze(-1)
        tgm_expand = tgm_unsquee.expand(bs, num_expert_group, ne // num_expert_group)
        topk_weight_mask = tgm_expand.reshape(bs, -1)
        logical_not_tmp = ~topk_weight_mask.bool()
        topk_weights_fill = topk_weights_g_add.masked_fill(
            logical_not_tmp, 0.0)

        topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32), k=top_k, dim=-1, sorted=False)[1]
        topk_ids_int32 = topk_ids_int64.to(torch.int32)

        topk_weights_gather = original_weights.gather(1, topk_ids_int64)

        if renormalize:
            topk_weights_out = topk_weights_gather / topk_weights_gather.sum(dim=-1, keepdim=True)
        else:
            topk_weights_out = topk_weights_gather

        topk_weight_2_tensor_list = topk_weights_out.cpu().flatten().tolist()
        topk_ids_tensor_list = topk_ids_int32.cpu().flatten().tolist()

        # begin share_experts
        x_dtype = hidden_states.dtype
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
        output_w13 = torch_npu.npu_quant_matmul(
            quantized_x,
            w13,
            w13_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=x_dtype
        )
        swiglu_out = torch_npu.npu_swiglu(output_w13)
        quantized_x, x_scale = torch_npu.npu_dynamic_quant(swiglu_out)
        golden = torch_npu.npu_quant_matmul(
            quantized_x,
            w2,
            w2_scale,
            pertoken_scale=x_scale,
            bias=None,
            output_dtype=x_dtype
        )

        # weight result
        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()), np.array(topk_weight_2_tensor_list),
                        rtol=5e-3, atol=5e-3)

        # idx result
        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()), np.array(topk_ids_tensor_list),
                        rtol=5e-3, atol=5e-3)

        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)

        # 获取编译总耗时
        import pypto.pypto_impl as pypto_impl
        total_elapsed = pypto_impl.GetCompilerMonitorTotalElapsed()
        check_cond(total_elapsed <= 30, f"glm_moe_fusion compile elapsed timeout {total_elapsed}s > 30s.")


@allow_in_graph
def moe_fusion(
        gate_weight: torch.Tensor,  # gate matmul weights
        hidden_states: torch.Tensor,  # Hidden states of shape (num_tokens, hidden_size).
        top_k: int,  # number of top k experts.
        renormalize: bool,
        topk_group: int,
        num_expert_group: int,
        e_score_bias: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,

        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ffn_res: torch.Tensor
):
    if isinstance(hidden_states, FakeTensor):
        return
    check_args(gate_weight, hidden_states, top_k, renormalize, topk_group, num_expert_group,
                e_score_bias, w13, w13_scale, w2, w2_scale)

    bs = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inputs = [hidden_states, gate_weight, e_score_bias, w13, w13_scale,
                w2, w2_scale, topk_weights, topk_ids, ffn_res]

    moe_fusion_kernel(*inputs, topk_group, num_expert_group)


def moe_fusion_pto(gate_layer, hidden_states, share_layer, top_k, renormalize, topk_group=None, num_expert_group=None,
                   e_score_correction_bias=None):
    bs = hidden_states.shape[0]
    ne = gate_layer.weight.shape[0]
    device_info = hidden_states.device
    topk_weights = torch.empty((bs, top_k), dtype=torch.float32, device=device_info)
    topk_ids = torch.empty((bs, top_k), dtype=torch.int32, device=device_info)

    ffn_res = torch.empty_like(hidden_states, device=device_info)
    w13_int8 = share_layer.gate_up_proj.weight
    w13_scale = share_layer.gate_up_proj.weight_scale
    w2_int8 = share_layer.down_proj.weight
    w2_scale = share_layer.down_proj.weight_scale

    moe_fusion(
        gate_layer.weight,
        hidden_states,
        top_k,
        renormalize,
        topk_group,
        num_expert_group,
        e_score_correction_bias,
        w13_int8,
        w13_scale,
        w2_int8,
        w2_scale,
        topk_weights,
        topk_ids,
        ffn_res
    )
    return topk_weights, topk_ids, ffn_res


def main():
    pypto.set_host_options(compile_monitor_enable=True,
        compile_timeout=10,
        compile_timeout_stage=5,
        compile_monitor_print_interval=2)
    test_moe_fusion()


if __name__ == "__main__":
    main()
