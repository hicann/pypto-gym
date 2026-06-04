#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pypto
from conftest import duration_estimate


SHAPE_DIM_0 = 0
SHAPE_DIM_1 = 1
SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3
NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_24 = 24
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_512 = 512
NUM_1024 = 1024
NUM_1127 = 1127
NUM_1536 = 1536
NUM_7168 = 7168

KEY_CUBE_NBUFFER_SETTING = "CUBE_NBUFFER_SETTING"
KEY_MG_COPYIN_UPPER_BOUND = "MG_COPYIN_UPPER_BOUND"


@dataclass
class MlaQuantInputs:
    dequant_scale_x: Optional[pypto.Tensor] = None
    dequant_scale_w_dq: Optional[pypto.Tensor] = None
    dequant_scale_w_uq_qr: Optional[pypto.Tensor] = None
    dequant_scale_w_dkv_kr: Optional[pypto.Tensor] = None
    quant_scale_ckv: Optional[pypto.Tensor] = None
    quant_scale_ckr: Optional[pypto.Tensor] = None
    smooth_scales_cq: Optional[pypto.Tensor] = None


@dataclass
class RopeTileShapeConfig:
    rope_2d: List[int]
    rope_3d_vals: List[int]
    rope_4d: List[int]


@dataclass
class MlaTileConfig:
    tile_b: int = NUM_8
    tile_s: int = NUM_1
    rope: RopeTileShapeConfig = field(
        default_factory=lambda: RopeTileShapeConfig(
            rope_2d=[NUM_128, NUM_128],
            rope_3d_vals=[NUM_32, NUM_128, NUM_128],
            rope_4d=[NUM_16, NUM_128, NUM_128, NUM_128],
        )
    )


@dataclass
class MlaParams:
    b: int
    s1: int
    n1: int
    n2: int
    h: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    rope_dim: int
    cache_mode: str
    block_size: int
    block_num: int
    eps_cq: float
    eps_ckv: float
    tiles: MlaTileConfig

    @staticmethod
    def common() -> "MlaParams":
        return MlaParams(
            b=NUM_0,
            s1=NUM_0,
            n1=NUM_0,
            n2=NUM_0,
            h=NUM_7168,
            q_lora_rank=NUM_1536,
            kv_lora_rank=NUM_512,
            qk_rope_head_dim=NUM_64,
            qk_nope_head_dim=NUM_128,
            rope_dim=NUM_64,
            cache_mode="PA_BSND",
            block_size=NUM_128,
            block_num=NUM_0,
            eps_cq=1e-5,
            eps_ckv=1e-5,
            tiles=MlaTileConfig(),
        )


@dataclass
class MlaTensors:
    x: pypto.Tensor
    w_dq: pypto.Tensor
    w_uq_qr: pypto.Tensor
    w_uk: pypto.Tensor
    w_dkv_kr: pypto.Tensor
    gamma_cq: pypto.Tensor
    gamma_ckv: pypto.Tensor
    sin: pypto.Tensor
    cos: pypto.Tensor
    cache_index: pypto.Tensor
    kv_cache: pypto.Tensor
    kr_cache: pypto.Tensor
    q_out: pypto.Tensor
    q_rope_out: pypto.Tensor
    kv_cache_out: pypto.Tensor
    kr_cache_out: pypto.Tensor
    rms_res: pypto.Tensor


@dataclass
class MlaArgs:
    params: MlaParams
    tensors: MlaTensors
    quant: MlaQuantInputs


def quantize(
    input_x: pypto.Tensor,
    is_symmetry: bool = True,
    has_smooth: bool = False,
    smooth_factor: pypto.Tensor = None,
) -> Tuple[pypto.Tensor, pypto.Tensor]:

    x = pypto.cast(input_x, pypto.DT_FP32)

    if has_smooth:
        x = x * smooth_factor

    ones = pypto.tensor(1.0, pypto.DT_FP32)
    z127 = pypto.tensor(127.0, pypto.DT_FP32)
    z255 = pypto.tensor(255.0, pypto.DT_FP32)
    eps = pypto.tensor(1e-12, pypto.DT_FP32)

    if is_symmetry:
        abs_x = pypto.abs(x)
        max_val = pypto.amax(abs_x, -1)
        scale_quant = z127 / max_val
        out_fp32 = x / scale_quant
        out_int_32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int_32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int_8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)
        scale_dequant = ones / scale_quant

        return out_int_8, scale_dequant
    else:
        max_v = pypto.amax(x, -1)
        min_v = pypto.amin(x, -1)
        scale_dequant = pypto.maximum((max_v - min_v) / z255, eps)

        scale_quant = ones / scale_dequant
        out_fp32 = x * scale_quant

        out_int_32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int_32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int_8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)

        return out_int_8, scale_dequant


def dequant(
    dtype: pypto.DataType,
    input_tensor: pypto.Tensor,
    scale: pypto.Tensor,
    w_scale: pypto.Tensor,
) -> pypto.Tensor:
    out = pypto.cast(input_tensor, pypto.DT_FP32)
    out = out * scale
    out = out * w_scale
    return pypto.cast(out, dtype)


def rotate_half(x: pypto.Tensor) -> pypto.Tensor:
    shape = x.shape
    nd = x.dim

    assert nd >= 1
    assert shape[nd - 1] % NUM_2 == 0

    new_shape = list(shape)
    new_shape[nd - 1] //= NUM_2

    off1 = [0] * nd
    off2 = [0] * nd
    off2[nd - 1] = new_shape[nd - 1]

    x1 = pypto.view(x, new_shape, off1)
    x2 = pypto.view(x, new_shape, off2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_2d(
    x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, tiles: RopeTileShapeConfig
) -> pypto.Tensor:
    assert x.dim == SHAPE_DIM_2 and cos.dim == SHAPE_DIM_2 and sin.dim == SHAPE_DIM_2
    seq = x.shape[NUM_0]
    d_r = x.shape[NUM_1]
    x_dtype = x.dtype

    pypto.set_vec_tile_shapes(tiles.rope_2d[0], tiles.rope_2d[1])
    cast_x = pypto.cast(x, pypto.DT_FP32)
    if x.dtype == pypto.DT_FP32:
        cast_x = cast_x + 0.0

    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    x_view = pypto.reshape(cast_x, [1, seq, d_r // NUM_2, NUM_2])
    pypto.set_vec_tile_shapes(*tiles.rope_4d)
    x_t = pypto.transpose(x_view, NUM_2, NUM_3)
    x_flat = pypto.reshape(x_t, [seq, d_r])

    pypto.set_vec_tile_shapes(tiles.rope_2d[0], tiles.rope_2d[1])
    if not (x.shape[0] == cos.shape[0] and x.shape[1] == cos.shape[1]):
        cast_cos[:] = pypto.expand(cast_cos, x.shape)
        cast_sin[:] = pypto.expand(cast_sin, x.shape)

    x_embed = x_flat * cast_cos + rotate_half(x_flat) * cast_sin

    return pypto.cast(x_embed, x_dtype)


def rope_3d(
    x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, tiles: RopeTileShapeConfig
) -> pypto.Tensor:
    assert x.dim == SHAPE_DIM_3 and cos.dim == SHAPE_DIM_2 and sin.dim == SHAPE_DIM_2
    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128)
    cast_x = pypto.cast(x, pypto.DT_FP32)

    if x.dtype == pypto.DT_FP32:
        cast_x = cast_x + 0.0

    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    cast_cos = pypto.reshape(cast_cos, [x.shape[NUM_0], 1, x.shape[NUM_2]])
    cast_sin = pypto.reshape(cast_sin, [x.shape[NUM_0], 1, x.shape[NUM_2]])

    x_view = pypto.reshape(
        cast_x, [x.shape[NUM_0], x.shape[NUM_1], x.shape[NUM_2] // NUM_2, NUM_2]
    )
    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_t = pypto.transpose(x_view, NUM_2, NUM_3)
    x_back = pypto.reshape(x_t, x.shape)

    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_embed = x_back * cast_cos + rotate_half(x_back) * cast_sin
    return pypto.cast(x_embed, x.dtype)


def pre_compute_2d(
    x_bs: pypto.Tensor, tens: MlaTensors, quant_inputs: MlaQuantInputs, eps_cq: float
) -> Tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:

    w_dq = tens.w_dq
    w_uq_qr = tens.w_uq_qr
    w_dkv_kr = tens.w_dkv_kr
    gamma_cq = tens.gamma_cq

    dq_w_scale = quant_inputs.dequant_scale_w_dq
    dkv_w_scale = quant_inputs.dequant_scale_w_dkv_kr
    uq_w_scale = quant_inputs.dequant_scale_w_uq_qr

    smooth_scales_cq = quant_inputs.smooth_scales_cq
    is_smooth = smooth_scales_cq is not None

    is_quant_a = (dq_w_scale is not None) and (dkv_w_scale is not None)
    is_quant_b = uq_w_scale is not None

    bs = x_bs.shape[SHAPE_DIM_0]
    q_rank = w_dq.shape[SHAPE_DIM_1]

    dtype = x_bs.dtype
    dtype_a = pypto.DT_INT32 if is_quant_a else dtype
    dtype_b = pypto.DT_INT32 if is_quant_b else dtype

    pypto.set_semantic_label("pre_reshape")

    c0 = NUM_16
    m = (min(NUM_32, bs) + c0 - 1) // c0 * c0
    mv = min(NUM_8, bs)
    q_a_proj = pypto.tensor()

    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_rank)
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
        pypto.set_semantic_label("Quant_x")
        quant_res = quantize(x_bs)
        x_q, x_q_scale = quant_res[0], quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qa")
        q_a_proj[:] = pypto.matmul(x_q, w_dq, dtype_a)
        pypto.set_semantic_label("Dequant_qa")
        q_a_proj[:] = dequant(dtype, q_a_proj, x_q_scale, dq_w_scale)
    else:
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
        pypto.set_semantic_label("Matmul_qa")
        q_a_proj[:] = pypto.matmul(x_bs, w_dq, dtype)

    pypto.set_vec_tile_shapes(mv, q_rank)
    pypto.set_semantic_label("RmsNorm_qa")
    q_rms = pypto.rms_norm(q_a_proj, gamma_cq, eps_cq)

    q_b_proj = pypto.tensor()
    if is_quant_b:
        pypto.set_vec_tile_shapes(mv, q_rank)
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
        pypto.set_semantic_label("Quant_qMmRes")
        if is_smooth:
            quant_res = quantize(q_rms, True, True, smooth_scales_cq)
        else:
            quant_res = quantize(q_rms, True, False)

        q_q, q_q_scale = quant_res[0], quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qb")
        q_b_proj[:] = pypto.matmul(q_q, w_uq_qr, dtype_b)
        pypto.set_semantic_label("Dequant_qb")
        q_b_proj[:] = dequant(dtype, q_b_proj, q_q_scale, uq_w_scale)
    else:
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
        pypto.set_semantic_label("Matmul_qb")
        q_b_proj[:] = pypto.matmul(q_rms, w_uq_qr, dtype)

    compressed_kv = pypto.tensor()
    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_rank)
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_256, NUM_256])
        pypto.set_semantic_label("QuantMatmul_kva")
        compressed_kv[:] = pypto.matmul(x_q, w_dkv_kr, dtype_a)
        pypto.set_semantic_label("Dequant_kva")
        compressed_kv[:] = dequant(dtype, compressed_kv, x_q_scale, dkv_w_scale)
    else:
        pypto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
        pypto.set_semantic_label("Matmul_kva")
        compressed_kv[:] = pypto.matmul(x_bs, w_dkv_kr, dtype)

    return q_b_proj, compressed_kv, q_rms


def mla_prolog_compute(args: MlaArgs):
    p = args.params
    t = args.tensors
    quant_inputs = args.quant
    tiles = p.tiles

    assert (
        len(t.x.shape) == NUM_3
        and len(t.w_uk.shape) == NUM_3
        and len(t.sin.shape) == NUM_3
    )
    assert len(t.kv_cache.shape) == NUM_4 and len(t.kr_cache.shape) == NUM_4
    assert p.cache_mode in ["PA_BSND", "PA_NZ"]

    dtype = t.x.dtype
    h = t.x.shape[SHAPE_DIM_2]
    n = t.w_uk.shape[SHAPE_DIM_0]
    q_lora_rank = t.w_dq.shape[SHAPE_DIM_1]
    qk_nope_head_dim = t.w_uk.shape[SHAPE_DIM_1]
    kv_lora_rank = t.w_uk.shape[SHAPE_DIM_2]
    qk_rope_head_dim = t.sin.shape[SHAPE_DIM_2]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    block_num = t.kv_cache.shape[SHAPE_DIM_0]
    block_size = t.kv_cache.shape[SHAPE_DIM_1]
    n2 = t.kv_cache.shape[SHAPE_DIM_2]
    assert qk_nope_head_dim == NUM_128 or qk_rope_head_dim == NUM_64

    tile_b = tiles.tile_b
    tile_s = tiles.tile_s
    tile_bs = tile_b * tile_s

    rope_cfg = p.tiles.rope
    b = t.x.shape[SHAPE_DIM_0]
    s = t.x.shape[SHAPE_DIM_1]
    bs_loop = (b * s + tile_bs - 1) // tile_bs

    x_2d = pypto.tensor([b * s, h], t.x.dtype, "x2D")
    cos_2d = pypto.tensor([b * s, qk_rope_head_dim], t.cos.dtype, "cos2D")
    sin_2d = pypto.tensor([b * s, qk_rope_head_dim], t.sin.dtype, "sin2D")
    k_cache_index_2d = pypto.tensor([b * s, 1], t.cache_index.dtype, "kCacheIndex2D")

    for _ in pypto.loop(0, 1, 1, name="LOOP_MLA_RESHAPE", idx_name="batch_id"):
        x_2d[:] = pypto.reshape(t.x, [b * s, h], inplace=True)
        cos_2d[:] = pypto.reshape(t.cos, [b * s, qk_rope_head_dim], inplace=True)
        sin_2d[:] = pypto.reshape(t.sin, [b * s, qk_rope_head_dim], inplace=True)
        k_cache_index_2d[:] = pypto.reshape(t.cache_index, [b * s, 1], inplace=True)

    kv_cache_res = pypto.tensor(
        [block_num * block_size * n2, kv_lora_rank], t.kv_cache.dtype, "kvCacheRes"
    )
    kr_cache_res = pypto.tensor(
        [block_num * block_size * n2, qk_rope_head_dim], t.kr_cache.dtype, "krCacheRes"
    )

    for _ in pypto.loop(0, 1, 1, name="MLA_RESHAPE", idx_name="unused_idx"):
        kv_cache_res[:] = pypto.reshape(
            t.kv_cache, [block_num * block_size * n2, kv_lora_rank], inplace=True
        )
        kr_cache_res[:] = pypto.reshape(
            t.kr_cache, [block_num * block_size * n2, qk_rope_head_dim], inplace=True
        )

    for bs_idx in pypto.loop(0, bs_loop, 1, name="MLA_BS_Loop", idx_name="bs_idx"):
        bs_offset = bs_idx * tile_bs
        output_offset = [bs_offset, 0, 0]
        pypto.set_vec_tile_shapes(tile_bs, NUM_128)
        x_view = pypto.view(x_2d, [tile_bs, h], [bs_offset, 0])
        x_view[:] = pypto.cast(pypto.cast(x_view, pypto.DT_FP32), dtype)

        q, kv_tmp, q_rms = pre_compute_2d(
            x_view,
            t,
            quant_inputs,
            p.eps_cq,
        )
        q_tmp = pypto.reshape(q, [tile_bs, n, q_head_dim])

        pypto.set_semantic_label("Prepare_qNope")
        q_nope = pypto.view(q_tmp, [tile_bs, n, qk_nope_head_dim], [0, 0, 0])
        tile_shape = [min(32, tile_bs), 1, qk_nope_head_dim]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_trans = pypto.transpose(q_nope, 0, 1)

        pypto.set_semantic_label("pre_reshape")

        c0 = NUM_16
        m = (min(NUM_32, tile_bs) + c0 - 1) // c0 * c0
        pypto.set_semantic_label("Matmul_qNope_wUk")
        pypto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128])
        q_nope_new = pypto.matmul(q_nope_trans, t.w_uk, dtype)

        pypto.set_semantic_label("queryOut")
        tile_shape = [NUM_1, min(NUM_32, tile_bs), kv_lora_rank]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_new_trans = pypto.transpose(q_nope_new, 0, 1)
        pypto.set_semantic_label("Assemble_queryOut")
        pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128)
        pypto.assemble(q_nope_new_trans, output_offset, t.q_out)

        q_pe_view = pypto.view(
            q_tmp, [tile_bs, n, qk_rope_head_dim], [0, 0, qk_nope_head_dim]
        )
        cos_2d[:] = pypto.view(cos_2d, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        sin_2d[:] = pypto.view(sin_2d, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        q_rope_view = rope_3d(q_pe_view, cos_2d, sin_2d, rope_cfg)
        pypto.set_semantic_label("Assemble_qRope")
        pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_64)
        pypto.assemble(q_rope_view, output_offset, t.q_rope_out)

        pypto.set_vec_tile_shapes(NUM_2, NUM_512)
        pypto.set_semantic_label("RotaryPosEmb")
        k_pe_view = pypto.view(
            kv_tmp, [tile_bs, qk_rope_head_dim], [0, kv_lora_rank]
        )
        k_rope_view = rope_2d(k_pe_view, cos_2d, sin_2d, rope_cfg)
        k_rope_res = pypto.reshape(k_rope_view, [tile_bs, 1, 1, qk_rope_head_dim])

        pypto.set_semantic_label("ScatterUpdate_krCache")
        tile_shape = [NUM_1, qk_rope_head_dim]
        pypto.set_vec_tile_shapes(*tile_shape)

        index = pypto.view(k_cache_index_2d, [tile_bs, 1], [bs_offset, 0])
        pypto.set_vec_tile_shapes(NUM_4, NUM_128, NUM_128, NUM_128)
        t.kr_cache_out[:] = pypto.scatter_update(t.kr_cache, -2, index, k_rope_res)

        compressed_kv = pypto.view(kv_tmp, [tile_bs, kv_lora_rank], [0, 0])
        tile_shape = [NUM_2, NUM_512]
        pypto.set_semantic_label("RmsNorm_compressedKv")
        pypto.set_vec_tile_shapes(*tile_shape)
        k_nope = pypto.rms_norm(compressed_kv, t.gamma_ckv, p.eps_ckv)
        k_nope[:] = pypto.reshape(k_nope, [tile_bs, 1, 1, kv_lora_rank])

        pypto.set_semantic_label("ScatterUpdate_kvCache")
        pypto.set_vec_tile_shapes(NUM_4, NUM_128, NUM_128, NUM_512)
        t.kv_cache_out[:] = pypto.scatter_update(t.kv_cache, -2, index, k_nope)

        pypto.set_vec_tile_shapes(tile_bs, q_lora_rank)
        rms_3d = pypto.cast(pypto.cast(q_rms, pypto.DT_FP32), dtype)
        pypto.assemble(rms_3d, [bs_offset, 0], t.rms_res)


def mla_prolog(args: MlaArgs):
    inp = [
        args.tensors.x,
        args.tensors.w_dq,
        args.tensors.w_uq_qr,
        args.tensors.w_uk,
        args.tensors.w_dkv_kr,
        args.tensors.gamma_cq,
        args.tensors.gamma_ckv,
        args.tensors.sin,
        args.tensors.cos,
        args.tensors.cache_index,
        args.tensors.kv_cache,
        args.tensors.kr_cache,
    ]
    out = [
        args.tensors.q_out,
        args.tensors.q_rope_out,
        args.tensors.kv_cache_out,
        args.tensors.kr_cache_out,
        args.tensors.rms_res,
    ]
    with pypto.function("MLAProlog", *inp, *out):
        mla_prolog_compute(args)


@dataclass
class MlaBuildConfig:
    b: int = NUM_4
    s1: int = NUM_2
    n1: int = NUM_128
    n2: int = NUM_1
    h: int = NUM_7168
    q_lora_rank: int = NUM_1536
    kv_lora_rank: int = NUM_512
    qk_rope_head_dim: int = NUM_64
    qk_nope_head_dim: int = NUM_128
    rope_dim: int = NUM_64
    cache_mode: str = "PA_BSND"
    block_size: int = NUM_128
    block_num: int = NUM_1127
    eps_cq: float = 1e-5
    eps_ckv: float = 1e-5
    tile_b_override: int = NUM_NEG1
    tile_s: int = NUM_1
    rope_2d: List[int] = field(default_factory=lambda: [NUM_128, NUM_128])
    rope_3d_vals: List[int] = field(default_factory=lambda: [NUM_32, NUM_128, NUM_128])
    rope_4d: List[int] = field(
        default_factory=lambda: [NUM_16, NUM_128, NUM_128, NUM_128]
    )


def setup_codegen_passes():
    pypto.set_pass_options(
        cube_l1_reuse_setting={-1: NUM_4},
        cube_nbuffer_setting={NUM_3: NUM_4},
    )


def build_args(cfg: MlaBuildConfig):
    tile_b = (
        cfg.tile_b_override
        if cfg.tile_b_override != NUM_NEG1
        else (NUM_8 if cfg.b == NUM_24 else cfg.b)
    )
    tiles = MlaTileConfig(
        tile_b=tile_b,
        tile_s=cfg.tile_s,
        rope=RopeTileShapeConfig(
            rope_2d=cfg.rope_2d,
            rope_3d_vals=cfg.rope_3d_vals,
            rope_4d=cfg.rope_4d,
        ),
    )
    params = MlaParams(
        b=cfg.b,
        s1=cfg.s1,
        n1=cfg.n1,
        n2=cfg.n2,
        h=cfg.h,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        rope_dim=cfg.rope_dim,
        cache_mode=cfg.cache_mode,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        eps_cq=cfg.eps_cq,
        eps_ckv=cfg.eps_ckv,
        tiles=tiles,
    )
    d_fp16 = pypto.DT_FP16
    d_int32 = pypto.DT_INT32
    x = pypto.tensor([cfg.b, cfg.s1, cfg.h], d_fp16, "x")

    w_dq = pypto.tensor(
        [cfg.h, cfg.q_lora_rank], d_fp16, "wDq", pypto.TileOpFormat.TILEOP_ND
    )
    w_uq_qr = pypto.tensor(
        [cfg.q_lora_rank, cfg.n1 * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)],
        d_fp16,
        "wUqQr",
        pypto.TileOpFormat.TILEOP_ND,
    )
    w_dkv_kr = pypto.tensor(
        [cfg.h, cfg.kv_lora_rank + cfg.qk_rope_head_dim],
        d_fp16,
        "wDkvKr",
        pypto.TileOpFormat.TILEOP_ND,
    )
    w_uk = pypto.tensor(
        [cfg.n1, cfg.qk_nope_head_dim, cfg.kv_lora_rank],
        d_fp16,
        "wUk",
        pypto.TileOpFormat.TILEOP_ND,
    )
    gamma_cq = pypto.tensor(
        [cfg.q_lora_rank], d_fp16, "gammaCq", pypto.TileOpFormat.TILEOP_ND
    )
    gamma_ckv = pypto.tensor(
        [cfg.kv_lora_rank], d_fp16, "gammaCkv", pypto.TileOpFormat.TILEOP_ND
    )
    cos = pypto.tensor([cfg.b, cfg.s1, cfg.qk_rope_head_dim], d_fp16, "cos")
    sin = pypto.tensor([cfg.b, cfg.s1, cfg.qk_rope_head_dim], d_fp16, "sin")
    cache_index = pypto.tensor([cfg.b, cfg.s1], d_int32, "cacheIndex")
    kv_cache = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, cfg.kv_lora_rank], d_fp16, "kvCache"
    )
    kr_cache = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, cfg.qk_rope_head_dim], d_fp16, "krCache"
    )
    q_out = pypto.tensor(
        [cfg.b * cfg.s1, cfg.n1, cfg.kv_lora_rank], d_fp16, "queryNopeOut"
    )
    q_rope_out = pypto.tensor(
        [cfg.b * cfg.s1, cfg.n1, cfg.qk_rope_head_dim], d_fp16, "queryRopeOut"
    )
    kv_cache_out = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, cfg.kv_lora_rank], d_fp16, "kvCacheOut"
    )
    kr_cache_out = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, cfg.qk_rope_head_dim],
        d_fp16,
        "krCacheOut",
    )
    rms_res = pypto.tensor(
        [cfg.b * cfg.s1, cfg.q_lora_rank],
        d_fp16,
        "rmsRes",
    )

    tensors = MlaTensors(
        x=x,
        w_dq=w_dq,
        w_uq_qr=w_uq_qr,
        w_uk=w_uk,
        w_dkv_kr=w_dkv_kr,
        gamma_cq=gamma_cq,
        gamma_ckv=gamma_ckv,
        sin=sin,
        cos=cos,
        cache_index=cache_index,
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        q_out=q_out,
        q_rope_out=q_rope_out,
        kv_cache_out=kv_cache_out,
        kr_cache_out=kr_cache_out,
        rms_res=rms_res,
    )

    return MlaArgs(params=params, tensors=tensors, quant=MlaQuantInputs())


@duration_estimate(12)
def test_dynamic_mla_prolog():
    args = build_args(MlaBuildConfig())
    setup_codegen_passes()
    mla_prolog(args)
    assert True


if __name__ == "__main__":
    test_dynamic_mla_prolog()
