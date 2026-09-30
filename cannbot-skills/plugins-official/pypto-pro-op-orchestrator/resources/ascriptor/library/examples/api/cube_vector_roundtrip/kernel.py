# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two directional ownership transfers with a17-row compact-NZ FP16 pitch."""

import ascriptor.a5 as api


@api.vf()
def preprocess(source: api.Tensor, packed: api.Tensor):
    value = api.Reg(api.f16)
    for row in range(16):
        value <<= source[row:, 0:128]
        value <<= value * 2.0
        api.reg_to_ub(packed[row * 16], value, 17)


@api.vf()
def postprocess(source: api.Tensor, output: api.Tensor):
    value = api.Reg(api.f32)
    for chunk in range(4):
        value <<= source[chunk * 64]
        value <<= value.abs() + 1.0
        output[chunk * 64] <<= value


@api.kernel(mode="mix", block_dim=1)
def roundtrip(x: api.GM[api.f16, (3, 32, 128)], y: api.GM[api.f16, (16, 128)],
    o: api.GM[api.f32, (3, 32, 16)]):
    incoming = api.DBuff(api.f16, [16, 128], api.Position.UB)
    packed = api.DBuff(api.f16, [17, 128], api.Position.UB)
    left = api.DBuff(api.f16, [32, 128], api.Position.L1)
    right = api.Tensor(api.f16, [16, 128], api.Position.L1)
    product = api.DBuff(api.f32, [32, 16], api.Position.L0C)
    middle = api.DBuff(api.f32, [16, 16], api.Position.UB)
    outgoing = api.DBuff(api.f32, [16, 16], api.Position.UB)
    vector_to_cube = api.VcMutex(0, guards=left, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.FIX)
    cube_to_vector = api.CvMutex(1, guards=middle, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    with api.auto_sync():
        right <<= y
        for beat in range(3):
            vector_to_cube.lock()
            begin = api.GetSubBlockIdx() * 16
            incoming[beat] <<= x[beat, begin : begin + 16, :]
            preprocess(incoming[beat], packed[beat])
            left[beat][begin : begin + 16, :] <<= packed[beat][0:16, :].nz()
            vector_to_cube.ready()

            vector_to_cube.wait()
            api.matmul(product[beat], left[beat], right, m=32, n=16, k=128)
            cube_to_vector.lock()
            middle[beat] <<= product[beat]
            cube_to_vector.ready()
            vector_to_cube.free()

            cube_to_vector.wait()
            postprocess(middle[beat], outgoing[beat])
            cube_to_vector.free()
            o[beat, begin : begin + 16, :] <<= outgoing[beat]
    return o


def make_narrow(dtype_name, live_rows, fault=None):
    """Observe four NZ columns and the full staging allocation for five beats.

    ``fault`` is a model-only negative control; public cases never select it.
    The extra four NZ columns are guards, not a larger publication footprint.
    """
    if dtype_name not in ("float16", "bfloat16") or live_rows not in (8, 16):
        raise ValueError("Narrow cases require float16/bfloat16 and 8/16 live rows")
    if fault not in (None, "missing_mask", "wrong_pitch", "missing_ready"):
        raise ValueError("Unknown model-only negative control")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    pitch = 8 if fault == "wrong_pitch" else 9

    @api.vf()
    def pack_lowhalf(source: api.Tensor, packed: api.Tensor, zero_rows: api.i32):
        value = api.Reg(dtype)
        half = api.MaskReg(dtype, init_mode=api.MaskType.LOWHALF)
        for row in range(8):
            # Each load reads 128 b16 lanes; every source row allocates that extent.
            value <<= source[row, :]
            if zero_rows != 0:
                value.fill(0)
            if fault == "missing_mask":
                api.reg_to_ub(packed[row * 16], value, pitch)
            else:
                api.reg_to_ub(packed[row * 16], value, pitch, mask=half)

    @api.kernel(mode="mix", block_dim=1)
    def narrow_roundtrip(x: api.GM[dtype, (5, 16, 128)], y: api.GM[dtype, (64, 64)],
                         initial: api.GM[dtype, (9, 128)],
                         o: api.GM[api.f32, (5, 16, 64)],
                         capture: api.GM[dtype, (5, 2, 9, 128)]):
        incoming = api.DBuff(dtype, [8, 128], api.Position.UB)
        packed = api.DBuff(dtype, [9, 128], api.Position.UB)
        left = api.DBuff(dtype, [16, 64], api.Position.L1)
        right = api.Tensor(dtype, [64, 64], api.Position.L1)
        product = api.DBuff(api.f32, [16, 64], api.Position.L0C)
        middle = api.DBuff(api.f32, [8, 64], api.Position.UB)
        vector_to_cube = api.VcMutex(0, depth=2, src_end_pipe=api.Pipe.MTE3,
                                    dst_end_pipe=api.Pipe.FIX)
        cube_to_vector = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX,
                                    dst_end_pipe=api.Pipe.MTE3)
        with api.auto_sync():
            right <<= y
            for beat in range(5):
                vector_to_cube.lock()
                participant = api.GetSubBlockIdx()
                begin = participant * 8
                incoming[beat] <<= x[beat, begin:begin + 8, :]
                # Initialize all staging bytes, including per-column padding and guards.
                packed[beat] <<= initial
                if live_rows == 8:
                    pack_lowhalf(incoming[beat], packed[beat], participant)
                else:
                    pack_lowhalf(incoming[beat], packed[beat], 0)
                capture[beat, participant, :, :] <<= packed[beat]
                left[beat][begin:begin + 8, :] <<= packed[beat][0:8, 0:64].nz()
                if fault != "missing_ready":
                    vector_to_cube.ready()

                vector_to_cube.wait()
                # All 16 rows are executed and read back, including explicitly zeroed rows.
                api.matmul(product[beat], left[beat], right, m=16, n=64, k=64)
                cube_to_vector.lock()
                middle[beat] <<= product[beat]
                cube_to_vector.ready()
                vector_to_cube.free()

                cube_to_vector.wait()
                o[beat, begin:begin + 8, :] <<= middle[beat]
                cube_to_vector.free()
        return o, capture

    return narrow_roundtrip


def make_fix_views(mode, *, slots=3, fault=None):
    """Read back every destination byte after each of two labelled FIX writes."""
    if mode not in ("contiguous", "pitched_columns", "independent_ubs") or slots not in (1, 3):
        raise ValueError("FIX controls use contiguous/pitched/independent UB and one/three slots")
    if fault not in (None, "compact_destination") or (fault and mode != "pitched_columns"):
        raise ValueError("The compact-destination mutation belongs only to the pitched model control")
    width = 128 if mode == "contiguous" else 256
    buffer = api.Tensor if slots == 1 else api.TBuff

    @api.kernel(mode="mix", block_dim=1)
    def fix_views(x: api.GM[api.f16, (5, 32, 16)], y: api.GM[api.f16, (2, 128, 16)],
                  initial: api.GM[api.f32, (16, 256)],
                  o: api.GM[api.f32, (5, 2, 2, 16, width)]):
        left = api.Tensor(api.f16, [32, 16], api.Position.L1)
        right = api.Tensor(api.f16, [128, 16], api.Position.L1)
        product = buffer(api.f32, [32, 128], api.Position.L0C)
        if mode == "independent_ubs":
            local_left = buffer(api.f32, [16, 128], api.Position.UB)
            local_right = buffer(api.f32, [16, 128], api.Position.UB)
        else:
            local = buffer(api.f32, [16, width], api.Position.UB)
        initialized = api.VcMutex(0, depth=1, src_end_pipe=api.Pipe.MTE2, dst_end_pipe=api.Pipe.FIX)
        published = api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.MTE3)
        with api.auto_sync():
            for beat in range(5):
                initialized.lock()
                if mode == "independent_ubs":
                    if slots == 1:
                        first, second = local_left, local_right
                    else:
                        first, second = local_left[beat], local_right[beat]
                    first <<= initial[:, :128]
                    second <<= initial[:, 128:256]
                else:
                    if slots == 1:
                        target = local
                    else:
                        target = local[beat]
                    target <<= initial[:, :width]
                initialized.ready()
                initialized.wait()
                left <<= x[beat, :, :]
                for part in api.unroll(2):
                    right <<= y[part, :, :]
                    if slots == 1:
                        accumulator = product
                    else:
                        accumulator = product[beat]
                    api.matmul(accumulator, left, right, m=32, n=128, k=16)
                    published.lock()
                    if mode == "independent_ubs":
                        if part == 0:
                            first <<= accumulator
                        else:
                            second <<= accumulator
                    elif mode == "contiguous":
                        target <<= accumulator
                    elif fault == "compact_destination":
                        # Model-only reproduction of the old compact alias's
                        # actual write addresses, not the legal parent pitch.
                        api.l0c_to_ub(target[:, part * 128:(part + 1) * 128], accumulator,
                                      M=32, N=128, M_src=32, N_dst=128,
                                      dual_mode=api.DualMode.SPLITM)
                    else:
                        target[:, part * 128:(part + 1) * 128] <<= accumulator
                    published.ready()
                    published.wait()
                    participant = api.GetSubBlockIdx()
                    if mode == "independent_ubs":
                        o[beat, part, participant, :, :128] <<= first
                        o[beat, part, participant, :, 128:256] <<= second
                    else:
                        o[beat, part, participant, :, :] <<= target
                    published.free()
                initialized.free()
        return o

    return fix_views
