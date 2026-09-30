# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent K/V lifetimes across two cube products and one vector cast."""

import ascriptor.a5 as api


def make_kernel(policy, *, width=64, items=3, beats=5, dtype_name="float16", fault=None):
    """Keep the K-load dependency explicit so its omitted edge is testable.

    The model-only faults are not selectable by any public unit case. The large
    K2/V2/P2 factory is also a capacity negative, not a supported case.
    """
    if policy not in ("k2v2p2", "k1v2p2", "sequential") or width not in (64, 256):
        raise ValueError("Require a declared slot policy and width 64 or 256")
    if dtype_name not in ("float16", "bfloat16") or items < 1 or beats < 1:
        raise ValueError("Require b16 storage and positive items/beats")
    if fault not in (None, "missing_k_available", "reuse_v_early"):
        raise ValueError("Unknown model-only ownership fault")
    sequential = policy == "sequential"
    k_slots = 2 if policy == "k2v2p2" else 1
    delayed_slots = 1 if sequential else 2
    v_slots = 1 if fault == "reuse_v_early" else delayed_slots
    k_buffer = api.DBuff if k_slots == 2 else api.Tensor
    v_buffer = api.DBuff if v_slots == 2 else api.Tensor
    p_buffer = api.DBuff if delayed_slots == 2 else api.Tensor
    k_event = api.DEvent if k_slots == 2 else api.SEvent
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    iterations = beats if sequential else beats + 1

    @api.vf()
    def cast_probability(source: api.Tensor, destination: api.Tensor):
        value = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(8):
            for chunk in api.unroll(width // 64):
                value <<= source[row * width + chunk * 64]
                packed <<= value.astype(dtype)
                # FP32->b16 cast uses sparse ZERO lanes; compact all 64 values.
                api.reg_to_ub_downsample(destination[row * width + chunk * 64], packed)

    @api.func()
    def produce(query, key, probability, score, ub_score, ub_p, qk, published,
                available, ready, lane_begin):
        ready.wait()
        api.matmul(score, query, key, m=16, n=width, k=width, splitk=64)
        if fault != "missing_k_available":
            # MTE1 is done reading K after every QK operand fragment was staged.
            available.set()
        qk.lock()
        ub_score <<= score
        qk.ready()
        qk.wait()
        cast_probability(ub_score, ub_p)
        qk.free()
        published.lock()
        probability[lane_begin:lane_begin + 8, :] <<= ub_p
        published.ready()

    @api.func()
    def consume(value, probability, product, ub_product, published, result,
                output, item, beat, lane_begin):
        published.wait()
        api.matmul(product, probability, value.T, m=16, n=width, k=width, splitn=64)
        # P is retired at its final L1->L0A read, independently of the result UB.
        published.free()
        result.lock()
        ub_product <<= product
        result.ready()
        result.wait()
        output[item, beat, lane_begin:lane_begin + 8, :] <<= ub_product
        # The result lease includes the real MTE3 reader, including on drain.
        result.free()

    @api.kernel(mode="mix")
    def independent_kv(q: api.GM[dtype, (items, 16, width)],
                       k: api.GM[dtype, (items, beats, width, width)],
                       v: api.GM[dtype, (items, beats, width, width)],
                       o: api.GM[api.f32, (items, beats, 32, width)]):
        query = api.Tensor(dtype, [16, width], api.Position.L1)
        keys = k_buffer(dtype, [width, width], api.Position.L1)
        values = v_buffer(dtype, [width, width], api.Position.L1)
        probabilities = p_buffer(dtype, [16, width], api.Position.L1)
        score = api.Tensor(api.f32, [16, width], api.Position.L0C)
        product = api.Tensor(api.f32, [16, width], api.Position.L0C)
        ub_score = api.Tensor(api.f32, [8, width], api.Position.UB)
        ub_p = api.Tensor(dtype, [8, width], api.Position.UB)
        ub_product = api.Tensor(api.f32, [8, width], api.Position.UB)
        k_ready = k_event(api.Pipe.MTE2, api.Pipe.MTE1)
        k_available = k_event(api.Pipe.MTE1, api.Pipe.MTE2, preset=True)
        qk = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        published = api.VcMutex(1, depth=delayed_slots, src_end_pipe=api.Pipe.MTE3,
                                dst_end_pipe=api.Pipe.MTE1)
        result = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.MTE3)
        lane_begin = api.Var(api.GetSubBlockIdx() * 8)
        for item in range(api.GetCubeIdx(), items, api.GetCubeNum()):
            with api.auto_sync():
                query <<= q[item, :, :]
            for tick in range(iterations):
                has_current = tick < beats
                if has_current:
                    if k_slots == 2:
                        incoming_key = keys[tick]
                    else:
                        incoming_key = keys
                    # This load intentionally stays outside autosync. Its ready
                    # and final-reader availability edges are both explicit.
                    if fault != "missing_k_available":
                        k_available.wait()
                    incoming_key <<= k[item, tick, :, :]
                    k_ready.set()
                with api.auto_sync():
                    if has_current:
                        if v_slots == 2:
                            incoming_value = values[tick]
                        else:
                            incoming_value = values
                        incoming_value <<= v[item, tick, :, :]
                    if not sequential:
                        if tick > 0:
                            previous = api.Var(tick - 1)
                            if v_slots == 2:
                                previous_value = values[previous]
                            else:
                                previous_value = values
                            consume(previous_value, probabilities[previous], product, ub_product,
                                    published, result, o, item, previous, lane_begin)
                    if has_current:
                        if k_slots == 2:
                            current_key = keys[tick]
                        else:
                            current_key = keys
                        if delayed_slots == 2:
                            current_p = probabilities[tick]
                        else:
                            current_p = probabilities
                        produce(query, current_key, current_p, score, ub_score, ub_p, qk,
                                published, k_available, k_ready, lane_begin)
                        if sequential:
                            consume(values, probabilities, product, ub_product, published,
                                    result, o, item, tick, lane_begin)
        return o

    return independent_kv
