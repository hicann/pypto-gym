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
"""
"""
import pypto
import pypto._controller as controller


def init_tensors():
    dtype = pypto.DT_FP32
    shape = (128, 128)
    a = pypto.tensor(shape, dtype, "a")
    b = pypto.tensor(shape, dtype, "b")
    c = pypto.tensor(shape, dtype, "c")
    return a, b, c


def parallel_add_single_parallel_func(left: pypto.tensor, right: pypto.tensor, res: pypto.tensor):
    n0 = left.shape[0]
    n1 = left.shape[1]
    n2 = left.shape[2]
    for n0_idx in pypto.loop(0, 2, 1, name="N0_LOOP", idx_name="n0_loop"):
        for n1_idx in pypto.loop(0, 2, 1, name="N1_LOOP", idx_name="n1_loop", parallel=True):
            for n2_idx in pypto.loop(0, 2, 1, name="N2_LOOP", idx_name="n2_loop"):
                pypto.set_vec_tile_shapes(1, 1, 2)
                left_view = pypto.view(left, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                right_view = pypto.view(right, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                output = left_view + right_view
                pypto.assemble(output, [n0_idx, n1_idx * 2, n2_idx * 2], res)


def test_parallel_add_single_parallel():
    dtype = pypto.DT_FP32
    shape = (2, 4, 4)
    left = pypto.tensor(shape, dtype, "left")
    right = pypto.tensor(shape, dtype, "right")
    res = pypto.tensor(shape, dtype, "res")
    with pypto.function("MAIN", left, right, res):
        parallel_add_single_parallel_func(left, right, res)


def test_pto_loop_end_only():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(10):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 2):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 5):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.div(b, a))
            b.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_end_only_with_custom_name():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(10, name="LOOP"):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 5):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 3):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.div(b, a))
            b.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_start_end():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(1, 10):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 7):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 8):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.div(b, a))
            b.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_start_end_step():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(1, 10, 2):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 6):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 2):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.div(b, a))
            b.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_start_end_step_and_name():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(1, 10, 2, name="LOOP"):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 3):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 8):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.div(b, a))
            b.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_start_end_step_and_name():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(1, 10, 2, name="LOOP"):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 5):
                b.move(pypto.mul(b, a))

    assert isinstance(b, pypto.tensor)


def test_pto_loop_unroll_n_submit_before_loop():
    a, b, c = init_tensors()
    controller.reset()

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(
                1, 10, 2, name="LOOP", submit_before_loop=True
        ):

            if pypto.cond(k < 5):
                b.move(pypto.sub(b, a))
            if pypto.cond(1):
                b.move(pypto.add(b, a))
            if pypto.cond(pypto.is_loop_end(k)):
                b.move(pypto.add(b, a))

            b.move(pypto.sub(a, a))

    assert isinstance(b, pypto.tensor)


def test_loop_issue52():
    pypto.runtime._device_init()

    a = pypto.tensor((128, 128), pypto.DT_FP32, "a")
    b = pypto.tensor((128, 128), pypto.DT_FP32, "b")
    c = pypto.tensor((128, 128), pypto.DT_FP32, "c")

    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for i in pypto.loop(a.shape[0] // 16):
            for j in pypto.loop(a.shape[1] // 16):
                view_a = a[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                view_b = b[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                assert isinstance(view_a, pypto.tensor)
                assert isinstance(view_b, pypto.tensor)
                c[i * 16:, j * 16:] = view_b + view_a

    pypto.runtime._device_fini()


def test_if_true():
    A = pypto.tensor((64, 64), pypto.DT_FP32, "A")
    B = pypto.tensor((64, 64), pypto.DT_FP32, "B")

    pypto.set_semantic_label("IF_TRUE")
    with pypto.function("MAIN", A, B):
        for _ in pypto.loop(1):
            pypto.set_vec_tile_shapes(16, 16)
            if pypto.cond(True):
                B[:] = A + 2
            else:
                B[:] = A - 2


def test_loop_manual_unroll():
    pypto.runtime._device_init()
    A = pypto.tensor((-1, 64), pypto.DT_FP32, "A")
    B = pypto.tensor((-1, 64), pypto.DT_FP32, "B")

    with pypto.function("MAIN", A, B):
        pypto.set_vec_tile_shapes(64, 64)
        for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[1, 2, 4], name="A", idx_name='b'):
            tile_a = A[b * 64:(b + k) * 64, :]
            tile_a = tile_a + 2
            B[b * 64:, :] = tile_a

    pypto.runtime._device_fini()


def test_loop_manual_unroll_const():
    A = pypto.tensor((64, 64), pypto.DT_FP32, "A")
    B = pypto.tensor((64, 64), pypto.DT_FP32, "B")

    k_list = []
    pypto.runtime._device_init()
    with pypto.function("MAIN", A, B):
        pypto.set_vec_tile_shapes(64, 64)
        for _, k in pypto.loop_unroll(1, 8, unroll_list=[1, 2, 4]):
            k_list.append(k)
            B[:] = A + 1
    assert k_list == [4, 2, 1]
    pypto.runtime._device_fini()


def test_pto_auto_unroll():
    A = pypto.tensor((-1, 64), pypto.DT_FP32, "A")
    B = pypto.tensor((-1, 64), pypto.DT_FP32, "B")

    pypto.runtime._device_init()
    with pypto.function("MAIN", A, B):
        pypto.set_vec_tile_shapes(64, 64)
        for idx in pypto.loop(128, unroll_list=[1, 4]):
            ATile = A[idx * 64:(idx + 1) * 64, :]
            if pypto.cond(pypto.is_loop_begin(idx)):
                ATile = ATile + 1
            elif pypto.cond(pypto.is_loop_end(idx)):
                ATile = ATile + 2
            B[idx * 64:, 0:] = ATile + 1
    pypto.runtime._device_fini()
