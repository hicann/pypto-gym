#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test pypto.frontend.jit config scope behavior."""

import pypto
import torch


@pypto.frontend.jit(runtime_options={"run_mode": 1})
def kernel_with_dynamic(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    # get the global config in the kernel and verify it
    assert 1 == pypto.get_debug_options().get("runtime_debug_mode")
    assert 1 == pypto.get_debug_options().get("compile_debug_mode")

    assert True == pypto.get_codegen_options().get("support_dynamic_aligned")
    assert {1: 4} == pypto.get_pass_options().get("cube_l1_reuse_setting")

    pypto.set_vec_tile_shapes(16, 16)
    for idx in pypto.loop(a.shape[0], name="LOOP", idx_name="k"):
        temp = a[idx: idx + 1, :]
        out[idx: idx + 1, :] = temp + 1


def test_config_scope():
    pypto.set_debug_options(compile_debug_mode=1)
    pypto.set_debug_options(runtime_debug_mode=1)

    pypto.set_codegen_options(support_dynamic_aligned=True)
    pypto.set_pass_options(cube_l1_reuse_setting={1: 4})

    a = torch.ones(1, 8, dtype=torch.float32)
    out = torch.zeros(1, 8, dtype=torch.float32)

    kernel_with_dynamic(a, out)


def test_reshape_assemble():
    a = pypto.Tensor([32, 64], pypto.DT_FP32)
    out = pypto.Tensor([64, 32], pypto.DT_FP32)

    with pypto.function("main", a, out):
        pypto.set_vec_tile_shapes(16, 16)
        b = a + 1
        pypto.assemble(pypto.reshape(b, [64, 32]), [0, 0], out)


if __name__ == "__main__":
    test_config_scope()
    test_reshape_assemble()
