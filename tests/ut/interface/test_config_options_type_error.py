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
"""Tests for frontend option validation errors."""

import pytest

import pypto
from pypto.experimental import set_operation_options


def assert_option_type_error(setter, pattern):
    with pytest.raises(Exception, match=pattern):
        setter()


@pytest.mark.parametrize(
    ("setter", "pattern"),
    [
        (
            lambda: pypto.set_pass_options(sg_set_scope="aa"),
            "Option 'pass.sg_set_scope' has invalid type. Expected int64 or tuple, but got str.",
        ),
        (
            lambda: pypto.set_pass_options(sg_set_scope=-2),
            r"Option 'pass\.sg_set_scope' scope_id -2 is out of range\. Expected -1~2147483647\.",
        ),
        (
            lambda: pypto.set_pass_options(sg_set_scope=(2147483648, True, False)),
            r"Option 'pass\.sg_set_scope' scope_id 2147483648 is out of range\. Expected -1~2147483647\.",
        ),
        (
            lambda: pypto.set_pass_options(cube_nbuffer_setting=[1, 2]),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'pass.cube_nbuffer_setting' has invalid type. "
            "Expected dict\\[int64, int64\\], but got list\\[int64\\]",
        ),
        (
            lambda: pypto.set_host_options(compile_monitor_enable="true"),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'host.compile_monitor_enable' has invalid type. Expected bool, but got string",
        ),
        (
            lambda: pypto.set_host_options(compile_timeout="100"),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'host.compile_timeout' has invalid type. Expected int64, but got string",
        ),
        (
            lambda: pypto.set_codegen_options(support_dynamic_aligned="true"),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'codegen.support_dynamic_aligned' has invalid type. Expected bool, but got string",
        ),
        (
            lambda: pypto.set_verify_options(pass_verify_save_tensor_dir=123),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'verify.pass_verify_save_tensor_dir' has invalid type. Expected string, but got int64",
        ),
        (
            lambda: pypto.set_debug_options(runtime_debug_mode="1"),
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'debug.runtime_debug_mode' has invalid type. Expected int64, but got string",
        )
    ],
)
def test_wrapper_option_type_mismatch_error(setter, pattern):
    assert_option_type_error(setter, pattern)


@pytest.mark.parametrize(
    ("options_kwargs", "pattern"),
    [
        (
            {"runtime_options": {"stitch_function_max_num": "aa"}},
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'runtime.stitch_function_max_num' has invalid type. Expected int64, but got string",
        ),
        (
            {"runtime_options": {"ready_on_host_tensors": "tensor0"}},
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'runtime.ready_on_host_tensors' has invalid type. Expected list\\[string\\], but got string",
        ),
        (
            {"runtime_options": {"device_sched_parallelism": "aa"}},
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'runtime.device_sched_parallelism' has invalid type. Expected int64, but got string",
        ),
        (
            {"verify_options": {"pass_verify_pass_filter": False}},
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'verify.pass_verify_pass_filter' has invalid type. Expected list\\[string\\], but got bool",
        ),
        (
            {"verify_options": {"pass_verify_error_tol": "0.1,0.1"}},
            "CHECK FAILED: ErrCode: F21003! Enum: FeError::INVALID_TYPE\n"
            "Option 'verify.pass_verify_error_tol' has invalid type. Expected list\\[double\\], but got string",
        ),
    ],
)
def test_set_options_type_mismatch_error(options_kwargs, pattern):
    assert_option_type_error(lambda: pypto.set_options(**options_kwargs), pattern)


def test_set_options_unknown_key_error():
    with pytest.raises(Exception, match="key: runtime.not_exist does not exist."):
        pypto.set_options(runtime_options={"not_exist": 1})


def test_pass_option_range_error():
    with pytest.raises(Exception, match=r"(?s)pass\.cube_l1_reuse_setting.*doesn't within the value range"):
        pypto.set_pass_options(cube_l1_reuse_setting={-11: 2})

    with pytest.raises(Exception, match=r"(?s)host\.compile_timeout.*doesn't within the value range"):
        pypto.set_host_options(compile_timeout=-10)
