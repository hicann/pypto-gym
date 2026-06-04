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
from pypto.experimental import set_operation_options, get_operation_options


def test_print_options():
    pypto.set_print_options(edgeitems=1,
                            precision=2,
                            threshold=3,
                            linewidth=4)


def test_pass_option():
    # tuple
    pypto.set_pass_options(sg_set_scope=48)
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (48, False, False)
    # map
    pypto.set_pass_options(cube_nbuffer_setting={3: 4})
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_setting"] == {3: 4}


def test_host_option():
    pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.EXECUTE_GRAPH.value
    pypto.set_host_options(compile_monitor_enable=False)
    host_option = pypto.get_host_options()
    assert host_option["compile_monitor_enable"] == False
    pypto.set_host_options(compile_monitor_print_interval=123)
    host_option = pypto.get_host_options()
    assert host_option["compile_monitor_print_interval"] == 123
    pypto.set_host_options(compile_timeout_stage=50)
    host_option = pypto.get_host_options()
    assert host_option["compile_timeout_stage"] == 50
    pypto.set_host_options(compile_timeout=1000)
    host_option = pypto.get_host_options()
    assert host_option["compile_timeout"] == 1000


def test_reset_option():
    pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.EXECUTE_GRAPH.value
    pypto.reset_options()
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.ALL_COMPLETE.value



def test_operation_option():
    set_operation_options(combine_axis=True)
    option = get_operation_options()
    assert option["combine_axis"] == True


def test_global_option():
    res = pypto.get_global_config("platform.enable_cost_model")
    assert res == False
    pypto.set_global_config("platform.enable_cost_model", True)
    res = pypto.get_global_config("platform.enable_cost_model")
    assert res == True

    pypto.set_global_config("codegen.parallel_compile", 10)
    res = pypto.get_global_config("codegen.parallel_compile")
    assert res == 10


def test_option_map():
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_setting"] == {-1: 1}


def test_sg_set_scope_new_format():
    pypto.set_pass_options(sg_set_scope=(1, True, True))
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (1, True, True)

    pypto.set_pass_options(sg_set_scope=48)
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (48, False, False)

    pypto.reset_options()
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (-1, False, False)
    try:
        pypto.set_pass_options(sg_set_scope=(1, True))  # 元素不足
        assert False, "Should raise FeError"
    except pypto.error.FeError as e:
        assert "Expected 3" in str(e)

    try:
        pypto.set_pass_options(sg_set_scope=(1, "True", True))  # 类型错误
        assert False, "Should raise FeError"
    except pypto.error.FeError as e:
        assert "Expected bool" in str(e)

if __name__ == "__main__":
    test_option_map()
    test_sg_set_scope_new_format()
