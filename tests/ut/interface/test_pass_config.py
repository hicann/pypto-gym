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
import pytest


def test_pass_config():
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True) is False
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    # reset
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, False)

    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction",
                                 pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction",
                                 pypto.PassConfigKey.KEY_DUMP_GRAPH, False) is True

    configs = pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
    assert configs.dumpGraph is True
    # reset
    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)

    with pytest.raises(pypto.error.PassError, match=r"Expected boolean type, but received int"):
        pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, -2)


def test_pass_option():
    test_params = {
        "sg_set_scope": (5, False, False),
        "vec_nbuffer_setting": {1: 2},
        "cube_l1_reuse_setting": {-1: 6, 2: 3},
        "cube_nbuffer_setting": {-1: 2}
    }
    pypto.set_pass_options(**test_params)
    option = pypto.get_pass_options()
    for key, expect_valuie in test_params.items():
        assert option[key] == expect_valuie


class TestHashOrderConfig:
    """hashOrder 相关配置测试"""

    @staticmethod
    def test_valid_integer_keys():
        """测试合法的整数 key，存储在 *_setting 中"""
        pypto.reset_options()
        pypto.set_pass_options(vec_nbuffer_setting={-1: 4, 0: 2, 1: 3})
        option = pypto.get_pass_options()
        assert option['vec_nbuffer_setting'] == {-1: 4, 0: 2, 1: 3}

    @staticmethod
    def test_valid_func_granularity_keys():
        """测试 func{magic}_{order} 格式的 key，存储在 *_by_func 中"""
        pypto.reset_options()
        pypto.set_pass_options(cube_l1_reuse_setting={"func123_0": 4, "func456_1": 2})
        option = pypto.get_pass_options()
        assert option['cube_l1_reuse_setting_by_func'] == {"func123_0": 4, "func456_1": 2}

    @staticmethod
    def test_valid_default_key():
        """测试 DEFAULT key，存储在 *_by_func 中"""
        pypto.reset_options()
        pypto.set_pass_options(cube_nbuffer_setting={"DEFAULT": 4})
        option = pypto.get_pass_options()
        assert option['cube_nbuffer_setting_by_func'] == {"DEFAULT": 4}

    @staticmethod
    def test_valid_mixed_default_and_func_keys():
        """测试混合使用 DEFAULT 和 func{magic}_{order} 格式 key"""
        pypto.reset_options()
        pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "func123_0": 4})
        option = pypto.get_pass_options()
        assert option['vec_nbuffer_setting_by_func'] == {"DEFAULT": 2, "func123_0": 4}

    @staticmethod
    def test_valid_semantic_label_keys():
        """测试语义标签（非 func{magic}_{order} 格式的字符串 key），存储在 *_by_label 中"""
        pypto.reset_options()
        pypto.set_pass_options(vec_nbuffer_setting={"conv": 1, "attention": 2})
        option = pypto.get_pass_options()
        assert option['vec_nbuffer_setting_by_label'] == {"conv": 1, "attention": 2}

    @staticmethod
    def test_valid_mixed_func_and_label_keys():
        """测试混合使用 func{magic}_{order} 格式和语义标签"""
        pypto.reset_options()
        pypto.set_pass_options(vec_nbuffer_setting={"func123_0": 1, "conv": 2})
        option = pypto.get_pass_options()
        assert option['vec_nbuffer_setting_by_func'] == {"func123_0": 1}
        assert option['vec_nbuffer_setting_by_label'] == {"conv": 2}

    @staticmethod
    def test_valid_mixed_int_and_label_keys():
        """测试混合使用整数 key 和语义标签（合法：两者是独立概念）"""
        pypto.reset_options()
        pypto.set_pass_options(vec_nbuffer_setting={-1: 4, "conv": 2})
        option = pypto.get_pass_options()
        assert option['vec_nbuffer_setting'] == {-1: 4}
        assert option['vec_nbuffer_setting_by_label'] == {"conv": 2}

    @staticmethod
    def test_invalid_mixed_int_and_func_keys():
        """测试混合使用整数 key 和 func/DEFAULT key（不合法）"""
        pypto.reset_options()
        with pytest.raises(pypto.error.FeError, match=r"cannot mix"):
            pypto.set_pass_options(vec_nbuffer_setting={-1: 4, "func123_0": 2})

    @staticmethod
    def test_valid_func_like_prefix_as_label():
        """测试以 func 开头但不匹配 func{magic}_{order} 格式的 key，视为语义标签"""
        pypto.reset_options()
        pypto.set_pass_options(cube_l1_reuse_setting={"func_0": 4})
        option = pypto.get_pass_options()
        assert option['cube_l1_reuse_setting_by_label'] == {"func_0": 4}

    @staticmethod
    def test_valid_func_without_order_as_label():
        """测试 func 前缀+数字但不含 _order 的 key，视为语义标签"""
        pypto.reset_options()
        pypto.set_pass_options(cube_nbuffer_setting={"func123": 4})
        option = pypto.get_pass_options()
        assert option['cube_nbuffer_setting_by_label'] == {"func123": 4}

    @staticmethod
    def test_invalid_semantic_label_hash_order_format():
        """测试与 hashOrder 格式冲突的 semantic label"""
        pypto.reset_options()
        with pytest.raises(pypto.error.FeError, match=r"conflicts with function-granularity hashOrder"):
            pypto.set_semantic_label("func123_0")
