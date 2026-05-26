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
"""Tests for generated profile scripts."""

from __future__ import annotations

from benchmark.verifier import script_builder


def test_profile_generation_prepares_swimlane_output_before_modelnew_import() -> None:
    script = script_builder.build_profile_generation_script(
        op_name="Matmul_Min_Subtract",
        framework_filename="Matmul_Min_Subtract.py",
        device_id=0,
        pypto_run_mode=0,
    )

    output_setup_idx = script.index('os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir')
    prepared_flag_idx = script.index("_swimlane_output_prepared = True")
    modelnew_import_idx = script.index("_impl_spec.loader.exec_module(_impl_module)")
    fallback_guard_idx = script.index('if not globals().get("_swimlane_output_prepared", False):')

    assert output_setup_idx < modelnew_import_idx
    assert prepared_flag_idx < modelnew_import_idx
    assert modelnew_import_idx < fallback_guard_idx
