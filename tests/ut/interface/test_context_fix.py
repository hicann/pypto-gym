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
from pypto.frontend.parser.context import Context


def test_loop_variable_update():
    ctx = Context()

    # Simulate function body frame
    with ctx.with_frame():
        # Define num_batch in the outer frame
        ctx.add("num_batch", 1)

        # Simulate loop body frame
        with ctx.with_frame():
            # Update num_batch in the loop body
            # This should update the value in the outer frame, not create a new variable
            ctx.add("num_batch", 2)

        # After the loop ends, the updated value should be retained
        assert ctx.get()['num_batch'] == 2, f"Expected 2, actual {ctx.get()['num_batch']}"
