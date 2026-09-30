# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math

import torch


def make_inputs(case):
    values = [-40000, -32769, -32768, -1, 0, 32767, 32768, 40000]
    floats = (
        [-math.inf, -(2.0**40), -32768.0, -1.0, 0.0, 2.0**40, math.inf, math.nan]
        if case["parameters"]["edges"]
        else [-40000.5, -32768.5, -32767.5, -0.5, 0.5, 32766.5, 32767.5, 40000.5]
    )
    return {
        "x": torch.tensor([values * 8], dtype=torch.int32),
        "xf": torch.tensor([floats * 8]),
        "initial_mode": case["parameters"]["initial_mode"],
    }


def convert(value, saturate):
    if isinstance(value, float):
        if math.isnan(value):
            return 0
        if math.isinf(value):
            return 32767 if value > 0 else -32768
        value = round(value)
    return max(-32768, min(32767, value)) if saturate else (value + 32768) % 65536 - 32768


def reference(inputs):
    output = torch.zeros(5, 512, dtype=torch.int16)
    for mode in range(4):
        for register in range(4):
            saturate = mode == 2 or (mode < 2 and register % 2 == 0)
            values = inputs["x" if register < 2 else "xf"][0].tolist()
            output[mode, register * 128 : (register + 1) * 128 : 2] = torch.tensor([convert(value, saturate) for value in values], dtype=torch.int16)
    return {"o": output}
