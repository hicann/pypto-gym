# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch magnitudes; signed integer minima are outside the domain."""

import torch

HOST_DTYPES = {"i8": torch.int8, "i16": torch.int16, "i32": torch.int32,
               "i64": torch.int64, "f16": torch.float16, "f32": torch.float32}


def make_inputs(case):
    kind = case["parameters"]["kind"]
    dtype = HOST_DTYPES[kind]
    if dtype.is_floating_point:
        tiny = torch.finfo(dtype).tiny
        values = [0., -0., 1., -1., 3.25, -3.25, tiny, -tiny,
                  float("inf"), -float("inf"), 123.5, -123.5, 0.5, -0.5, 16., -16.]
    else:
        maximum = torch.iinfo(dtype).max
        values = [0, -1, 1, -2, 2, -7, 7, -31, 31, -63, 63,
                  -maximum, maximum, -maximum + 1, maximum - 1, -17]
    return {"source": torch.tensor(values, dtype=dtype).reshape(2, 8), "kind": kind}


def reference(inputs):
    return {"out": torch.abs(inputs["source"])}
