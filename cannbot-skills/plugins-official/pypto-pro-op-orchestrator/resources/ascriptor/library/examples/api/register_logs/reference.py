# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch


def make_inputs(case):
    x = torch.rand(16, 64, generator=torch.Generator().manual_seed(case["seed"])) * 1000 + 0.001
    x[0, :8] = torch.tensor([1.0, 2.0, 4.0, 10.0, 100.0, 0.5, 0.1, 1024.0])
    return {"x": x}


def reference(inputs):
    x = inputs["x"]
    return {"ln": x.log(), "log2": x.log2(), "log10": x.log10()}
