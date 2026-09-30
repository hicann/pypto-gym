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
    return {
        "x": torch.randn(4, 48, generator=torch.Generator().manual_seed(case["seed"])),
        "variant": case["parameters"]["variant"],
    }


def reference(inputs):
    x = inputs["x"]
    kind = inputs["variant"]
    if kind == "overlap":
        expected = x.clone()
        expected[:, 16:48] = x[:, 8:40]
    elif kind == "gather":
        expected = x[:, ::4].clone()
    elif kind == "rank3":
        flat = x.flatten()
        expected = torch.stack([flat[start : start + 8] for start in (0, 16, 96, 112)])
    else:
        raise ValueError("unknown view case")
    return {"o": expected}
