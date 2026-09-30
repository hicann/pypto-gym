# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    return {"x": torch.randn(32, 64, generator=g), "xh": torch.randn(32, 64, generator=g).bfloat16()}


def reference(inputs):
    x, xh = inputs["x"], inputs["xh"]
    return {
        "constant": F.pad(x[2:14, 3:27], (3, 5, 2, 2), value=-1.5),
        "nearest": F.pad(x[None, None, 2:14, 3:27], (3, 5, 2, 2), mode="replicate")[0, 0],
        "rows": x[0:16:2, :32].clone(),
        "transpose": x[:16, :32].T.contiguous(),
        "bf16_pad": F.pad(xh[4:20, :48].float(), (8, 8, 8, 8), value=2.0).bfloat16(),
    }
