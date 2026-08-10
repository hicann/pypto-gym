# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Golden for the pl softmax sample (pure torch, no pypto_pro).
# VALIDATED-CODE-SHA256: 07a8335dca255120047359aad0eede02a2d5ae8a7beb2284d3ab46bc7680dd9f
import logging

import torch

LOGGER = logging.getLogger(__name__)


def softmax_golden(x):
    """Row-wise softmax over the last axis: exp(x - rowmax) / rowsum."""
    return torch.softmax(x.float(), dim=-1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    x = torch.rand(64, 64) * 8.0 - 4.0
    LOGGER.info("golden shape: %s", tuple(softmax_golden(x).shape))
