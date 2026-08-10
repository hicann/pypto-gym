# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Golden for the matmul_float_mmad transfer (pure torch, NO pypto — lint D3).
# twin_of: easyasc:kernels/a5/matmul/matmul_float_mmad.py
#
# Oracle demonstrated on Tier 1 (Mac) via
#   python tools/agent/easyasc_oracle_dump.py \
#       ../easyasc/kernels/a5/matmul/matmul_float_mmad.py --out oracle/
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# The oracle and simulator artifacts matched exactly; the reference performance target is 1,483 cycles.
# VALIDATED-CODE-SHA256: ed7722de58bb8cb621265fd66ec9dac1b1fc950b6c1d58f2bd245c0be032b839
import logging

import torch

LOGGER = logging.getLogger(__name__)


def matmul_float_mmad_golden(x, y):
    """z = x @ y.T with FP32 accumulation.

    x: [M, K] float32,  y: [N, K] float32  ->  z: [M, N] float32.
    Matches the easyasc kernel's embedded reference `z_ref = x @ y.t()`.
    The `.float()` is load-bearing: the pypto twin accumulates in FP32 in L0C
    (constraints/precision.md), so the golden must accumulate in FP32 too.
    """
    return x.float() @ y.float().t()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    M, N, K = 32, 48, 16
    x = torch.randn(M, K, dtype=torch.float32)
    y = torch.randn(N, K, dtype=torch.float32)
    z = matmul_float_mmad_golden(x, y)
    LOGGER.info("golden z shape: %s", tuple(z.shape))
