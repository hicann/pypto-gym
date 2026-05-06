# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""suspicious fixture: 表面合法但有可疑注释, cheat_detector 应判 suspicious."""
import torch
import pypto


@pypto.jit
def relu_kernel(x):
    return pypto.dsl.relu(x)


# workaround: fixture 故意保留可疑措辞以触发 suspicious verdict (非生产代码).
def relu_wrapper(x):
    return relu_kernel(x)
