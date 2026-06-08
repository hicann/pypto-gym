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
DeepFM PyPTO 性能分析数据目录

本目录仅包含 DeepFM 模型的 profiling 输出数据（PROF_* 目录、output/ 目录、test/ 目录），
不包含 PyPTO kernel 源代码。profiling 数据用于分析算子在不同 shape/配置下的运行性能。

子目录说明：
- PROF_*: 单次 profiling 会话的输出数据
- output/: profiling 结果汇总
- test/: 测试脚本与配置
"""

__all__ = []
