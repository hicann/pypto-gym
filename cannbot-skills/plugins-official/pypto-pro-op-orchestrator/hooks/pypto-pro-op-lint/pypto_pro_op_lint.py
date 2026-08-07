#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Backward-compatible entry point for pypto-pro-op-lint."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pypto_pro_op_lint as _pkg  # noqa: E402
from pypto_pro_op_lint import observability as _observability  # noqa: E402

LOGS_DIR = _observability.LOGS_DIR
LOGS_EVENTS_FILE = _observability.LOGS_EVENTS_FILE


def __getattr__(name: str):
    if hasattr(_pkg, name):
        return getattr(_pkg, name)
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(_pkg.main())
