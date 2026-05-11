#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Timeouts shared across benchmark CLI, runners, and loaders."""

# ``npu-smi info`` may block for a long time on busy hosts; enforced per subprocess call.
NPU_SMI_INFO_TIMEOUT_SEC = 300

# Detached ``run``: parent only waits for stub ``state.json`` (written right after child ``_build_cfg``).
STATE_JSON_STUB_WAIT_SEC = 90

# 后台 detached ``run`` 是否自动 attach monitor TUI：`1|true|yes|on` 强制打开，`0|false|no|off` 强制关闭；
# 任一显式取值均优先于 CLI ``--no-auto-monitor`` 与默认行为。
BENCHMARK_AUTO_MONITOR_ENV = "BENCHMARK_AUTO_MONITOR"
