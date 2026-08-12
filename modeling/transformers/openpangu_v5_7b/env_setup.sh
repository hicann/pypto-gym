#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# openPangu-Embedded-7B PyPTO 环境配置脚本
# 用法: source env_setup.sh

set -euo pipefail

# ===== 1. CANN 环境 =====
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0}"
if [ -f "${CANN_HOME}/bin/setenv.bash" ]; then
    source "${CANN_HOME}/bin/setenv.bash"
    export ASCEND_HOME_PATH="${CANN_HOME}"
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "[ERROR] Cannot find CANN environment. Set CANN_HOME or install CANN toolkit."
    return 1
fi

# ===== 2. PyPTO JIT 编译器所需的 pto-isa 头文件路径 =====
#    指向 pto-isa 仓库根目录（含 include/pto 子目录）
export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-/data/h50058642/h00949854/pto-isa}"

# ===== 3. NPU 设备与分布式通信 =====
export TILE_FWK_DEVICE_ID="${TILE_FWK_DEVICE_ID:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ===== 4. cann-recipes-infer 路径（executor/module 包）=====
export CANN_RECIPES_PATH="${CANN_RECIPES_PATH:-/data/l00504208/l00504208/pangu_7B_net/cann-recipes-infer_pangu}"

echo "[env_setup] CANN:           ${ASCEND_HOME_PATH}"
echo "[env_setup] PTO_TILE_LIB:   ${PTO_TILE_LIB_CODE_PATH}"
echo "[env_setup] NPU device:     ${TILE_FWK_DEVICE_ID}"
echo "[env_setup] CANN_RECIPES:   ${CANN_RECIPES_PATH}"
