#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# 下载 PyPTO 维护的 KernelBench fork (PyTorch 原版 case 的支持子集) 并切到指定分支, 落地到
# pypto-gym/benchmark/.cache/KernelBench/.
#
# 用法:
#   bash benchmark/scripts/download_kernelbench.sh
#
# 自定义下载位置:
#   KERNELBENCH_DIR=/path/to/elsewhere bash .../download_kernelbench.sh
#
# 升级分支:
#   修改下方 KERNELBENCH_BRANCH 常量, 并同步 case_loader.py / run_kernelbench.py 里的引用.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(dirname "${SCRIPT_DIR}")"
DEFAULT_TARGET_DIR="${BRIDGE_ROOT}/.cache/KernelBench"

KERNELBENCH_REPO_URL="https://github.com/zwx2238/KernelBench.git"
KERNELBENCH_BRANCH="pypto-supported-21fbe"

TARGET_DIR="${KERNELBENCH_DIR:-${DEFAULT_TARGET_DIR}}"

if ! command -v git &> /dev/null; then
  echo "ERROR: git not found in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "[$(date +%H:%M:%S)] KernelBench 已存在, 更新远端信息: ${TARGET_DIR}"
  current_repo_url="$(git -C "${TARGET_DIR}" remote get-url origin 2>/dev/null || true)"
  if [ "${current_repo_url}" != "${KERNELBENCH_REPO_URL}" ]; then
    echo "[$(date +%H:%M:%S)] 切换 origin 到 ${KERNELBENCH_REPO_URL}"
    git -C "${TARGET_DIR}" remote set-url origin "${KERNELBENCH_REPO_URL}"
  fi
  git -C "${TARGET_DIR}" fetch --tags origin "${KERNELBENCH_BRANCH}:refs/remotes/origin/${KERNELBENCH_BRANCH}"
else
  if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: 目录 ${TARGET_DIR} 已存在且非空, 但不是 git 仓; 请手动清理后重试." >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] 克隆 KernelBench 到 ${TARGET_DIR}..."
  git clone --branch "${KERNELBENCH_BRANCH}" "${KERNELBENCH_REPO_URL}" "${TARGET_DIR}"
fi

if git -C "${TARGET_DIR}" rev-parse --verify "origin/${KERNELBENCH_BRANCH}^{commit}" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] checkout ${KERNELBENCH_BRANCH}..."
  git -C "${TARGET_DIR}" checkout -B "${KERNELBENCH_BRANCH}" "origin/${KERNELBENCH_BRANCH}"
else
  echo "ERROR: 未能找到目标分支 ${KERNELBENCH_BRANCH}" >&2
  exit 1
fi

echo ""
echo "[OK] KernelBench ready"
echo "  path:   ${TARGET_DIR}"
echo "  branch: ${KERNELBENCH_BRANCH}"

LEVELS_DIR="${TARGET_DIR}/KernelBench"
if [ -d "${LEVELS_DIR}" ]; then
  echo "  levels:"
  for d in "${LEVELS_DIR}"/level*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name '*.py' | wc -l)
    echo "    - $(basename "$d") (${n} cases)"
  done
fi
