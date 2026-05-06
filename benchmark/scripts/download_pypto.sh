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
# 下载 PyPTO 源码仓 master 最新版本，作为 pypto-gym benchmark 的外部工作仓。
#
# 用法:
#   bash benchmark/scripts/download_pypto.sh
#
# 自定义下载位置:
#   PYPTO_DIR=/path/to/pypto bash benchmark/scripts/download_pypto.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(dirname "${SCRIPT_DIR}")"
DEFAULT_TARGET_DIR="${BRIDGE_ROOT}/.cache/pypto"

PYPTO_REPO_URL="https://gitcode.com/cann/pypto.git"
PYPTO_BRANCH="master"

TARGET_DIR="${PYPTO_DIR:-${DEFAULT_TARGET_DIR}}"

if ! command -v git &> /dev/null; then
  echo "ERROR: git not found in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "[$(date +%H:%M:%S)] PyPTO 已存在, 更新远端信息: ${TARGET_DIR}"
  current_repo_url="$(git -C "${TARGET_DIR}" remote get-url origin 2>/dev/null || true)"
  if [ "${current_repo_url}" != "${PYPTO_REPO_URL}" ]; then
    echo "[$(date +%H:%M:%S)] 切换 origin 到 ${PYPTO_REPO_URL}"
    git -C "${TARGET_DIR}" remote set-url origin "${PYPTO_REPO_URL}"
  fi
  git -C "${TARGET_DIR}" fetch --tags origin "${PYPTO_BRANCH}:refs/remotes/origin/${PYPTO_BRANCH}"
else
  if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: 目录 ${TARGET_DIR} 已存在且非空, 但不是 git 仓; 请手动清理后重试." >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] 克隆 PyPTO 到 ${TARGET_DIR}..."
  git clone --branch "${PYPTO_BRANCH}" "${PYPTO_REPO_URL}" "${TARGET_DIR}"
fi

if git -C "${TARGET_DIR}" rev-parse --verify "origin/${PYPTO_BRANCH}^{commit}" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] checkout ${PYPTO_BRANCH}..."
  git -C "${TARGET_DIR}" checkout -f -B "${PYPTO_BRANCH}" "origin/${PYPTO_BRANCH}"
else
  echo "ERROR: 未能找到目标分支 ${PYPTO_BRANCH}" >&2
  exit 1
fi

echo ""
echo "[OK] PyPTO ready"
echo "  path:   ${TARGET_DIR}"
echo "  branch: ${PYPTO_BRANCH}"
echo "  commit: $(git -C "${TARGET_DIR}" rev-parse --short HEAD)"
