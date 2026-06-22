#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# PyPTO SMoke 控制
# 注意: 该脚本可能由 Host 通过远程调用方式, 使其运行在 Docker 中


set -e

echo "======== 脚本原始参数 \$* = [$*]"
echo "======== 脚本参数个数 \$# = [$#]"

# ======================================================================================================================
# 1. 基础函数实现
# ======================================================================================================================

BRed='\e[1;31m'         # Red
BGreen='\e[1;32m'       # Green
Purple='\e[0;35m'       # Purple
BPurple='\e[1;35m'      # Bold Purple
Color_Off='\e[0m'       # Text Reset

function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "\n${BGreen}[INFO] ${date_time} ${assert_msg}${Color_Off}"
}

function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}${cmd}${Color_Off}"
   ${cmd}
}

function LOG_INFO() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}" >&2
}

function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}


# ======================================================================================================================
# 2. 全局配置
#    全局初始化操作调用
#    自定义全局变量及函数实现
# ======================================================================================================================
LOG_DO "source $HOME/.bashrc"  # 如果是新环境, 注意修改 .bashrc 内容，去除 [ -z "$PS1" ] && return

# 获取当前脚本目录
CUR_DIR=$(cd "$(dirname "$0")" && pwd)

# DATA_DIR 是当前目录的 **上一级目录**
DATA_DIR=$(dirname "${CUR_DIR}")

# 打印确认（可选）
LOG_INFO "CUR_DIR=$CUR_DIR"
LOG_INFO "DATA_DIR=$DATA_DIR"
if [ -d "/usr/local/Ascend/ascend-toolkit/latest/" ]; then
    echo "ascend-toolkit 目录存在"
fi
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/cann-9.1.0/bin/setenv.bash

# 检查升级锁文件，升级锁检查已在外部设置，此处注掉
# ENV_UPGRADE_LOCK_FILE="$DATA_DIR/env_upgrade_flag_v1"
# if [ -f "$ENV_UPGRADE_LOCK_FILE" ]; then
#     LOG_ERROR "Current Environment upgrading, please wait a moment."
#     exit 1
# fi

run_build_ci() {
    echo "========run build ci==========="
    local _python3="$1"       # 第一个参数是 python3 路径
    local desc="$2"           # 第二个参数是任务描述
    shift 2                   # 剩余参数传给 build_ci.py

    # 检查 python3 是否存在
    if ! command -v "$_python3" &>/dev/null; then
        LOG_ERROR "Python interpreter '$_python3' not found!"
        exit 1
    fi

    # 开始执行
    start_time=$(date +%s)
    # LOG_DO "ccache" "-z"  # 监测 CCACHE 命中率
    LOG_DO "npu-smi" "info"  # 检测进程残留
    LOG_HEAD "[BGN] $desc "
    LOG_DO "$_python3" "build_ci.py" "$@"
    local ret=$?

    # 结束执行
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    LOG_HEAD "[END] $desc, Ret $ret, duration $elapsed secs."
    # LOG_DO "ccache" "-s"  # 监测 CCACHE 命中率

    # 结果处理
    if [ $ret -ne 0 ]; then
        LOG_ERROR "$desc failed"
        LOG_DO "$_python3" "kill_npu_processes.py"  # 终止 NPU 进程
        LOG_DO "npu-smi" "info"  # 检测进程残留
        exit $ret
    else
        LOG_INFO "$desc succeeded"
        LOG_DO "npu-smi" "info"  # 检测进程残留
    fi
}

parse_config_key() {
    # 定义局部变量，避免污染全局
    # local config_file="$DATA_DIR/pypto_cann.cfg"
    CUR_DIR=$(cd "$(dirname "$0")" && pwd)
    local config_file="$CUR_DIR/pypto_cann.cfg"
    local config_query_key="$1"
    local config_query_val=""

    # 1. 校验配置文件是否存在
    if [ ! -f "$config_file" ]; then
        LOG_ERROR "Config file $config_file not exist."
        exit 1
    fi
    # 2. 校验参数：目标键名不能为空
    if [ -z "$config_query_key" ]; then
        LOG_ERROR "Config query key empty."
        exit 1
    fi
    # 3. 核心逻辑：精准匹配键名，提取值并去除所有空白字符
    # -F '='：以等号为分隔符
    # -v key="$config_query_key"：将shell变量传入awk
    # $1 == key：严格匹配键名（避免部分匹配）
    # gsub(/[[:space:]]/, "")：去除值中的空格、制表符、换行等所有空白字符
    # exit：找到后立即退出，提升效率
    config_query_val=$(awk -F '=' -v key="$config_query_key" '
        $1 == key {
            gsub(/[[:space:]]/, "");  # 清洗值的空白字符
            print $2;
            exit;
        }
    ' "$config_file")
    # 4. 直接输出值（无匹配/值为空时，输出空字符串）
    echo "$config_query_val"
    return 0
}

PYTHON3_EXE=$(which python3)
LOG_INFO "PYTHON3_EXE=$PYTHON3_EXE"

PYPTO_GOLDEN_PATH=$(parse_config_key "PYPTO_GOLDEN_PATH")
if [ -z "$PYPTO_GOLDEN_PATH" ]; then
    LOG_ERROR "Can't get PYPTO_GOLDEN_PATH."
    exit 1
fi
LOG_INFO "PYPTO_GOLDEN_PATH=$PYPTO_GOLDEN_PATH"

PYPTO_3RD_LIB_PATH=$(parse_config_key "PYPTO_3RD_LIB_PATH")
if [ -z "$PYPTO_3RD_LIB_PATH" ]; then
    LOG_ERROR "Can't get PYPTO_3RD_LIB_PATH."
    exit 1
fi
LOG_INFO "PYPTO_3RD_LIB_PATH=$PYPTO_3RD_LIB_PATH"

# ======================================================================================================================
# 3: 解析脚本参数
# ======================================================================================================================
SRC_DIR=""
SRC_TARGET_BRANCH="master"  # 不传入本参数时默认是 master
PTO_ISA_URL=""
CI_MODE=false
CHANGED_FILE_FROM_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)
            SRC_DIR="$2"
            shift 2
            ;;
        --src_target_branch)
            SRC_TARGET_BRANCH="$2"
            shift 2
            ;;
        --pto_isa_url)
            PTO_ISA_URL="$2"
            shift 2
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --f)
            CHANGED_FILE_FROM_ARG="$2"  # <-- 接收 --f 参数
            shift 2
            ;;
        *)
            LOG_ERROR "Unknown argument $1"
            exit 1
            ;;
    esac
done
# 检查必须参数
if [ -z "$SRC_DIR" ]; then
    LOG_ERROR "--src argument is required"
    exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
    LOG_ERROR "$SRC_DIR is not a valid directory"
    exit 1
fi
# 检查输出可选参数
LOG_INFO "SRC_DIR=$SRC_DIR"
LOG_INFO "SRC_TARGET_BRANCH=$SRC_TARGET_BRANCH"
LOG_INFO "PTO_ISA_URL=$PTO_ISA_URL"

# ======================================================================================================================
# 4: 设置 CANN 环境变量
# ======================================================================================================================
TARGET_CONFIG_KEY="CANN_$SRC_TARGET_BRANCH"
CANN_PATH=$(parse_config_key "$TARGET_CONFIG_KEY")
if [ -z "$CANN_PATH" ]; then
    # 错误提示包含关键信息：缺失的键名、配置文件路径、分支名，方便快速定位
    LOG_ERROR "Failed to get CANN path! "
    LOG_ERROR "  - Target key: $TARGET_CONFIG_KEY (branch: $SRC_TARGET_BRANCH)"
    LOG_ERROR "  Please check if the key exists and has a valid value in the config file."
    exit 1
fi
LOG_INFO "CANN_PATH=$CANN_PATH"

# 安装 PTO-ISA 包
if [ -n "${PTO_ISA_URL}" ]; then

    # 目标目录
    PTO_ISA_PKG_DIR="$DATA_DIR/pto_isa_pkg/$SRC_TARGET_BRANCH"

    # 创建目录（不存在则自动创建）
    mkdir -p "${PTO_ISA_PKG_DIR}"

    # 进入目录，并记住进入前的路径
    pushd "${PTO_ISA_PKG_DIR}" > /dev/null || exit 1

    # 删除所有 .run 文件
    rm -f ./*.run

    # wget .run 包并安装
    PTO_ISA_PKG_NAME="cann-pto-isa_linux.run"
    wget -O "$PTO_ISA_PKG_NAME" "${PTO_ISA_URL}"
    chmod +x ./$PTO_ISA_PKG_NAME
    LOG_DO "bash" "$PTO_ISA_PKG_NAME" "--full" "--quiet" "--install-path=${CANN_PATH}../"

    echo "安装pto-isa包成功：${CANN_PATH}"

    # 自动回到进入前的工作目录
    popd > /dev/null

fi

# 验证setenv.bash文件存在性
SETENV_SH="$CANN_PATH/bin/setenv.bash"
if [ ! -f "$SETENV_SH" ]; then
    LOG_ERROR "$SETENV_SH not found (CANN path: $CANN_PATH)"
    exit 1
fi
LOG_DO "source $SETENV_SH"

# 下载pypto代码
work_dir=$(pwd)
# 定位上级目录
work_parent_dir="${work_dir}/.."
target_repo="${work_parent_dir}/pypto"

# 判断上级目录是否存在pypto文件夹，存在则彻底删除
if [ -d "${target_repo}" ]; then
    echo "检测到上级目录已存在pypto仓库，开始删除..."
    rm -rf "${target_repo}"
fi

# if [ ! -d "${target_repo}" ]; then
#     LOG_INFO "上级目录 ${target_repo} 不存在，开始创建"
#     mkdir -p "${target_repo}"
#     chmod 755 "${target_repo}"
#     LOG_INFO "目录创建完成，权限已设置755"
# fi

# 进入上级目录执行克隆操作
cd "${work_parent_dir}" || exit 1
git clone https://gitcode.com/cann/pypto.git

echo "target_repo 变量值为: ${target_repo}"

echo "========== 当前目录文件列表 =========="
ls "${target_repo}"
echo "======================================"

# 进入上级目录执行克隆操作
cd "${work_dir}" || exit 1


# ======================================================================================================================
# 5: 调用任务, 并统计耗时
# ======================================================================================================================
cd "$SRC_DIR" || { LOG_ERROR "Failed to cd to $SRC_DIR"; exit 1; }

# 调用参数准备
CHANGED_FILES_PARAM=""
if [[ "$CI_MODE" == true ]]; then
    if [ -n "$CHANGED_FILE_FROM_ARG" ]; then
        changed_file_path="$CHANGED_FILE_FROM_ARG"
        LOG_INFO "使用外部传入文件: $changed_file_path"
    fi

    LOG_INFO "$changed_file_path"
    # if [ -f "$changed_file_path" ]; then
    #     rm "$changed_file_path"
    # fi

    CHANGED_FILES_PARAM="--changed_files=$changed_file_path"
    LOG_INFO "Changed files content as follows:"
    LOG_DO "cat $changed_file_path"
fi

# 开始执行任务
total_start=$(date +%s)

device_params=(
    "-d=0"  "-d=1"
    "-d=2"  "-d=3"
    "-d=4"  "-d=5"
    "-d=6"  "-d=7"
    "-d=8"  "-d=9"
    "-d=10" "-d=11"
    "-d=12" "-d=13"
    "-d=14" "-d=15"
)
common_params=(
    "--clean"
    "--verbose"
    "--cann_3rd_lib_path=$PYPTO_3RD_LIB_PATH"
    "--golden_path=$PYPTO_GOLDEN_PATH" "$CHANGED_FILES_PARAM"
)

# 参数默认值
PYTHON_TOTAL_TIMEOUT=$(parse_config_key "PYTHON_TOTAL_TIMEOUT")
if [ -z "$PYTHON_TOTAL_TIMEOUT" ]; then
    PYTHON_TOTAL_TIMEOUT=900
fi
PYTHON_TOTAL_TIMEOUT="$PYTHON_TOTAL_TIMEOUT"
LOG_INFO "PYTHON_TOTAL_TIMEOUT=$PYTHON_TOTAL_TIMEOUT"

LOG_HEAD "Python Environment:"
LOG_DO "$PYTHON3_EXE --version"
LOG_DO "$PYTHON3_EXE -m pip list"
run_build_ci "$PYTHON3_EXE" "Python(Examples)" "${common_params[@]}" "--timeout=$PYTHON_TOTAL_TIMEOUT" --models "${device_params[@]}"

# run_build_ci "$PYTHON3_EXE" "Python(STest)" "${common_params[@]}" "--timeout=$PYTHON_TOTAL_TIMEOUT" --stest "${device_params[@]}"
# # 2026/1/31 增加集合通信测试用例
# # 2026/3/20: 重新梳理及明确规格 --timeout=300 --case_execute_timeout=35
# run_build_ci "$PYTHON3_EXE" "C++(STest Distributed)" --frontend=cpp "${common_params[@]}" "--timeout=$CPP_HCCL_TOTAL_TIMEOUT" --stest_distributed "${device_params[@]}" "--case_execute_timeout=$CPP_CASE_EXEC_TIMEOUT"

# 总耗时统计
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
LOG_HEAD "All builds completed successfully, Total execution time: $total_elapsed seconds"
echo -e "execute sample success"
