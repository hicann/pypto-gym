#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

echo "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh



log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 运行冒烟测试
# ==============================
log "start run test case, please wait ..."

source /usr/local/Ascend/cann/set_env.sh
pip3 install psutil
echo "==============start smoke testing================="
pto_url="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run"
bash pypto_gym_smoke.sh --src ${WORKSPACE} --pto_isa_url "${pto_url}" --src_target_branch master  --f ${WORKSPACE}/pr_filelist.txt --ci 2>&1 | tee -a ./run_test.log
smoke_rc=${PIPESTATUS[0]}

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${repo_name}/package/${pr_id}/${slog_name}"; then
  echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${repo_name}/package/${pr_id}/slog.tar.gz"
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if [ "${smoke_rc}" -ne 0 ]; then
    echo "$date_time : run test case failed, smoke exit code ${smoke_rc}"
    exit 1
fi
if grep -E '(^|[[:space:]])(FAILED|ERRORS?)([[:space:]]|$)|[0-9]+ (failed|errors?)\b' "./run_test.log"; then
    echo "$date_time : run test case failed"
    exit 1
fi
