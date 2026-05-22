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

"""KernelBench case NPU 预检脚本.

在 case 文件生成后立即在 NPU 上执行 ``Model.forward()``，验证：
1. Model 可正常实例化并迁移到 NPU
2. forward() 在 NPU 上可正常执行（无 aclnn 不支持的 dtype/算子）
3. 返回值类型正确

用法::

    python .agents/skills/pypto-testcase-to-benchmark/scripts/validate_case_npu.py \\
        /path/to/KernelBench/pto_case/5_GlmAttentionPreQuant.py \\
        --device 0

退出码:
    0  - NPU 预检通过
    1  - NPU 预检失败（case 有兼容性问题）
    2  - NPU 不可用（无法完成验证）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_NPU_PROBE_SCRIPT = '''
import sys, os, traceback

os.environ.setdefault("TILE_FWK_DEVICE_ID", "{device_id}")

# 第一步：尝试导入 torch_npu，判断 NPU 环境
try:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device({device_id})
    _NPU_AVAILABLE = True
except Exception as e:
    print(f"__NPU_UNAVAILABLE__: {{e}}")
    sys.exit(0)

# 第二步：加载 case 文件
sys.path.insert(0, "{case_dir}")
mod = __import__("{case_stem}")

# 第三步：实例化 Model 并迁移到 NPU
device = f"npu:{device_id}"
try:
    init_args = mod.get_init_inputs()
    model = mod.Model(*init_args)
    model = model.to(device)
except Exception as e:
    print(f"__NPU_FAIL__: Model 初始化/迁移到 NPU 失败: {{type(e).__name__}}: {{e}}")
    sys.exit(1)

# 第四步：将输入迁移到 NPU 并执行 forward
try:
    inputs = mod.get_inputs()
    inputs_npu = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            inputs_npu.append(t.to(device))
        else:
            # 非 tensor 输入（KernelBench 格式禁止，但做防御处理）
            inputs_npu.append(t)

    # 同步，确保所有数据已就位
    torch.npu.synchronize()

    out = model(*inputs_npu)

    torch.npu.synchronize()

    # 第五步：验证返回值
    if isinstance(out, tuple):
        for i, o in enumerate(out):
            if not isinstance(o, torch.Tensor):
                print(f"__NPU_FAIL__: forward() 返回值第 {{i}} 项不是 Tensor, 类型为 {{type(o)}}")
                sys.exit(1)
        print(f"__NPU_PASS__: forward() 返回 {{len(out)}} 个 Tensor")
    elif isinstance(out, torch.Tensor):
        print(f"__NPU_PASS__: forward() 返回 Tensor, shape={{list(out.shape)}}")
    else:
        print(f"__NPU_FAIL__: forward() 返回值不是 Tensor 也不是 tuple, 类型为 {{type(out)}}")
        sys.exit(1)

except Exception as e:
    tb = traceback.format_exc()
    print(f"__NPU_FAIL__: forward() 执行失败: {{type(e).__name__}}: {{e}}")
    # 打印关键的错误行
    for line in tb.split("\\n"):
        if "forward" in line.lower() or "matmul" in line.lower() or "int32" in line.lower():
            print(f"  TRACE: {{line.strip()}}")
    sys.exit(1)
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KernelBench case NPU 预检 — 在 NPU 上执行 Model.forward() 验证兼容性"
    )
    parser.add_argument("case_path", type=str, help="Case 文件路径")
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU 设备 ID (默认: 0, 可通过 TILE_FWK_DEVICE_ID 环境变量覆盖)"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="子进程超时秒数 (默认: 60)"
    )
    args = parser.parse_args()

    case_path = Path(args.case_path).resolve()
    if not case_path.exists():
        print(f"❌ 文件不存在: {case_path}")
        return 2

    if not case_path.suffix == ".py":
        print(f"❌ 不是 .py 文件: {case_path}")
        return 2

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", args.device))

    case_dir = str(case_path.parent)
    case_stem = case_path.stem

    script = _NPU_PROBE_SCRIPT.format(
        device_id=device_id,
        case_dir=case_dir,
        case_stem=case_stem,
    )

    print(f"🔍 NPU 预检: {case_path.name}")
    print(f"   设备: npu:{device_id}")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
            env={**os.environ, "TILE_FWK_DEVICE_ID": str(device_id)},
        )
    except subprocess.TimeoutExpired:
        print(f"⏱️ 超时 ({args.timeout}s)，可能 NPU 资源不足或 case 计算量过大")
        return 1

    stdout = proc.stdout
    stderr = proc.stderr

    if "__NPU_PASS__" in stdout:
        for line in stdout.splitlines():
            if "__NPU_PASS__" in line:
                print(f"   ✅ {line.replace('__NPU_PASS__: ', '')}")
        print(f"\n   NPU 预检通过 ✓")
        return 0

    if "__NPU_UNAVAILABLE__" in stdout:
        for line in stdout.splitlines():
            if "__NPU_UNAVAILABLE__" in line:
                reason = line.replace("__NPU_UNAVAILABLE__: ", "")
                print(f"   ⚠️  NPU 不可用: {reason}")
        print(f"\n   ❌ NPU 预检失败 — 无法获取 NPU 设备")
        return 2

    if "__NPU_FAIL__" in stdout:
        for line in stdout.splitlines():
            if "__NPU_FAIL__" in line or "TRACE" in line:
                print(f"   {line.replace('__NPU_FAIL__: ', '')}")
        if stderr.strip():
            err_lines = [l for l in stderr.strip().splitlines() if l.strip()]
            for line in err_lines[-5:]:
                print(f"   {line.strip()}")
        print(f"\n   ❌ NPU 预检失败 — case 不兼容 NPU")
        return 1

    print(f"   ❌ 子进程异常退出 (code={proc.returncode})")
    if stdout.strip():
        print(f"   stdout: {stdout.strip()[-500:]}")
    if stderr.strip():
        print(f"   stderr: {stderr.strip()[-500:]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
