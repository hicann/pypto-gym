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
"""PyPTO 专属代码片段生成器.

把 ``script_builder`` 需要的 PyPTO 相关代码片段以函数形式集中暴露:
    - ``get_pypto_imports``            — verify/profile 脚本头部 import 块.
    - ``get_runtime_env_overrides``    — profile 脚本必需的运行期环境调整,
                                         含 monkey-patch ``pypto.frontend.jit`` 让
                                         swimlane 写到本进程可见的位置.
    - ``get_modelnew_loader``          — 加载 ``{op}_pypto_impl.py`` 中的 ModelNew,
                                         缺类或缺文件时自动从 ``{op}_impl.py`` 的
                                         ``{op}_wrapper`` 包出 ModelNew (wrapper-only fallback).
    - ``get_swimlane_benchmark_body``  — 用 PyPTO swimlane trace 聚合算 generation 时间.

只覆盖 ``backend="ascend"`` + ``framework="torch"`` 一种组合.
"""

from __future__ import annotations

import re
from typing import Optional


def get_pypto_imports() -> str:
    """torch + pypto + os 的 import 语句串."""
    return "import torch\nimport pypto\nimport os\n"


def get_runtime_env_overrides(
    pypto_run_mode: Optional[int] = None,
    pypto_runtime_debug_mode: Optional[int] = None,
) -> str:
    """生成 PyPTO runtime 环境变量覆盖代码.

    ``pypto_runtime_debug_mode=1`` 会同时:
        1. 设置 ``AIKG_PYPTO_RUNTIME_DEBUG_MODE=1``
        2. **monkey-patch ``pypto.frontend.jit``** — PyPTO 源码不读上面那个 env
           var, 真实开关是 jit 装饰器的 ``debug_options.runtime_debug_mode=1``,
           因此必须在 ``import {op}_impl`` (会触发 jit 编译) **之前** rebind jit.

    Verify 路径传 0 (不开 swimlane, 不影响精度路径性能).
    Profile generation 路径传 1.
    """
    lines = []
    if pypto_run_mode is not None:
        lines.append(f'os.environ["AIKG_PYPTO_RUN_MODE"] = "{pypto_run_mode}"')
        lines.append(
            'print(f"[INFO] Task override: AIKG_PYPTO_RUN_MODE={os.environ[\'AIKG_PYPTO_RUN_MODE\']}")'
        )
    if pypto_runtime_debug_mode is not None:
        lines.append(
            f'os.environ["AIKG_PYPTO_RUNTIME_DEBUG_MODE"] = "{pypto_runtime_debug_mode}"'
        )
        lines.append(
            'print(f"[INFO] Task override: AIKG_PYPTO_RUNTIME_DEBUG_MODE={os.environ[\'AIKG_PYPTO_RUNTIME_DEBUG_MODE\']}")'
        )
        if pypto_runtime_debug_mode == 1:
            lines.extend([
                "# Monkey-patch pypto.frontend.jit: PyPTO 不读 AIKG_PYPTO_RUNTIME_DEBUG_MODE",
                "# env var, 真实开关是 jit 装饰器的 debug_options.runtime_debug_mode=1.",
                "# 必须在 import {op}_impl 之前 rebind, jit 编译时机才能拿到.",
                "import pypto.frontend as _jit_pf",
                "_jit_orig = _jit_pf.jit",
                "def _jit_with_debug(*_a, **_kw):",
                "    _kw.setdefault('debug_options', {})",
                "    _kw['debug_options'].setdefault('runtime_debug_mode', 1)",
                "    return _jit_orig(*_a, **_kw)",
                "_jit_pf.jit = _jit_with_debug",
                "print('[INFO] Monkey-patched pypto.frontend.jit with runtime_debug_mode=1 for swimlane')",
            ])
    return "\n".join(lines) + ("\n" if lines else "")


def get_modelnew_loader(op_name: str) -> str:
    """生成动态加载 ``ModelNew`` 的代码片段.

    加载顺序:
        1. 优先 ``{op}_pypto_impl.py`` 里的 ``ModelNew``.
        2. 缺类或缺文件时回退到 ``{op}_impl.py`` 里的 ``{op}_wrapper``,
           动态包装成最简 ``ModelNew(nn.Module)``. 用于兼容只生成
           ``{op}_impl.py`` 的场景.
    """
    module_name = re.sub(r"\W", "_", op_name)
    if not module_name or module_name[0].isdigit():
        module_name = f"op_{module_name}"
    return (
        "import importlib.util\n"
        "import os\n"
        "_impl_dir = os.path.dirname(__file__)\n"
        f"_impl_module_name = '{module_name}_pypto_impl'\n"
        f"_impl_module_path = os.path.join(_impl_dir, '{op_name}_pypto_impl.py')\n"
        f"_wrapper_module_path = os.path.join(_impl_dir, '{op_name}_impl.py')\n"
        "ModelNew = None\n"
        "if os.path.exists(_impl_module_path):\n"
        "    _impl_spec = importlib.util.spec_from_file_location(_impl_module_name, _impl_module_path)\n"
        "    _impl_module = importlib.util.module_from_spec(_impl_spec)\n"
        "    _impl_spec.loader.exec_module(_impl_module)\n"
        "    ModelNew = getattr(_impl_module, 'ModelNew', None)\n"
        "if ModelNew is None and os.path.exists(_wrapper_module_path):\n"
        "    # Fallback: pypto produced only {op}_impl.py with {op}_wrapper.\n"
        "    import torch.nn as nn\n"
        f"    _wrapper_module_name = '{module_name}_impl'\n"
        "    _wrapper_spec = importlib.util.spec_from_file_location(_wrapper_module_name, _wrapper_module_path)\n"
        "    _wrapper_module = importlib.util.module_from_spec(_wrapper_spec)\n"
        "    _wrapper_spec.loader.exec_module(_wrapper_module)\n"
        f"    _wrapper_fn = getattr(_wrapper_module, '{op_name}_wrapper', None)\n"
        "    if _wrapper_fn is None:\n"
        f"        raise AttributeError('Neither {op_name}_pypto_impl.ModelNew nor '\n"
        f"                             '{op_name}_impl.{op_name}_wrapper was found.')\n"
        "    class ModelNew(nn.Module):\n"
        "        def __init__(self, *init_args, **init_kwargs):\n"
        "            super().__init__()\n"
        "        def forward(self, *inputs):\n"
        "            return _wrapper_fn(*inputs)\n"
        "if ModelNew is None:\n"
        f"    raise FileNotFoundError('No PyPTO impl found: tried '\n"
        f"                            '{op_name}_pypto_impl.py and {op_name}_impl.py')\n"
    )


def get_swimlane_benchmark_body() -> str:
    """返回 swimlane 计时代码 — 直接 inline 进 profile_<op>_generation.py.

    依赖外层环境:
        - ``impl_model``  (已 .to('npu:<device_id>'))
        - ``inputs``      (list, 已 .to('npu:<device_id>'))
        - ``case_idx``    (int, 用于隔离 prof_generation_output 子目录)

    输出 (写入 stdout, 由 KernelVerifier / skill 解析):
        - ``PROFILE_RESULT_GEN_US: <us>``  正常: 单 kernel swimlane span.
        - ``PROFILE_RESULT_GEN_US: inf``   异常: 0 个或多于 1 个 kernel trace.
        - ``CHEAT_MULTI_KERNEL: n=<count>; dirs=<csv>``  当检测到 >1 个
          swimlane 子目录时打出, 表示算子被拆成多个 jit kernel — 这违反
          PyPTO "一个算子 = 一个融合 kernel" 的约定, 必须由调用方判定为作弊.

    设计意图:
        PyPTO 算子要求是融合 kernel, 一个 forward 应只触发 1 个 jit 编译产物.
        过去的"多 trace span 累加"实质上掩盖了"把算子拆成 N 个 kernel"
        这种作弊形态: 精度可能依然通过, 性能甚至能达标, 但 kernel 不合规.
        因此本函数只承认单 kernel; >1 直接判 CHEAT, 让上层流程拒绝该实现.
    """
    return '''
import json
import glob

def _read_x_events(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]

def _span_of_events(events):
    if not events:
        return 0.0
    min_ts = min(float(e.get("ts", 0) or 0) for e in events)
    max_end = max(
        float(e.get("ts", 0) or 0) + float(e.get("dur", 0) or 0)
        for e in events
    )
    return max_end - min_ts

def _find_kernel_swimlanes(base_dir):
    """返回 base_dir 下每个 jit kernel 子目录里 mtime 最新的 merged_swimlane.json.

    PyPTO 在 runtime_debug_mode=1 下, 把每个 @pypto.jit kernel 的 trace
    写到独立的 output_<timestamp>_<pid>_<host>/ 子目录; autotune 多次
    试错时同一文件被覆盖, 取 mtime 最新的即收敛后最优 trace.

    返回 (trace_paths, kernel_dirs); 调用方负责判定数量是否合规.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Profiler output dir not found: {base_dir}")

    kernel_dirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "output_*"))
        if os.path.isdir(d)
    )
    results = []
    for kd in kernel_dirs:
        cands = glob.glob(os.path.join(kd, "merged_swimlane.json"))
        cands += glob.glob(os.path.join(kd, "**", "merged_swimlane.json"), recursive=True)
        cands = sorted(set(cands), key=os.path.getmtime, reverse=True)
        if cands:
            results.append(cands[0])
    if results:
        return results, kernel_dirs

    # 兜底: 老布局, base_dir 直接放 merged_swimlane.json (没有 output_* 层).
    cands = glob.glob(os.path.join(base_dir, "merged_swimlane.json"))
    cands += glob.glob(os.path.join(base_dir, "**", "merged_swimlane.json"), recursive=True)
    cands = sorted(set(cands), key=os.path.getmtime, reverse=True)
    if cands:
        return [cands[0]], [base_dir]

    try:
        entries = sorted(os.listdir(base_dir))
    except Exception:
        entries = []
    raise FileNotFoundError(
        f"No merged_swimlane.json found under {base_dir}. "
        f"Top-level entries: {entries[:30]}"
    )


def pypto_benchmark_fn():
    return impl_model(*inputs)

# persistent 场景下必须每次重置输出目录与日志状态;
# 否则 pypto 可能复用上一次缓存的 output_* 子目录, 导致二次运行找不到文件.
output_dir = os.path.abspath(f"prof_generation_output_case{case_idx}")
os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
os.makedirs(output_dir, exist_ok=True)
try:
    if hasattr(pypto, "pypto_impl") and hasattr(pypto.pypto_impl, "ResetLog"):
        pypto.pypto_impl.ResetLog("")
except Exception as _e:
    print(f"[WARN] pypto ResetLog failed: {_e}")

# PyPTO profile 不 warmup; jit kernel autotune 已在 forward 内自行收敛.
pypto_benchmark_fn()

trace_paths, kernel_dirs = _find_kernel_swimlanes(output_dir)
print(f"[INFO] PyPTO swimlane traces found: n={len(trace_paths)}")
for _i, _tp in enumerate(trace_paths):
    print(f"  [{_i}] {_tp}")

if len(trace_paths) == 0:
    print("[ERROR] No swimlane trace produced; profile cannot continue.")
    print("PROFILE_RESULT_GEN_US: inf")
elif len(trace_paths) > 1:
    # 多 kernel = 算子被拆成多个 jit, 违反"一个算子 = 一个融合 kernel"约定.
    # 这是作弊产物, 不再用 sum-of-span 把它当作合法性能数据回传.
    _dirs_csv = ",".join(os.path.basename(d) for d in kernel_dirs)
    print(f"CHEAT_MULTI_KERNEL: n={len(trace_paths)}; dirs={_dirs_csv}")
    print(
        "[ERROR] Detected multiple PyPTO jit kernels in one forward. "
        "PyPTO 算子要求是融合 kernel, 多 kernel = 没融合 = 作弊."
    )
    print("PROFILE_RESULT_GEN_US: inf")
else:
    execution_time_us = _span_of_events(_read_x_events(trace_paths[0]))
    print(f"[INFO] PyPTO swimlane span (single kernel): {execution_time_us:.2f}us")
    print(f"PROFILE_RESULT_GEN_US: {execution_time_us:.6f}")
'''
