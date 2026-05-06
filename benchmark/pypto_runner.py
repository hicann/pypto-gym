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
"""pypto 7 阶段 agent 工作流子进程调度器.

通过 ``opencode run --agent pypto-op-orchestrator`` 在 pypto 仓内启动基于
7 阶段状态机的 agent 工作流, 等待算子产物落到 ``custom/{op}/`` 后返回结果.

设计要点:
- ``opencode`` 是外部 CLI; 我们不做 IPC, 只通过 ``stdout/stderr`` 抓日志, 通过
  目标产物文件存在性 + ``.orchestrator_state.json`` 状态判断成功/失败.
- 单 case 一次子进程调用; 多 case 并发交给上层调度.
- 设备号通过 ``TILE_FWK_DEVICE_ID`` 环境变量隔离.
- 子进程 stdout 通过 ``Popen`` + 后台线程**逐行落盘**到 ``log_file``,
  ``tail -f log_file`` 能实时看到 agent 进度 (默认不传 ``--print-logs``,
  避免 opencode 内部 server log 把真正的 agent 输出淹没).
- 主线程仅做 ``timeout_sec`` 硬墙轮询; 不做 early-stop, 让 agent 跑完它
  自己的状态机 (含 Stage 5↔6 修正循环; Stage 7 可按外部参数决定是否跳过
  迭代性能调优).
- prompt 含 ``{op}_pypto_impl.py`` (ModelNew 包装) 的硬约束; runner 不替
  agent 兜底生成. 若 agent 没产出, ``ARTIFACT_MISSING``, 由调用方决策.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from benchmark.opencode_exporter import (
    OpencodeExportResult,
    append_export_result_to_log,
    export_session_from_log,
    make_session_title,
)
from benchmark.process_registry import register, terminate_process_group, unregister


class PyptoRunStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ARTIFACT_MISSING = "artifact_missing"
    BLOCKED = "blocked"
    OPENCODE_NOT_FOUND = "opencode_not_found"
    SUBPROCESS_ERROR = "subprocess_error"


_REQUIRED_ORCHESTRATOR_STAGES = tuple(str(i) for i in range(1, 8))


@dataclass
class PyptoRunResult:
    op_name: str
    status: PyptoRunStatus
    workdir: Path
    artifacts: Dict[str, Path] = field(default_factory=dict)
    log_file: Optional[Path] = None
    attempt_log_files: List[Path] = field(default_factory=list)
    retry_count: int = 0
    duration_sec: float = 0.0
    message: str = ""
    orchestrator_state: Optional[dict] = None
    opencode_session_id: Optional[str] = None
    opencode_session_md_file: Optional[Path] = None
    opencode_session_export_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (PyptoRunStatus.SUCCESS, PyptoRunStatus.SKIPPED)

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "status": self.status.value,
            "workdir": str(self.workdir),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "log_file": str(self.log_file) if self.log_file else None,
            "attempt_log_files": [str(p) for p in self.attempt_log_files],
            "retry_count": self.retry_count,
            "duration_sec": round(self.duration_sec, 2),
            "message": self.message,
            "orchestrator_state": self.orchestrator_state,
            "opencode_session_id": self.opencode_session_id,
            "opencode_session_md_file": (
                str(self.opencode_session_md_file)
                if self.opencode_session_md_file else None
            ),
            "opencode_session_export_message": self.opencode_session_export_message,
        }


# ────────────────────────────────────────────────────────────
# 产物清单
# ────────────────────────────────────────────────────────────

# 必须存在才算成功的最少产物 (按 plan 与 pypto-op-orchestrator 工件契约).
REQUIRED_ARTIFACTS = (
    "{op}_impl.py",
    "{op}_golden.py",
    "test_{op}.py",
)

# KernelBench 桥接所需的额外产物. agent 应根据外层 prompt 里的硬约束自行产出.
KERNELBENCH_ARTIFACTS = (
    "{op}_pypto_impl.py",
)


def expected_artifact_paths(op_name: str, op_dir: Path,
                            need_kernelbench: bool = True) -> Dict[str, Path]:
    """返回 ``{key: path}`` 形式的预期产物表."""
    out: Dict[str, Path] = {}
    for tpl in REQUIRED_ARTIFACTS:
        rel = tpl.format(op=op_name)
        out[rel] = op_dir / rel
    if need_kernelbench:
        for tpl in KERNELBENCH_ARTIFACTS:
            rel = tpl.format(op=op_name)
            out[rel] = op_dir / rel
    out["SPEC.md"] = op_dir / "SPEC.md"
    return out


def all_artifacts_present(artifacts: Dict[str, Path]) -> List[str]:
    """返回缺失的 artifact 文件名列表; 空列表表示齐全."""
    return [k for k, p in artifacts.items() if not p.exists()]


# ────────────────────────────────────────────────────────────
# Prompt 渲染
# ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
请以 pypto-op-orchestrator 角色为算子 `{op_name}` 跑完本次 benchmark 所需的 pypto 工作流.

工作目录: `{op_dir_rel}/`
SPEC.md (已就绪, 请直接读取并按其内容推进): `{op_dir_rel}/SPEC.md`
KernelBench task_desc (已就绪, 需要用它校准包装接口): `{task_desc_rel}`

请严格按 pypto 现有 7 阶段产出以下标准产物 (按 pypto-op-orchestrator 自带规范):
  - SPEC.md (已存在)
  - API_REPORT.md
  - DESIGN.md
  - {op_name}_golden.py
  - {op_name}_impl.py            (必须导出 {op_name}_wrapper)
  - test_{op_name}.py
  - README.md
  - .orchestrator_state.json     (由 state_transition 维护的阶段状态文件)

================================================================
【外部桥接附加要求 -- 仅本次任务额外完成, 不要修改 pypto 内置 SKILL/agent】
================================================================

下游 KernelVerifier 的真实调用约定不是“把所有参数拍平成一个 wrapper 调用”,
而是严格遵循 KernelBench task_desc 的两段式接口:

----------------------------------------------------------------------
init_inputs = get_init_inputs()
raw_inputs = get_inputs()
model = ModelNew(*init_inputs)
outputs = model(*raw_inputs)
----------------------------------------------------------------------

本 case 的 task_desc 关键信息:
- task_desc 文件: `{task_desc_rel}`
- get_init_inputs() 探针 repr: `{init_args_repr}`

Model.__init__ 参考源码:
----------------------------------------------------------------------
{model_init_source}
----------------------------------------------------------------------

Model.forward / __call__ 参考源码:
----------------------------------------------------------------------
{forward_source}
----------------------------------------------------------------------

Stage 5 完成且自验证通过后, 请在同一目录 `{op_dir_rel}/` 下额外生成一个文件
`{op_name}_pypto_impl.py`. 该文件是给下游 KernelBench 风格评测器
(KernelVerifier) 的入口. 你必须根据上面的 task_desc 约定自行确定
`ModelNew.__init__` / `ModelNew.forward` 与 `{op_name}_wrapper` 的绑定方式.

建议结构如下 (import 关系必须保持, 但 `forward()` 内的实参绑定可按 task_desc
和 wrapper 签名调整, 不要求逐字照抄):

----------------------------------------------------------------------
import torch
import torch.nn as nn

from {op_name}_impl import {op_name}_wrapper


class ModelNew(nn.Module):
    \"\"\"KernelBench-compatible entry, forwards to {op_name}_wrapper.\"\"\"

    def __init__(self, ...):
        super().__init__()
        # 必须复刻 task_desc.Model 中会进入 state_dict 的子模块/Parameter/buffer 注册路径.
        # 例如 task_desc.Model 使用 self.gemm = nn.Linear(...), ModelNew 也应注册
        # self.gemm, 并在 forward 中传 self.gemm.weight / self.gemm.bias 给 wrapper.
        ...

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # 只整理参数并转发给 {op_name}_wrapper; 不在这里复制 kernel 计算逻辑.
        return {op_name}_wrapper(...)
----------------------------------------------------------------------

ModelNew 文件硬约束:
1. 必须是新文件 `{op_name}_pypto_impl.py`, 与 `{op_name}_impl.py` 同目录.
2. 必须 `from {op_name}_impl import {op_name}_wrapper` (不得内联 wrapper 实现).
3. `ModelNew` 必须继承 `torch.nn.Module`, `__init__` 必须调用 `super().__init__()`;
   下游 verifier 会调用 `ModelNew(*get_init_inputs()).to(device)`, 因此缺少 `.to()`
   视为接口错误, 不得通过自检.
4. `ModelNew(*get_init_inputs()).forward(*get_inputs())` 必须与 task_desc 严格兼容.
5. 如果 task_desc 把参数拆成 init 和 forward 两部分, 必须保持这个拆分;
   禁止要求下游验证器把 init_args 和 raw_inputs 错误拍平成一个外部调用接口.
6. `ModelNew.state_dict().keys()` 必须与 `task_desc.Model.state_dict().keys()` 完全一致,
   且每个 key 的 shape / dtype 必须一致; 否则 verifier 加载权重时会出现 missing /
   unexpected keys, 导致实现模型使用随机权重、精度失败.
7. 必须复刻 task_desc.Model 中带参数或 buffer 的 submodule / Parameter / buffer 注册路径;
   禁止为了适配 wrapper 随意把 `self.gemm.weight` / `self.gemm.bias` 拍平成
   `self.weight` / `self.bias` 这类不同 key. 若 wrapper 需要权重, 应在 `forward()`
   中从已注册的同名 submodule 取出后传入, 例如 `self.gemm.weight`.
8. 若 `{op_name}_wrapper` 需要 init 参数, 允许在 `ModelNew.forward()` 内部把
   保存的标量/配置参数按正确顺序转发给 wrapper; 但不得改变 state_dict 结构.
9. `forward` 内部禁止复制 kernel 逻辑或调用任何其他实现, 只能做参数整理并
   转发到 `{op_name}_wrapper`.
10. 不得修改已生成的 `{op_name}_impl.py` 的导出符号或函数签名, 除非是为修正
   与 task_desc 调用约定不兼容的问题.
11. 输出 shape 必须与 task_desc 参考 `Model` 完全一致; scalar `torch.Size([])`
   与 `torch.Size([1])` 不等价, 不得作为通过处理.

ModelNew 文件自检 (必须通过):
----------------------------------------------------------------------
python - <<'PY'
import importlib.util
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '{op_dir_rel}')

task_spec = importlib.util.spec_from_file_location('kb_task', '{task_desc_rel}')
task_mod = importlib.util.module_from_spec(task_spec)
task_spec.loader.exec_module(task_mod)

impl_spec = importlib.util.spec_from_file_location('kb_impl', '{op_dir_rel}/{op_name}_pypto_impl.py')
impl_mod = importlib.util.module_from_spec(impl_spec)
impl_spec.loader.exec_module(impl_mod)

calls = {{}}

def _stub(*args, **kwargs):
    calls['args_len'] = len(args)
    calls['kwargs_keys'] = sorted(kwargs)
    return None

impl_mod.{op_name}_wrapper = _stub
task_model = task_mod.Model(*task_mod.get_init_inputs())
model = impl_mod.ModelNew(*task_mod.get_init_inputs())
assert isinstance(model, nn.Module), "ModelNew must inherit torch.nn.Module"
assert hasattr(model, "to"), "ModelNew must support .to(device)"
task_state = task_model.state_dict()
impl_state = model.state_dict()
task_keys = list(task_state.keys())
impl_keys = list(impl_state.keys())
missing = [k for k in task_keys if k not in impl_state]
unexpected = [k for k in impl_keys if k not in task_state]
assert task_keys == impl_keys, (
    "ModelNew state_dict keys must match task_desc.Model; "
    f"missing={{missing}}, unexpected={{unexpected}}, "
    f"expected={{task_keys}}, actual={{impl_keys}}"
)
for key in task_keys:
    assert tuple(task_state[key].shape) == tuple(impl_state[key].shape), (
        f"state_dict shape mismatch for {{key}}: "
        f"expected {{tuple(task_state[key].shape)}}, got {{tuple(impl_state[key].shape)}}"
    )
    assert task_state[key].dtype == impl_state[key].dtype, (
        f"state_dict dtype mismatch for {{key}}: "
        f"expected {{task_state[key].dtype}}, got {{impl_state[key].dtype}}"
    )
model.to("cpu")
model(*task_mod.get_inputs())
print(type(model).__name__, calls)
PY
----------------------------------------------------------------------
预期行为: 不抛异常, 且输出里包含 `ModelNew`. 如果真实 wrapper 可运行,
还必须额外对比 `task_desc.Model(*get_init_inputs())(*get_inputs())` 与
`ModelNew(*get_init_inputs())(*get_inputs())` 的输出 shape, 包括 scalar rank.

精度二次校验硬约束:
- Stage 5/6 只有在重新运行测试且确定性解析到 `[PRECISION_PASS]` 时才允许判定通过.
- 测试超时、`no_marker`、权限拒绝、只做文件/接口结构检查、只做 stub 调用检查,
  都不得被解释为精度通过.
- 若二次校验未拿到有效 `[PRECISION_PASS]`, 必须通过 `state_transition` 将当前
  stage 标记为 failed, 不得继续推进到后续阶段.

================================================================
Stage 7 约束:
- Stage 7 必须按 pypto-op-orchestrator 自带规范正常执行性能调优与收尾.
- benchmark 不改写 PyPTO 原生 Stage 7 行为.

================================================================
其它约束:
- 所有产物落在 `{op_dir_rel}/`, 不要写到其它目录.
- 禁止清理 `/tmp/*`, `~/.cache/*`, `/home/*/.cache/*` 等全局/外部缓存目录.
  如怀疑编译缓存导致瞬态 AiCore Error, 只能清理 `{op_dir_rel}/` 内本算子的
  可再生产物后重跑; 不要请求 external_directory 权限.
- 走真实 NPU 验证 (有可用 NPU 时), 不要降级到 sim 模式.
- 完成或阻塞状态只以 `.orchestrator_state.json` 为准: 成功必须是 Stage 1-7
  全部 `completed`; 无法继续时必须把失败 stage 标记为 `failed`.

请立即开始, 不要再问我问题.
"""


def render_prompt(op_name: str, op_dir_rel: str, *,
                  task_desc_rel: Optional[str] = None,
                  init_args_repr: str = "[]",
                  model_init_source: str = "# (未提取到 __init__ 源码)",
                  forward_source: str = "# (未提取到 forward 源码)") -> str:
    task_desc_rel = task_desc_rel or f"{op_dir_rel}/task_desc.py"
    return _PROMPT_TEMPLATE.format(
        op_name=op_name,
        op_dir_rel=op_dir_rel,
        task_desc_rel=task_desc_rel,
        init_args_repr=init_args_repr,
        model_init_source=model_init_source,
        forward_source=forward_source,
    )


# ────────────────────────────────────────────────────────────
# 子进程调度
# ────────────────────────────────────────────────────────────

def _resolve_opencode(opencode_bin: str = "") -> Optional[str]:
    if opencode_bin:
        return opencode_bin if Path(opencode_bin).exists() else None
    return shutil.which("opencode")


def _read_orchestrator_state(op_dir: Path) -> Optional[dict]:
    state_file = op_dir / ".orchestrator_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _state_has_failed_stage(state: Optional[dict]) -> bool:
    if not isinstance(state, dict):
        return False
    stage_status = state.get("stage_status")
    if not isinstance(stage_status, dict):
        return False
    return any(str(v).lower() in {"failed", "blocked", "cancelled"} for v in stage_status.values())


def _state_all_stages_completed(state: Optional[dict]) -> bool:
    if not isinstance(state, dict):
        return False
    stage_status = state.get("stage_status")
    if not isinstance(stage_status, dict) or not stage_status:
        return False
    return all(str(stage_status.get(k)).lower() == "completed" for k in _REQUIRED_ORCHESTRATOR_STAGES)


def _state_incomplete_without_failure(state: Optional[dict]) -> bool:
    return not _state_has_failed_stage(state) and not _state_all_stages_completed(state)


def _blocked_message(state: Optional[dict]) -> str:
    if _state_has_failed_stage(state):
        return "PyPTO workflow 状态机存在 failed/blocked/cancelled 阶段."
    if not isinstance(state, dict):
        return "PyPTO workflow 缺少合法 .orchestrator_state.json, 不进入 verifier."
    return "PyPTO workflow 状态机未达到全阶段 completed, 不进入 verifier."


def _attempt_log_file(log_file: Optional[Path], attempt_index: int) -> Optional[Path]:
    if log_file is None or attempt_index <= 1:
        return log_file
    return log_file.with_name(f"{log_file.stem}.attempt{attempt_index}{log_file.suffix}")


def _attempt_logs(log_file: Optional[Path]) -> List[Path]:
    return [log_file] if log_file is not None else []


def _incomplete_retry_reason(timed_out: bool, returncode: Optional[int]) -> str:
    if timed_out:
        return "OpenCode 硬超时且 PyPTO 状态机未完成"
    if returncode == 0:
        return "OpenCode 正常退出但 PyPTO 状态机未完成"
    return f"OpenCode 异常退出 code={returncode} 且 PyPTO 状态机未完成"


# ────────────────────────────────────────────────────────────
# 后台 stdout 排水线程 (实时落盘)
# ────────────────────────────────────────────────────────────

def _stream_stdout(proc: subprocess.Popen, log_handle: Optional[TextIO]) -> None:
    """把 ``proc.stdout`` 逐行写到 ``log_handle`` (实时落盘 + flush).

    设计:
    - 与 ``Popen(bufsize=1, text=True)`` 配套, 让 ``tail -f log_file`` 能实时
      看到 agent 输出. ``subprocess.run`` 默认全缓冲, 看不到进度 — 这是 runner
      改用 ``Popen`` 的唯一动机.
    - 子进程被 SIGTERM/SIGKILL 时 ``readline`` 会拿到 ``''`` 然后退出, 不需要
      额外信号.
    """
    if log_handle is None or proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            log_handle.write(line)
            log_handle.flush()
    except (ValueError, OSError):
        pass


def _kill_process_group(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    """SIGTERM 子进程组, 不退就 SIGKILL.

    ``Popen(start_new_session=True)`` 让 opencode 跟它 spawn 的所有 node 子进程
    在同一个 pgid 下, 一次性能干掉.
    """
    terminate_process_group(proc, grace_sec=grace_sec)


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

# 主循环轮询 (检查进程是否退出 / 是否硬超时) 频率.
_POLL_INTERVAL_SEC = 5


def run_pypto_workflow(
    op_name: str,
    pypto_repo_root: Path,
    workdir_root: str = "custom",
    *,
    opencode_bin: str = "",
    opencode_model: str = "",
    agent: str = "pypto-op-orchestrator",
    timeout_sec: int = 7200,
    device_id: Optional[int] = None,
    log_file: Optional[Path] = None,
    output_format: str = "default",
    extra_env: Optional[Dict[str, str]] = None,
    skip_if_done: bool = True,
    need_kernelbench: bool = True,
    task_desc_rel: Optional[str] = None,
    case_init_args_repr: str = "[]",
    case_init_source: str = "# (未提取到 __init__ 源码)",
    case_forward_source: str = "# (未提取到 forward 源码)",
    stop_event: Optional[threading.Event] = None,
    incomplete_workflow_retry: int = 1,
    _attempt_index: int = 1,
) -> PyptoRunResult:
    """跑一次 pypto 7 阶段工作流.

    架构:
    - ``subprocess.Popen`` + 后台线程逐行排水 stdout 到 ``log_file``,
      ``tail -f log_file`` 立刻能看到 agent 进度.
    - 主线程每 ``_POLL_INTERVAL_SEC`` 秒检查 2 件事:
        (1) 子进程是否退出
        (2) 是否到 ``timeout_sec`` (硬墙)
      不做任何 artifact-based 的 early-stop — agent 自己有 Stage 5↔6 修正
      循环; Stage 7 始终按 PyPTO 原生工作流执行.
    - 子进程退出后, 用 ``expected_artifact_paths`` 检查产物齐全性. 缺
      ``{op}_pypto_impl.py`` 也算 ``ARTIFACT_MISSING`` — runner 不兜底,
      由 prompt 里的硬约束驱动 agent 自己产出.

    Args:
        op_name: 算子名 (= ``custom/{op_name}/`` 子目录名).
        pypto_repo_root: pypto 仓根, 子进程 cwd.
        workdir_root: 算子产物根目录 (相对 pypto 仓根).
        opencode_bin: opencode 可执行路径; 留空则按 PATH 查找.
        agent: opencode agent 名.
        opencode_model: 显式传给 ``opencode run -m`` 的模型名; 留空则沿用 CLI 当前默认配置.
        timeout_sec: 子进程整体超时 (硬墙). 默认 120 min, 覆盖 7 阶段
            含 Stage 7 性能调优 10 轮迭代.
        device_id: 注入 ``TILE_FWK_DEVICE_ID``; ``None`` 时不覆盖外部已设值.
        log_file: 子进程 stdout+stderr 落地; ``None`` 时不落盘.
        output_format: ``opencode run --format`` 参数.
        extra_env: 额外环境变量, 优先级最高.
        skip_if_done: 若产物齐全且 state file 存在则跳过.
        need_kernelbench: 是否要求 ``{op}_pypto_impl.py`` 也存在才算齐全.
        task_desc_rel: 传给 prompt 的 ``task_desc.py`` 相对路径.
        case_init_args_repr: 传给 prompt 的 ``get_init_inputs()`` 探针 repr.
        case_init_source: 传给 prompt 的 ``Model.__init__`` 源码摘要.
        case_forward_source: 传给 prompt 的 ``Model.forward/__call__`` 源码摘要.
        incomplete_workflow_retry: 若状态机未完成且无失败阶段,
            自动重跑 PyPTO workflow 的次数.

    Returns:
        ``PyptoRunResult``.
    """
    pypto_repo_root = pypto_repo_root.resolve()
    base_log_file = log_file
    log_file = _attempt_log_file(log_file, _attempt_index)
    op_dir = pypto_repo_root / workdir_root / op_name
    op_dir_rel = f"{workdir_root}/{op_name}"
    artifacts = expected_artifact_paths(op_name, op_dir, need_kernelbench=need_kernelbench)

    # 断点续跑: 已完成则跳过
    if skip_if_done:
        missing = all_artifacts_present(artifacts)
        state = _read_orchestrator_state(op_dir)
        if not missing and _state_has_failed_stage(state):
            session_export = OpencodeExportResult(
                status="skipped",
                message="PyPTO 工作流已阻塞, 本次没有新的 OpenCode session 可导出.",
            )
            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(
                    "[pypto workflow skipped] 已有产物齐全, 但 workflow 处于阻塞/失败状态; 不进入 verifier.\n",
                    encoding="utf-8",
                )
                append_export_result_to_log(log_file, session_export, label="pypto")
            return PyptoRunResult(
                op_name=op_name,
                status=PyptoRunStatus.BLOCKED,
                workdir=op_dir,
                artifacts=artifacts,
                log_file=log_file,
                attempt_log_files=_attempt_logs(log_file),
                duration_sec=0.0,
                message=_blocked_message(state),
                orchestrator_state=state,
                opencode_session_export_message=session_export.message,
            )
        state_success = _state_all_stages_completed(state)
        if not missing and state_success:
            session_export = OpencodeExportResult(
                status="skipped",
                message="PyPTO 工作流已跳过, 本次没有新的 OpenCode session 可导出.",
            )
            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(
                    "[pypto workflow skipped] 所有产物齐全, 且状态文件 Stage 1-7 均 completed.\n",
                    encoding="utf-8",
                )
                append_export_result_to_log(log_file, session_export, label="pypto")
            return PyptoRunResult(
                op_name=op_name,
                status=PyptoRunStatus.SKIPPED,
                workdir=op_dir,
                artifacts=artifacts,
                log_file=log_file,
                attempt_log_files=_attempt_logs(log_file),
                duration_sec=0.0,
                message="所有产物齐全, 且状态文件 Stage 1-7 均 completed; 跳过.",
                orchestrator_state=state,
                opencode_session_export_message=session_export.message,
            )

    opencode = _resolve_opencode(opencode_bin)
    if opencode is None:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.OPENCODE_NOT_FOUND,
            workdir=op_dir,
            message="opencode 可执行未找到; 请安装或在 YAML config 中指定 pypto.opencode_bin.",
        )

    spec_path = op_dir / "SPEC.md"
    if not spec_path.exists():
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.ARTIFACT_MISSING,
            workdir=op_dir,
            message=f"SPEC.md 不存在: {spec_path} (应由 case_loader 预先写入).",
        )

    prompt = render_prompt(
        op_name,
        op_dir_rel,
        task_desc_rel=task_desc_rel or f"{op_dir_rel}/task_desc.py",
        init_args_repr=case_init_args_repr,
        model_init_source=case_init_source,
        forward_source=case_forward_source,
    )

    # 注意: 不加 --print-logs! 它会把 opencode 内部 server/storage/agent 的
    # 每一笔操作都以单行平铺 JSON 倒进 stdout (一行常 5000+ 字符), 单 case
    # 跑下来 log 能轻松 20MB+ 还淹没掉 agent 真正的思考输出. 去掉之后只剩
    # agent message stream, 人能读, tail -f 也清爽.
    session_title = make_session_title(op_name, "pypto")
    cmd = [
        opencode, "run",
        "--dangerously-skip-permissions",
        "--agent", agent,
        "--format", output_format,
        "--title", session_title,
    ]
    if opencode_model:
        cmd.extend(["-m", opencode_model])
    cmd.append(prompt)

    env = os.environ.copy()
    if device_id is not None:
        env["TILE_FWK_DEVICE_ID"] = str(device_id)
    if extra_env:
        env.update(extra_env)

    log_handle: Optional[TextIO] = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8", buffering=1)
        log_handle.write(f"$ cd {pypto_repo_root}\n")
        log_handle.write(f"$ TILE_FWK_DEVICE_ID={env.get('TILE_FWK_DEVICE_ID', '<unset>')} "
                         f"{shlex.join(cmd[:-1])} <prompt>\n")
        log_handle.write(f"# opencode session title: {session_title}\n")
        log_handle.write("# (实时 stdout 从下行起追加; tail -f 可观察 agent 进度)\n")
        log_handle.flush()

    start = time.monotonic()
    deadline = start + timeout_sec
    timed_out = False
    interrupted = False

    if stop_event is not None and stop_event.is_set():
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.SUBPROCESS_ERROR,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=0.0,
            message="收到中断信号, 未启动 opencode.",
        )

    proc = subprocess.Popen(
        cmd,
        cwd=str(pypto_repo_root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register(proc)

    reader = threading.Thread(target=_stream_stdout, args=(proc, log_handle), daemon=True)
    reader.start()

    try:
        while True:
            if proc.poll() is not None:
                break

            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                if log_handle is not None:
                    log_handle.write(f"\n[TIMEOUT] {timeout_sec}s 硬超时, SIGTERM 进程组\n")
                _kill_process_group(proc)
                break

            # 使用 event.wait 代替 time.sleep, 让主线程 signal handler 可以通过
            # set event 立即中断子线程的 sleep, 从而执行 finally 清理 opencode.
            if stop_event is not None:
                if stop_event.wait(_POLL_INTERVAL_SEC):
                    interrupted = True
                    if log_handle is not None:
                        log_handle.write("\n[INTERRUPTED] 收到中断信号, SIGTERM 进程组\n")
                    _kill_process_group(proc)
                    break
            else:
                time.sleep(_POLL_INTERVAL_SEC)

        # 等 reader 把最后一段 stdout 排干
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
        reader.join(timeout=5)
    finally:
        # 防御: 若本线程被外部中断(KeyboardInterrupt / CancelledError),
        # 确保子进程不会变成孤儿.
        if proc.poll() is None:
            _kill_process_group(proc)
        unregister(proc)
        if log_handle is not None:
            try:
                log_handle.write(
                    f"\n[run finished, returncode={proc.returncode}, "
                    f"timed_out={timed_out}]\n"
                )
                log_handle.close()
            except (ValueError, OSError):
                pass

    duration = time.monotonic() - start
    if interrupted:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.SUBPROCESS_ERROR,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message="收到中断信号, 已清理 opencode 进程组.",
        )

    session_md_output = (
        log_file.parent / "pypto_session.md"
        if log_file else Path("pypto_session.md")
    )
    session_export = export_session_from_log(
        log_file=log_file,
        output_file=session_md_output,
        session_title=session_title,
        opencode_bin=opencode,
        cwd=pypto_repo_root,
    )
    append_export_result_to_log(log_file, session_export, label="pypto")

    session_id = session_export.session_id
    session_md_file = session_export.markdown_file if session_export.ok else None
    session_export_message = session_export.message
    state = _read_orchestrator_state(op_dir)
    missing = all_artifacts_present(artifacts)

    workflow_incomplete = _state_incomplete_without_failure(state)
    if workflow_incomplete and _attempt_index <= max(0, incomplete_workflow_retry):
        retry_reason = _incomplete_retry_reason(timed_out, proc.returncode)
        retry_result = run_pypto_workflow(
            op_name=op_name,
            pypto_repo_root=pypto_repo_root,
            workdir_root=workdir_root,
            opencode_bin=opencode_bin,
            opencode_model=opencode_model,
            agent=agent,
            timeout_sec=timeout_sec,
            device_id=device_id,
            log_file=base_log_file,
            output_format=output_format,
            extra_env=extra_env,
            skip_if_done=skip_if_done,
            need_kernelbench=need_kernelbench,
            task_desc_rel=task_desc_rel,
            case_init_args_repr=case_init_args_repr,
            case_init_source=case_init_source,
            case_forward_source=case_forward_source,
            stop_event=stop_event,
            incomplete_workflow_retry=incomplete_workflow_retry,
            _attempt_index=_attempt_index + 1,
        )
        retry_result.duration_sec += duration
        retry_result.retry_count += 1
        retry_result.attempt_log_files = _attempt_logs(log_file) + retry_result.attempt_log_files
        retry_result.message = (
            f"第{_attempt_index}次 {retry_reason}, 已自动重试; "
            f"{retry_result.message}"
        )
        return retry_result

    if timed_out:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.TIMEOUT,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message=f"opencode run 超时 (>{timeout_sec}s); 缺失产物: {missing or '(齐全)'}",
            orchestrator_state=state,
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    if missing:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.ARTIFACT_MISSING,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message=f"opencode 退出 code={proc.returncode}, 缺少产物: {missing}",
            orchestrator_state=state,
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    if proc.returncode != 0:
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.SUBPROCESS_ERROR,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message=f"opencode 异常退出 code={proc.returncode}, 但产物齐全 (可能仍可用).",
            orchestrator_state=state,
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    if _state_has_failed_stage(state):
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.BLOCKED,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message=_blocked_message(state),
            orchestrator_state=state,
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    if not _state_all_stages_completed(state):
        return PyptoRunResult(
            op_name=op_name,
            status=PyptoRunStatus.BLOCKED,
            workdir=op_dir,
            artifacts=artifacts,
            log_file=log_file,
            attempt_log_files=_attempt_logs(log_file),
            duration_sec=duration,
            message=_blocked_message(state),
            orchestrator_state=state,
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    return PyptoRunResult(
        op_name=op_name,
        status=PyptoRunStatus.SUCCESS,
        workdir=op_dir,
        artifacts=artifacts,
        log_file=log_file,
        attempt_log_files=_attempt_logs(log_file),
        duration_sec=duration,
        message="产物齐全, opencode 正常退出, 且状态文件 Stage 1-7 均 completed.",
        orchestrator_state=state,
        opencode_session_id=session_id,
        opencode_session_md_file=session_md_file,
        opencode_session_export_message=session_export_message,
    )


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    _default_repo_root = Path(__file__).resolve().parent / ".cache" / "pypto"
    parser = argparse.ArgumentParser(description="Run pypto 7-stage workflow for a single op")
    parser.add_argument("op_name")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="PyPTO 源码仓根; 缺省使用 benchmark/.cache/pypto",
    )
    parser.add_argument("--workdir-root", default="custom")
    parser.add_argument("--opencode-model", default="",
                        help="显式传给 opencode run -m 的模型名")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--no-skip", action="store_true",
                        help="即使产物齐全也强制重跑")
    parser.add_argument("--incomplete-workflow-retry", type=int, default=1,
                        help="PyPTO 状态机未完成且无失败阶段时自动重试次数")
    args = parser.parse_args()

    using_default_repo_root = args.repo_root is None
    repo_root = args.repo_root if args.repo_root is not None else _default_repo_root
    repo_root = repo_root.expanduser()
    try:
        resolved_root = repo_root.resolve()
    except OSError as exc:
        sys.stderr.write(f"错误: 无法解析 --repo-root {repo_root}: {exc}\n")
        return 2

    if not resolved_root.exists() or not resolved_root.is_dir():
        msg = f"错误: PyPTO 仓根不存在或不是目录: {resolved_root}\n"
        if using_default_repo_root:
            msg += (
                "提示: 默认缓存目录尚未初始化, 请运行:\n"
                "  bash benchmark/scripts/download_pypto.sh\n"
            )
        sys.stderr.write(msg)
        return 2

    opencode_cfg_dir = resolved_root / ".opencode"
    if not opencode_cfg_dir.is_dir():
        msg = (
            "错误: PyPTO 仓根缺少 .opencode/ 目录 "
            f"(不是可用的 PyPTO/OpenCode 工作区): {resolved_root}\n"
        )
        if using_default_repo_root:
            msg += (
                "提示: 默认缓存可能不完整或尚未克隆 PyPTO 仓, 请运行:\n"
                "  bash benchmark/scripts/download_pypto.sh\n"
            )
        sys.stderr.write(msg)
        return 2

    result = run_pypto_workflow(
        op_name=args.op_name,
        pypto_repo_root=resolved_root,
        workdir_root=args.workdir_root,
        opencode_model=args.opencode_model,
        timeout_sec=args.timeout_sec,
        device_id=args.device,
        log_file=args.log_file,
        skip_if_done=not args.no_skip,
        incomplete_workflow_retry=args.incomplete_workflow_retry,
    )
    sys.stdout.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_main_cli())
