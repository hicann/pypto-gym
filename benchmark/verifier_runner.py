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
"""桥接层验证编排封装 — 双模式 (opencode skill / direct).

输入: pypto 7 阶段产物 (``custom/{op}/{op}_impl.py`` + ``{op}_pypto_impl.py``)
       + 原始 KernelBench task_desc.
输出: 精度判定 + 反作弊判定 + (可选) 性能数据 + 完整日志.

两种执行模式:

- ``verifier_mode="opencode"`` (默认):
  spawn ``opencode run --agent pypto-kernel-validator``, cwd=pypto-gym 仓根,
  使用 gym 侧 ``.agents/skills/pypto-kernel-validate``.
  agent 按 SKILL 4 步执行 (脚本机械检测 → LLM 语义审阅 → 精度+性能 → JSON 报告),
  落 ``<output_dir>/skill_report.json``. 本 runner 读该 JSON 转 ``VerifierResult``.
  这是对外贡献候选物 + 默认评测路径.

- ``verifier_mode="direct"``:
  直接 await ``KernelVerifier.run/run_profile``, 不经 LLM, 没有语义层反作弊.
  仅用于 CI / 离线 dev 调试 — 跳过 opencode 的 LLM 开销.
  机械层 cheat 检测仍在跑 (``cheat_detector`` 在 KernelVerifier 之前过一遍),
  运行时多 kernel 也仍由 ``pypto_adapter`` 触发 ``CHEAT_MULTI_KERNEL``.

工件落盘:
- ``KernelVerifier`` 把 ``framework_code`` 写到 ``{op}_torch.py``, 把传入的
  ``source_files`` 原样写到 verify_dir. 这样 ``{op}_pypto_impl.py`` 中
  ``from {op}_impl import {op}_wrapper`` 可以自然解析, 不再拼接或改写源码.
  opencode 模式下统一 verifier CLI
  (``python -m benchmark.verifier verify``) 也走同样的多文件落盘.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from typing import Any, Dict, List, Optional, TextIO

from benchmark.opencode_exporter import (
    append_export_result_to_log,
    export_session_from_log,
    make_session_title,
)
from benchmark.process_registry import register, terminate_process_group, unregister


logger = logging.getLogger(__name__)


_VERIFIER_MODE_CHOICES = ("opencode", "direct")
_DEFAULT_VALIDATOR_AGENT = "pypto-kernel-validator"
_DEFAULT_VALIDATOR_SKILL = "pypto-kernel-validate"


# ────────────────────────────────────────────────────────────
# 数据模型
# ────────────────────────────────────────────────────────────

class VerifierStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BASELINE_FAILED = "baseline_failed"
    ERROR = "error"
    MISSING_INPUT = "missing_input"


@dataclass
class VerifierResult:
    op_name: str
    status: VerifierStatus
    correctness: Optional[bool] = None
    log_text: str = ""
    log_file: Optional[Path] = None
    duration_sec: float = 0.0
    message: str = ""
    failure_category: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    opencode_session_id: Optional[str] = None
    opencode_session_md_file: Optional[Path] = None
    opencode_session_export_message: str = ""

    # 性能字段 (mode=performance/full 才有数值; mode=correctness 全为 None).
    # 来源: KernelVerifier.run_profile() 返回 dict.
    perf_gen_time_us: Optional[float] = None       # agent 生成实现的执行时间 (us)
    perf_base_time_us: Optional[float] = None      # KernelBench Model 基线时间 (us)
    perf_speedup: Optional[float] = None           # base_time / gen_time, >1 = 更快
    perf_roofline_time_us: Optional[float] = None  # SOLAR fused roofline (可选)
    perf_roofline_speedup: Optional[float] = None
    perf_message: str = ""                         # perf 阶段独立的消息 (例如 "skipped" / "failed")

    @property
    def ok(self) -> bool:
        return self.status == VerifierStatus.PASSED

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "status": self.status.value,
            "correctness": self.correctness,
            "log_file": str(self.log_file) if self.log_file else None,
            "duration_sec": round(self.duration_sec, 2),
            "message": self.message,
            "failure_category": self.failure_category,
            "opencode_session_id": self.opencode_session_id,
            "opencode_session_md_file": (
                str(self.opencode_session_md_file)
                if self.opencode_session_md_file else None
            ),
            "opencode_session_export_message": self.opencode_session_export_message,
            "extra": self.extra,
            "perf": {
                "gen_time_us": self.perf_gen_time_us,
                "base_time_us": self.perf_base_time_us,
                "speedup": self.perf_speedup,
                "roofline_time_us": self.perf_roofline_time_us,
                "roofline_speedup": self.perf_roofline_speedup,
                "message": self.perf_message,
            },
        }


def collect_pypto_source_files(op_dir: Path, op_name: str) -> Dict[str, str]:
    """读取 verifier 需要的 PyPTO 源文件, 后续在 verify_dir 原样落盘."""
    impl_file = op_dir / f"{op_name}_impl.py"
    pypto_impl_file = op_dir / f"{op_name}_pypto_impl.py"

    if not impl_file.exists():
        raise FileNotFoundError(f"PyPTO impl 缺失: {impl_file}")

    source_files = {
        impl_file.name: impl_file.read_text(encoding="utf-8"),
    }
    if pypto_impl_file.exists():
        source_files[pypto_impl_file.name] = pypto_impl_file.read_text(encoding="utf-8")
        return source_files

    logger.warning(
        f"[{op_name}] {pypto_impl_file} 不存在; 仅返回 {op_name}_impl.py, "
        "依赖 verifier 的 wrapper-only fallback."
    )
    return source_files


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

_MODE_CHOICES = ("correctness", "performance", "full")


async def run_verifier(
    *,
    op_name: str,
    op_dir: Path,
    task_desc: str,
    arch: str = "ascend910b4",
    backend: str = "ascend",
    framework: str = "torch",
    device_id: int = 0,
    log_dir: Optional[Path] = None,
    task_id: str = "0",
    verify_timeout: int = 900,
    extra_config: Optional[Dict[str, Any]] = None,
    log_file: Optional[Path] = None,
    mode: str = "correctness",
    profile_warmup_times: Optional[int] = None,
    profile_run_times: Optional[int] = None,
    verifier_mode: str = "opencode",
    opencode_bin: str = "",
    opencode_model: str = "",
    validator_agent: str = _DEFAULT_VALIDATOR_AGENT,
    skill_timeout_sec: int = 1800,
    skill_retry: int = 2,
    skill_retry_interval_sec: int = 600,
    output_dir: Optional[Path] = None,
) -> VerifierResult:
    """异步入口: 跑一次精度 (+ 可选性能 + 反作弊) 验证.

    模式 (验证范围):
        - ``correctness``: 精度验证. ``perf_*`` 字段全部 None.
        - ``performance`` / ``full``: 精度 + 性能 (gen_time / base_time / speedup).
          精度不通过则跳过性能.

    执行模式 (``verifier_mode``):
        - ``opencode`` (默认): spawn ``opencode run --agent pypto-kernel-validator``,
          opencode 加载 ``.agents/skills/pypto-kernel-validate``, agent 在
          skill 引导下做"脚本机械检测 + LLM 语义审阅 + 精度 + 性能", 落
          ``<output_dir>/skill_report.json``. 本函数读 JSON 转 VerifierResult.
        - ``direct``: 跳过 opencode, 直接 await KernelVerifier (没有 LLM 语义层).
          仅 CI / 离线 dev 用.

    Args:
        op_name / op_dir / task_desc / arch / backend / framework / device_id /
        log_dir / task_id / verify_timeout / extra_config / log_file / mode /
        profile_warmup_times / profile_run_times: 与原签名一致.
        verifier_mode: 见上方"执行模式".
        opencode_bin: opencode 可执行路径; 空则按 PATH 查找. (仅 opencode 模式)
        opencode_model: 显式传给 ``opencode run -m`` 的模型名; 空则沿用默认配置. (仅 opencode 模式)
        validator_agent: opencode agent 名. (仅 opencode 模式)
        skill_timeout_sec: skill agent 子进程硬超时. (仅 opencode 模式)
        skill_retry: opencode/API 层失败后的重试次数; 业务验证失败不重试. (仅 opencode 模式)
        skill_retry_interval_sec: opencode/API 层重试间隔秒数. (仅 opencode 模式)
        output_dir: skill 报告输出目录. None 时自动取 ``op_dir/.skill_validate``.
            注意必须在 pypto 仓内或 op_dir 内, 否则 opencode sandbox 会以
            ``external_directory`` 拒绝写入. (仅 opencode 模式)

    Returns:
        ``VerifierResult``. opencode 模式额外把 ``skill_report.json`` 全文塞到
        ``extra["skill_report"]`` 字段, 调用方按需读.
    """
    if mode not in _MODE_CHOICES:
        raise ValueError(f"mode 必须是 {_MODE_CHOICES} 之一, 实际: {mode!r}")
    if verifier_mode not in _VERIFIER_MODE_CHOICES:
        raise ValueError(
            f"verifier_mode 必须是 {_VERIFIER_MODE_CHOICES} 之一, 实际: {verifier_mode!r}"
        )
    op_dir = op_dir.resolve()
    if not op_dir.exists():
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.MISSING_INPUT,
            message=f"op_dir 不存在: {op_dir}",
            failure_category=_failure_category_for_direct(
                VerifierStatus.MISSING_INPUT,
                f"op_dir 不存在: {op_dir}",
            ),
        )

    if verifier_mode == "opencode":
        return await _run_via_opencode_skill(
            op_name=op_name,
            op_dir=op_dir,
            task_desc=task_desc,
            arch=arch,
            device_id=device_id,
            verify_timeout=verify_timeout,
            extra_config=extra_config,
            log_file=log_file,
            mode=mode,
            opencode_bin=opencode_bin,
            opencode_model=opencode_model,
            validator_agent=validator_agent,
            skill_timeout_sec=skill_timeout_sec,
            skill_retry=skill_retry,
            skill_retry_interval_sec=skill_retry_interval_sec,
            output_dir=output_dir,
        )

    return await _run_direct(
        op_name=op_name,
        op_dir=op_dir,
        task_desc=task_desc,
        arch=arch,
        backend=backend,
        framework=framework,
        device_id=device_id,
        log_dir=log_dir,
        task_id=task_id,
        verify_timeout=verify_timeout,
        extra_config=extra_config,
        log_file=log_file,
        mode=mode,
        profile_warmup_times=profile_warmup_times,
        profile_run_times=profile_run_times,
    )


def run_verifier_sync(**kwargs) -> VerifierResult:
    """同步阻塞入口, 适合 CLI / sequential 测试."""
    return asyncio.run(run_verifier(**kwargs))


# ────────────────────────────────────────────────────────────
# direct 模式: 直调 KernelVerifier (无 LLM 语义层)
# ────────────────────────────────────────────────────────────

async def _run_direct(
    *,
    op_name: str,
    op_dir: Path,
    task_desc: str,
    arch: str,
    backend: str,
    framework: str,
    device_id: int,
    log_dir: Optional[Path],
    task_id: str,
    verify_timeout: int,
    extra_config: Optional[Dict[str, Any]],
    log_file: Optional[Path],
    mode: str,
    profile_warmup_times: Optional[int],
    profile_run_times: Optional[int],
) -> VerifierResult:
    try:
        source_files = collect_pypto_source_files(op_dir, op_name)
    except FileNotFoundError as e:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.MISSING_INPUT,
            message=str(e),
            failure_category=_failure_category_for_direct(
                VerifierStatus.MISSING_INPUT, str(e)
            ),
        )

    # lazy import: 让 case_loader 等纯解析模块在不需要 verifier 时也能 import
    # (verifier 模块会拉 torch / torch_npu, 解析路径不希望付这个开销).
    from benchmark.verifier import (
        KernelVerifier,
        get_worker_manager,
        load_config,
        register_local_worker,
    )

    try:
        config = load_config("pypto", backend=backend)
    except Exception as e:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            message=f"load_config('pypto') 失败: {e}",
            failure_category=_failure_category_for_direct(
                VerifierStatus.ERROR, f"load_config('pypto') 失败: {e}"
            ),
        )

    if log_dir is not None:
        config["log_dir"] = str(log_dir)
    config["verify_timeout"] = verify_timeout
    if extra_config:
        config.update(extra_config)

    # 注册 / 复用 LocalWorker.
    manager = get_worker_manager()
    has_match = await manager.has_worker(backend=backend, arch=arch)
    if not has_match:
        await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await manager.select(backend=backend, arch=arch)
    if worker is None:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            message=f"WorkerManager 中没有匹配 backend={backend}/arch={arch} 的 worker.",
            failure_category=_failure_category_for_direct(
                VerifierStatus.ERROR,
                f"WorkerManager 中没有匹配 backend={backend}/arch={arch} 的 worker.",
            ),
        )

    start = time.time()
    log_text = ""
    success = False
    err: Optional[str] = None
    perf_dict: Optional[Dict[str, Any]] = None
    perf_message = ""

    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc,
            task_id=task_id,
            framework=framework,
            dsl="pypto",
            backend=backend,
            arch=arch,
            config=config,
            worker=worker,
        )
        task_info = {"source_files": source_files}

        # 1) 精度验证 — 所有 mode 都跑.
        success, log_text = await verifier.run(task_info, current_step=0, device_id=device_id)

        # 2) 性能验证 — 仅在 mode=performance/full 且精度通过时跑.
        if mode in ("performance", "full"):
            if not success:
                perf_message = "skipped: correctness failed"
            else:
                profile_settings: Dict[str, Any] = {}
                if profile_warmup_times is not None:
                    profile_settings["warmup_times"] = profile_warmup_times
                if profile_run_times is not None:
                    profile_settings["run_times"] = profile_run_times
                try:
                    perf_dict = await verifier.run_profile(
                        task_info,
                        current_step=0,
                        device_id=device_id,
                        profile_settings=profile_settings,
                    )
                    if perf_dict is None or perf_dict.get("gen_time") is None:
                        perf_message = "run_profile 返回无效结果 (gen_time=None), 详见日志"
                except Exception as pe:
                    perf_message = f"run_profile 异常: {pe}"
                    logger.exception(perf_message)
    except Exception as e:
        err = f"KernelVerifier.run/run_profile 异常: {e}"
        logger.exception(err)
        log_text = log_text or err
    finally:
        try:
            await manager.release(worker)
        except Exception:
            logger.warning("release worker failed", exc_info=True)

    duration = time.time() - start

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(log_text or "", encoding="utf-8")
        except OSError as e:
            logger.warning(f"写 verifier log 失败 ({log_file}): {e}")

    perf_kw: Dict[str, Any] = {"perf_message": perf_message}
    if perf_dict is not None:
        perf_kw["perf_gen_time_us"] = perf_dict.get("gen_time")
        perf_kw["perf_base_time_us"] = perf_dict.get("base_time")
        perf_kw["perf_speedup"] = perf_dict.get("speedup")
        perf_kw["perf_roofline_time_us"] = perf_dict.get("roofline_time")
        perf_kw["perf_roofline_speedup"] = perf_dict.get("roofline_speedup")

    if err is not None:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            correctness=None,
            log_text=log_text,
            log_file=log_file,
            duration_sec=duration,
            message=err,
            failure_category=_failure_category_for_direct(
                VerifierStatus.ERROR, err, perf_message=perf_message
            ),
            **perf_kw,
        )

    st = VerifierStatus.PASSED if success else VerifierStatus.FAILED
    msg = "" if success else "KernelVerifier.run 返回 False, 详见日志."
    return VerifierResult(
        op_name=op_name,
        status=st,
        correctness=bool(success),
        log_text=log_text,
        log_file=log_file,
        duration_sec=duration,
        message=msg,
        failure_category=_failure_category_for_direct(
            st, msg, perf_message=perf_message
        ),
        **perf_kw,
    )


# ────────────────────────────────────────────────────────────
# opencode skill 模式: spawn opencode run --agent pypto-kernel-validator
# ────────────────────────────────────────────────────────────

_BENCHMARK_REPO_ROOT = Path(__file__).resolve().parents[1]
_OC_POLL_INTERVAL_SEC = 5

_VALIDATOR_PROMPT_TEMPLATE = """\
请按 SKILL `{skill_name}` 校验下面这个 PyPTO 算子产物.

参数:
- op_name = {op_name}
- op_dir = {op_dir}
- task_desc_file = {task_desc_file}
- output_dir = {output_dir}
- mode = {mode}
- device_id = {device_id}
- arch = {arch}
- verify_timeout = {verify_timeout}
- verify_rtol = {verify_rtol}
- verify_atol = {verify_atol}
- keep_artifacts = {keep_artifacts}

立即调用 skill({{ name: "{skill_name}" }}) 加载完整指引, 严格按 SKILL 4 步执行,
最终把 `skill_report.json` 写到 `{output_dir}/skill_report.json`.

不要在 chat 输出冗余总结, 只回一行:
  `skill_report.json written: <path>; final_verdict=<verdict>`
"""


def _resolve_opencode(opencode_bin: str = "") -> Optional[str]:
    if opencode_bin:
        return opencode_bin if Path(opencode_bin).exists() else None
    return shutil.which("opencode")


def _stream_to_file(proc: subprocess.Popen, log_handle: Optional[TextIO]) -> None:
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


def _kill_pgid(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    terminate_process_group(proc, grace_sec=grace_sec)


_FINAL_VERDICT_TO_STATUS = {
    "PASS": VerifierStatus.PASSED,
    "FAIL_CHEAT": VerifierStatus.FAILED,
    "FAIL_CORRECTNESS": VerifierStatus.FAILED,
    "FAIL_PERFORMANCE": VerifierStatus.FAILED,
    "BASELINE_FAILED": VerifierStatus.BASELINE_FAILED,
    "ERROR": VerifierStatus.ERROR,
}


def _coalesce_report_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _failure_category_from_final_verdict(final: str) -> str:
    """从 skill_report.final_verdict 机械推导 failure_category (含未知 verdict)."""
    final_u = (final or "ERROR").upper()
    if final_u == "PASS":
        return ""
    if final_u == "FAIL_CHEAT":
        return "cheat"
    if final_u == "FAIL_CORRECTNESS":
        return "correctness"
    if final_u == "FAIL_PERFORMANCE":
        return "performance"
    if final_u == "BASELINE_FAILED":
        return "baseline_failed"
    if final_u == "ERROR":
        return "error"
    return "unknown_verdict"


def _failure_category_for_direct(
    status: VerifierStatus,
    message: str,
    *,
    perf_message: str = "",
) -> str:
    if status == VerifierStatus.PASSED:
        return ""
    if status == VerifierStatus.MISSING_INPUT:
        return "missing_input"
    if status == VerifierStatus.FAILED:
        msg_l = (message or "").lower()
        perf_l = (perf_message or "").lower()
        if "run_profile" in perf_l or "run_profile" in msg_l:
            return "performance"
        return "correctness"
    if status == VerifierStatus.BASELINE_FAILED:
        return "baseline_failed"
    m = (message or "").lower()
    if "load_config" in m:
        return "config_error"
    if "worker" in m:
        return "worker_error"
    if "kernelverifier.run/run_profile" in m:
        return "runtime_error"
    return "error"


def _excerpt(text: str, max_chars: int = 4000) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-tail:]


def _attempt_log_file(log_file: Optional[Path], attempt_index: int) -> Optional[Path]:
    if log_file is None or attempt_index <= 1:
        return log_file
    return log_file.with_name(f"{log_file.stem}.attempt{attempt_index}{log_file.suffix}")


def _is_opencode_retryable(result: VerifierResult) -> bool:
    if result.status != VerifierStatus.ERROR:
        return False
    return bool(
        result.extra.get("opencode_no_skill_report")
        and result.extra.get("opencode_timed_out")
    )


def _skill_report_to_result(
    *, op_name: str, report: Dict[str, Any], log_text: str,
    log_file: Optional[Path], duration: float,
) -> VerifierResult:
    final = (report.get("final_verdict") or "ERROR").upper()
    status = _FINAL_VERDICT_TO_STATUS.get(final, VerifierStatus.ERROR)

    raw_fc = _coalesce_report_str(report.get("failure_category"))
    failure_category = (
        raw_fc if raw_fc else _failure_category_from_final_verdict(final)
    )

    correctness_block = report.get("correctness") or {}
    perf_block = report.get("performance") or {}
    correctness_ok: Optional[bool]
    c_status = (correctness_block.get("status") or "").lower()
    if c_status == "passed":
        correctness_ok = True
    elif c_status in ("failed", "error"):
        correctness_ok = False
    else:
        correctness_ok = None

    perf_kw: Dict[str, Any] = {
        "perf_message": (perf_block.get("status") or "")
                        + (f" | cheat_multi_kernel" if perf_block.get("cheat_multi_kernel") else ""),
        "perf_gen_time_us": perf_block.get("gen_time_us"),
        "perf_base_time_us": perf_block.get("base_time_us"),
        "perf_speedup": perf_block.get("speedup"),
        "perf_roofline_time_us": None,
        "perf_roofline_speedup": 0.0,
    }

    return VerifierResult(
        op_name=op_name,
        status=status,
        correctness=correctness_ok,
        log_text=log_text,
        log_file=log_file,
        duration_sec=duration,
        message=report.get("final_reasoning") or "",
        failure_category=failure_category,
        extra={"skill_report": report, "skill_final_verdict": final},
        **perf_kw,
    )


async def _run_via_opencode_skill(
    *,
    op_name: str,
    op_dir: Path,
    task_desc: str,
    arch: str,
    device_id: int,
    verify_timeout: int,
    extra_config: Optional[Dict[str, Any]],
    log_file: Optional[Path],
    mode: str,
    opencode_bin: str,
    opencode_model: str,
    validator_agent: str,
    skill_timeout_sec: int,
    skill_retry: int,
    skill_retry_interval_sec: int,
    output_dir: Optional[Path],
) -> VerifierResult:
    max_attempts = max(1, int(skill_retry) + 1)
    retry_interval = max(0, int(skill_retry_interval_sec))
    attempts: List[Dict[str, Any]] = []
    total_duration = 0.0

    for attempt_index in range(1, max_attempts + 1):
        attempt_log = _attempt_log_file(log_file, attempt_index)
        logger.info(
            "[%s] launching opencode validator attempt %s/%s; timeout=%ss",
            op_name, attempt_index, max_attempts, skill_timeout_sec,
        )
        result = await _run_via_opencode_skill_once(
            op_name=op_name,
            op_dir=op_dir,
            task_desc=task_desc,
            arch=arch,
            device_id=device_id,
            verify_timeout=verify_timeout,
            extra_config=extra_config,
            log_file=attempt_log,
            mode=mode,
            opencode_bin=opencode_bin,
            opencode_model=opencode_model,
            validator_agent=validator_agent,
            skill_timeout_sec=skill_timeout_sec,
            output_dir=output_dir,
        )
        total_duration += result.duration_sec
        attempts.append({
            "attempt": attempt_index,
            "status": result.status.value,
            "timeout": bool(result.extra.get("opencode_timed_out")),
            "retryable": _is_opencode_retryable(result),
            "log_file": str(result.log_file) if result.log_file else None,
            "session_id": result.opencode_session_id,
            "message": result.message,
        })
        result.extra["opencode_attempts"] = attempts
        result.extra["opencode_retry_count"] = attempt_index - 1
        result.duration_sec = total_duration

        if not _is_opencode_retryable(result) or attempt_index >= max_attempts:
            if attempt_index > 1:
                result.message = (
                    f"opencode validator attempts={attempt_index}/{max_attempts}; "
                    f"{result.message}"
                )
            return result

        logger.warning(
            "[%s] opencode validator attempt %s/%s failed with retryable API/timeout error; "
            "sleep %ss before retry",
            op_name, attempt_index, max_attempts, retry_interval,
        )
        if retry_interval > 0:
            await asyncio.sleep(retry_interval)
            total_duration += retry_interval

    return result


async def _run_via_opencode_skill_once(
    *,
    op_name: str,
    op_dir: Path,
    task_desc: str,
    arch: str,
    device_id: int,
    verify_timeout: int,
    extra_config: Optional[Dict[str, Any]],
    log_file: Optional[Path],
    mode: str,
    opencode_bin: str,
    opencode_model: str,
    validator_agent: str,
    skill_timeout_sec: int,
    output_dir: Optional[Path],
) -> VerifierResult:
    opencode = _resolve_opencode(opencode_bin)
    if opencode is None:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            message=(
                "opencode 可执行未找到; verifier_mode='opencode' 必须装 opencode CLI. "
                "用 verifier_mode='direct' 跳过 LLM 语义层 (无反作弊审阅)."
            ),
            failure_category="opencode_unavailable",
        )

    if output_dir is None:
        output_dir = (
            log_file.parent / ".skill_validate"
            if log_file is not None else
            _BENCHMARK_REPO_ROOT / "benchmark_runs" / ".skill_validate" / op_name
        ).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_desc_file = output_dir / f"{op_name}_task_desc.py"
    task_desc_file.write_text(task_desc, encoding="utf-8")
    skill_report_path = output_dir / "skill_report.json"
    if skill_report_path.exists():
        skill_report_path.unlink()

    prompt = _VALIDATOR_PROMPT_TEMPLATE.format(
        skill_name=_DEFAULT_VALIDATOR_SKILL,
        op_name=op_name,
        op_dir=str(op_dir),
        task_desc_file=str(task_desc_file),
        output_dir=str(output_dir),
        mode=mode,
        device_id=device_id,
        arch=arch,
        verify_timeout=verify_timeout,
        verify_rtol=(
            extra_config.get("verify_rtol")
            if extra_config and extra_config.get("verify_rtol") is not None
            else "default"
        ),
        verify_atol=(
            extra_config.get("verify_atol")
            if extra_config and extra_config.get("verify_atol") is not None
            else "default"
        ),
        keep_artifacts=(
            bool(extra_config.get("keep_artifacts"))
            if extra_config and extra_config.get("keep_artifacts") is not None
            else False
        ),
    )

    session_title = make_session_title(op_name, "verifier")
    cmd = [
        opencode, "run",
        "--dangerously-skip-permissions",
        "--agent", validator_agent,
        "--title", session_title,
    ]
    if opencode_model:
        cmd.extend(["-m", opencode_model])
    cmd.append(prompt)

    env = os.environ.copy()
    env["TILE_FWK_DEVICE_ID"] = str(device_id)

    log_handle: Optional[TextIO] = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8", buffering=1)
        log_handle.write(f"$ cd {_BENCHMARK_REPO_ROOT}\n")
        log_handle.write(
            f"$ TILE_FWK_DEVICE_ID={device_id} {shlex.join(cmd[:-1])} <prompt>\n"
        )
        log_handle.write(f"# opencode session title: {session_title}\n")
        log_handle.flush()

    start = time.monotonic()
    deadline = start + skill_timeout_sec
    timed_out = False

    proc = subprocess.Popen(
        cmd,
        cwd=str(_BENCHMARK_REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register(proc)

    reader = threading.Thread(target=_stream_to_file, args=(proc, log_handle), daemon=True)
    reader.start()

    try:
        while True:
            if proc.poll() is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                if log_handle is not None:
                    log_handle.write(f"\n[TIMEOUT] {skill_timeout_sec}s 硬超时, SIGTERM 进程组\n")
                _kill_pgid(proc)
                break
            await asyncio.sleep(_OC_POLL_INTERVAL_SEC)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_pgid(proc)
        reader.join(timeout=5)
    finally:
        if proc.poll() is None:
            _kill_pgid(proc)
        unregister(proc)
        if log_handle is not None:
            try:
                log_handle.write(
                    f"\n[opencode validator finished, returncode={proc.returncode}, "
                    f"timed_out={timed_out}]\n"
                )
                log_handle.close()
            except (ValueError, OSError):
                pass

    duration = time.monotonic() - start
    session_md_output = (
        log_file.parent / "verifier_session.md"
        if log_file else Path("verifier_session.md")
    )
    session_export = export_session_from_log(
        log_file=log_file,
        output_file=session_md_output,
        session_title=session_title,
        opencode_bin=opencode,
        cwd=_BENCHMARK_REPO_ROOT,
    )
    append_export_result_to_log(log_file, session_export, label="verifier")
    session_id = session_export.session_id
    session_md_file = session_export.markdown_file if session_export.ok else None
    session_export_message = session_export.message

    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file else ""

    if not skill_report_path.exists():
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            log_text=log_text,
            log_file=log_file,
            duration_sec=duration,
            message=(
                f"opencode validator 未产出 skill_report.json (timeout={timed_out}, "
                f"returncode={proc.returncode}); 检查 log_file 排错."
            ),
            failure_category=(
                "opencode_timeout" if timed_out else "skill_report_missing"
            ),
            extra={
                "opencode_no_skill_report": True,
                "opencode_timed_out": timed_out,
                "opencode_returncode": proc.returncode,
            },
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    try:
        report = json.loads(skill_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return VerifierResult(
            op_name=op_name,
            status=VerifierStatus.ERROR,
            log_text=log_text,
            log_file=log_file,
            duration_sec=duration,
            message=f"skill_report.json 解析失败: {e}",
            failure_category="invalid_skill_report",
            opencode_session_id=session_id,
            opencode_session_md_file=session_md_file,
            opencode_session_export_message=session_export_message,
        )

    result = _skill_report_to_result(
        op_name=op_name,
        report=report,
        log_text=_excerpt(log_text),
        log_file=log_file,
        duration=duration,
    )
    result.opencode_session_id = session_id
    result.opencode_session_md_file = session_md_file
    result.opencode_session_export_message = session_export_message
    return result


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run pypto self-contained KernelVerifier on pypto artifacts")
    parser.add_argument("op_name")
    parser.add_argument("--op-dir", type=Path, required=True,
                        help="custom/{op}/ 路径")
    parser.add_argument("--task-desc-file", type=Path, required=True,
                        help="KernelBench task_desc 文件 (case_loader 写出的 task_desc.py)")
    parser.add_argument("--arch", default="ascend910b4")
    parser.add_argument("--backend", default="ascend")
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--opencode-model", default="",
                        help="显式传给 opencode run -m 的模型名")
    parser.add_argument("--log-dir", type=Path, default=Path("~/pypto_bench_logs").expanduser())
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--task-id", type=str, default="0")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--mode", choices=_MODE_CHOICES, default="correctness",
                        help="correctness=精度; performance/full=精度+性能 (gen vs base speedup)")
    parser.add_argument("--profile-warmup", type=int, default=None)
    parser.add_argument("--profile-run", type=int, default=None)
    parser.add_argument("--verifier-mode", choices=_VERIFIER_MODE_CHOICES, default="opencode",
                        help="opencode=经 skill 走 LLM 语义层反作弊+精度+性能 (默认); "
                             "direct=直调 KernelVerifier, 跳过 LLM 语义层")
    parser.add_argument("--opencode-bin", default="",
                        help="opencode 可执行路径; 空则按 PATH 查找 (仅 opencode 模式)")
    parser.add_argument("--validator-agent", default=_DEFAULT_VALIDATOR_AGENT,
                        help="opencode agent 名 (仅 opencode 模式)")
    parser.add_argument("--skill-timeout", type=int, default=1800,
                        help="opencode validator 子进程硬超时, 秒 (仅 opencode 模式)")
    parser.add_argument("--skill-retry", type=int, default=2,
                        help="opencode/API 层失败后的重试次数; 业务验证失败不重试 (仅 opencode 模式)")
    parser.add_argument("--skill-retry-interval", type=int, default=600,
                        help="opencode/API 层重试间隔, 秒 (仅 opencode 模式)")
    parser.add_argument("--skill-output-dir", type=Path, default=None,
                        help="skill_report.json 落盘目录; 缺省取 log-file.parent (仅 opencode 模式)")
    parser.add_argument("--keep-artifacts",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="保留 KernelVerifier verify/profile 临时工作目录; 默认运行后自动清理")
    args = parser.parse_args()

    task_desc = args.task_desc_file.read_text(encoding="utf-8")
    result = run_verifier_sync(
        op_name=args.op_name,
        op_dir=args.op_dir,
        task_desc=task_desc,
        arch=args.arch,
        backend=args.backend,
        framework=args.framework,
        device_id=args.device,
        log_dir=args.log_dir,
        task_id=args.task_id,
        verify_timeout=args.timeout,
        log_file=args.log_file,
        mode=args.mode,
        profile_warmup_times=args.profile_warmup,
        profile_run_times=args.profile_run,
        extra_config=(
            {"keep_artifacts": args.keep_artifacts}
            if args.keep_artifacts is not None else None
        ),
        verifier_mode=args.verifier_mode,
        opencode_model=args.opencode_model,
        opencode_bin=args.opencode_bin,
        validator_agent=args.validator_agent,
        skill_timeout_sec=args.skill_timeout,
        skill_retry=args.skill_retry,
        skill_retry_interval_sec=args.skill_retry_interval,
        output_dir=args.skill_output_dir,
    )
    sys.stdout.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_main_cli())
