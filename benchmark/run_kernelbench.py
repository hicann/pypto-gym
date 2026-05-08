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
"""KernelBench × pypto 端到端批处理执行逻辑.

使用示例:

公开入口统一走 ``python -m benchmark run --config configs/xxx.yaml``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from benchmark import case_loader, monitor, pypto_runner, verifier_runner, report
from benchmark.case_loader import CaseSpec, derive_op_name
from benchmark.opencode_exporter import OpencodeExportResult, append_export_result_to_log
from benchmark.process_registry import cleanup_registered_process_groups
from benchmark.pypto_runner import PyptoRunResult, PyptoRunStatus, run_pypto_workflow
from benchmark.report import CaseRunRecord, derive_overall_status, write_case_result, write_summary
from benchmark.verifier_runner import VerifierResult, VerifierStatus, run_verifier


logger = logging.getLogger("benchmark")

_ANSI_RESET = "\033[0m"
_PYPTO_STATUS_COLORS = {
    PyptoRunStatus.SUCCESS.value: "\033[1;32m",
    PyptoRunStatus.SKIPPED.value: "\033[1;36m",
    PyptoRunStatus.TIMEOUT.value: "\033[1;33m",
    PyptoRunStatus.ARTIFACT_MISSING.value: "\033[1;31m",
    PyptoRunStatus.BLOCKED.value: "\033[1;31m",
    PyptoRunStatus.OPENCODE_NOT_FOUND.value: "\033[1;31m",
    PyptoRunStatus.SUBPROCESS_ERROR.value: "\033[1;31m",
}


def _color_pypto_finish_log(status: str, text: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    color = _PYPTO_STATUS_COLORS.get(status)
    return f"{color}{text}{_ANSI_RESET}" if color else text


# ────────────────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────────────────

BENCHMARK_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BENCHMARK_ROOT / "configs" / "__default__.yaml"
DEFAULT_PYPTO_REPO_ROOT = BENCHMARK_ROOT / ".cache" / "pypto"
# Fork 模式下父进程先解析 artifact 路径并注入，子进程 _build_artifact_root() 可读此变量避免重复 UUID。
BENCHMARK_ARTIFACT_ROOT_ENV = "_BENCHMARK_ARTIFACT_ROOT_DIR"
# 仅由 ``benchmark.__main__`` 在 fork 子进程中设置，用于区分默认后台 run 与 --foreground。
_BENCHMARK_BACKGROUND_CHILD_ENV = "_BENCHMARK_BACKGROUND_CHILD"


def resolve_config_path(path: Path) -> Path:
    """Resolve benchmark-local config paths used by the public CLI."""
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path
    if path.parts and path.parts[0] == "configs":
        return Path(__file__).parent / path
    return path


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _merge_yaml_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_yaml_config(base_value, value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: Path) -> Dict[str, Any]:
    path = resolve_config_path(path)
    default_cfg = _read_yaml_mapping(DEFAULT_CONFIG_PATH)
    if path == DEFAULT_CONFIG_PATH:
        return default_cfg
    return _merge_yaml_config(default_cfg, _read_yaml_mapping(path))


def parse_csv_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_csv_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _expand_case_selector(selector: str) -> List[str]:
    selector = selector.strip()
    if not selector:
        return []
    if ":" not in selector:
        return [selector]

    start, end = (part.strip() for part in selector.split(":", 1))
    if start.isdigit() and end.isdigit():
        lo = int(start)
        hi = int(end)
        if lo > hi:
            raise ValueError(f"case range 必须递增: {selector}")
        return [str(i) for i in range(lo, hi + 1)]
    return [selector]


def parse_case_selectors(text: str) -> List[str]:
    selectors: List[str] = []
    for token in parse_csv_str_list(text):
        selectors.extend(_expand_case_selector(token))
    return selectors


def parse_cases_by_level(cases_text: str) -> Dict[str, Optional[List[str]]]:
    """解析 YAML 中的 cases 字段.

    每个 level 都必须直接在 cases 中写完整坐标:
    ``level1=1:21;level2=31,41:50;pto_case=1:6``.
    使用 ``level1=`` 可选择该 level 下全部 case.
    """
    if not cases_text:
        raise ValueError(
            "config.cases 必须显式指定 level, "
            "例如 'level1=19_ReLU' 或 'level1=1:21;pto_case=1:6'."
        )

    if "=" not in cases_text:
        raise ValueError(
            "config.cases 必须使用 'level=cases' 写法, "
            f"收到裸 selector: {cases_text!r}"
        )

    mapping: Dict[str, Optional[List[str]]] = {}
    for chunk in (part.strip() for part in cases_text.split(";") if part.strip()):
        if "=" not in chunk:
            raise ValueError(
                "分 level 指定 config.cases 时必须使用 'level=cases' 片段, "
                f"收到: {chunk!r}"
            )
        level, selectors_text = (part.strip() for part in chunk.split("=", 1))
        if not level:
            raise ValueError(f"config.cases 中存在空 level 片段: {chunk!r}")
        if level in mapping:
            raise ValueError(f"config.cases 中重复指定 level: {level}")
        mapping[level] = parse_case_selectors(selectors_text) if selectors_text else None
    if not mapping:
        raise ValueError(f"config.cases 未解析到任何 level: {cases_text!r}")
    return mapping


# ────────────────────────────────────────────────────────────
# 用例发现
# ────────────────────────────────────────────────────────────

def discover_cases(level_dir: Path, requested: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> List[Path]:
    """在 ``level_dir`` (即 ``KernelBench/<level>/``) 下查找用例.

    PyPTO 维护的 KernelBench fork (github.com/zwx2238/KernelBench @ e7f018e)
    用例文件形如 ``KernelBench/level1/19_ReLU.py`` 或
    ``KernelBench/pto_case/1_Foo.py``. 每个 .py
    即一个用例, ``case_id`` = 文件名 stem (不含 ``.py``).

    Args:
        level_dir: ``KernelBench/<level>/`` 目录绝对路径.
        requested: 用户指定的 case_id 子集; ``None`` 表示全选.
            可写完整 stem (``19_ReLU``) 或仅序号前缀 (``19``).
        limit: 截断数量, 仅在 ``requested is None`` 时生效.

    Returns:
        每个用例对应的 ``.py`` 文件绝对路径列表.

    Raises:
        FileNotFoundError: ``level_dir`` 不存在.
        ValueError: ``level_dir`` 下没有任何 .py; 或
            ``requested`` 中有 case 找不到.
    """
    if not level_dir.exists():
        raise FileNotFoundError(f"level dir 不存在: {level_dir}")
    if not level_dir.is_dir():
        raise ValueError(f"level dir 不是目录: {level_dir}")

    py_files = sorted(p for p in level_dir.iterdir() if p.is_file() and p.suffix == ".py")
    if not py_files:
        raise ValueError(
            f"{level_dir} 下找不到任何 .py 用例. 检查路径是否指向 "
            f"KernelBench/<level>/ (例如 .cache/KernelBench/KernelBench/level1 "
            f"或 .cache/KernelBench/KernelBench/pto_case)."
        )

    by_stem: Dict[str, Path] = {p.stem: p for p in py_files}
    by_index_prefix: Dict[str, Path] = {}
    for stem, path in by_stem.items():
        idx = stem.split("_", 1)[0]
        if idx.isdigit():
            by_index_prefix.setdefault(idx, path)

    if requested is not None:
        resolved: List[Path] = []
        missing: List[str] = []
        for r in requested:
            if r in by_stem:
                resolved.append(by_stem[r])
            elif r in by_index_prefix:
                resolved.append(by_index_prefix[r])
            else:
                missing.append(r)
        if missing:
            raise ValueError(
                f"以下 case 在 {level_dir} 中找不到: {missing}. "
                f"可用 case (前 10 个): {sorted(by_stem)[:10]}"
            )
        return resolved

    cases = list(by_stem.values())
    if limit is not None:
        cases = cases[:limit]
    return cases


# ────────────────────────────────────────────────────────────
# 单 case 流水线
# ────────────────────────────────────────────────────────────

def _write_case_phase(
    case_report_dir: Path,
    *,
    op_name: str,
    case_id: str,
    phase: str,
    status: str,
    level: str = "",
    message: str = "",
    pypto_status: str = "",
    verifier_status: str = "",
) -> None:
    """Write a tiny per-case phase marker for the live monitor."""
    payload = {
        "op_name": op_name,
        "case_id": case_id,
        "level": level,
        "phase": phase,
        "status": status,
        "message": message,
        "pypto_status": pypto_status,
        "verifier_status": verifier_status,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    case_report_dir.mkdir(parents=True, exist_ok=True)
    out = case_report_dir / "phase_state.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


def _copy_pypto_custom_to_report(op_dir: Path, case_report_dir: Path, op_name: str) -> Optional[Path]:
    """Copy ``custom/<op>`` artifacts into the case report dir, excluding bulky output* paths."""
    if not op_dir.is_dir():
        return None

    dest = case_report_dir / "custom" / op_name
    if dest.exists():
        shutil.rmtree(dest)

    def _ignore_output_paths(_dir: str, names: List[str]) -> set[str]:
        return {name for name in names if name.startswith("output")}

    shutil.copytree(op_dir, dest, ignore=_ignore_output_paths)
    return dest


@dataclass
class _RunCfg:
    pypto_repo_root: Path
    artifact_root_dir: Path
    run_subdir: str
    workdir_root: str
    opencode_bin: str
    opencode_model: str
    pypto_agent: str
    pypto_timeout: int
    pypto_output_format: str
    incomplete_workflow_retry: int
    arch: str
    arch_by_device: Dict[int, str]
    backend: str
    framework: str
    verify_timeout: int
    log_dir: Path
    report_dir: Path
    mode: str                      # correctness / performance / full
    skip_pypto_gen: bool
    force_regen: bool
    use_level_dirs: bool
    extra_verifier_config: Dict[str, Any]
    verifier_mode: str             # opencode / direct
    validator_agent: str           # opencode 模式: agent 名
    skill_timeout_sec: int         # opencode 模式: skill 子进程硬超时
    skill_retry: int               # opencode/API 层失败后的重试次数
    skill_retry_interval_sec: int  # opencode/API 层重试间隔秒数
    monitor_state_dir: Path
    monitor_poll_sec: int


def _normalize_ascend_arch(soc_version: Any) -> str:
    text = str(soc_version or "").strip()
    if not text:
        raise ValueError("empty soc version")
    normalized = re.sub(r"[\s_-]+", "", text).lower()
    match = re.search(r"ascend\d+[a-z]?\d*", normalized)
    if match:
        return match.group(0)
    match = re.search(r"\d+[a-z]\d*", normalized)
    if match:
        return f"ascend{match.group(0)}"
    raise ValueError(f"unrecognized soc version: {soc_version!r}")


def _parse_npu_smi_arches(text: str) -> Dict[int, str]:
    arches: Dict[int, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\|\s*(\d+)\s+([0-9A-Za-z]+)\s+\|", line)
        if not match:
            continue
        device_id = int(match.group(1))
        raw_name = match.group(2)
        try:
            arches[device_id] = _normalize_ascend_arch(raw_name)
        except ValueError:
            continue
    return arches


def _detect_ascend_arches_from_npu_smi() -> Dict[int, str]:
    proc = subprocess.run(
        ["npu-smi", "info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "npu-smi info failed")
    arches = _parse_npu_smi_arches(proc.stdout)
    if not arches:
        raise RuntimeError("npu-smi info 未解析到 NPU 型号")
    return arches


def detect_ascend_arch(device_id: Optional[int] = None) -> str:
    """Detect Ascend SOC version at runtime."""
    try:
        arches = _detect_ascend_arches_from_npu_smi()
        if device_id is None:
            return arches[sorted(arches)[0]]
        if device_id in arches:
            return arches[device_id]
    except Exception:
        arches = {}

    try:
        import torch_npu  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "无法自动识别 Ascend 架构: npu-smi/torch_npu 不可用. "
            "请先配置 NPU/CANN/torch_npu 环境."
        ) from exc

    try:
        return _normalize_ascend_arch(torch_npu.npu.get_soc_version())
    except Exception as exc:
        raise RuntimeError(
            "无法自动识别 Ascend 架构: torch_npu.npu.get_soc_version() 失败."
        ) from exc


def detect_ascend_arches(device_ids: List[int]) -> Dict[int, str]:
    """Detect Ascend arch once and return an arch per configured device."""
    if not device_ids:
        return {}
    try:
        detected = _detect_ascend_arches_from_npu_smi()
    except Exception as npu_smi_exc:
        try:
            fallback = detect_ascend_arch()
        except RuntimeError as torch_npu_exc:
            raise RuntimeError(
                f"无法自动识别 Ascend 架构: npu-smi 失败 ({npu_smi_exc}); "
                f"torch_npu 失败 ({torch_npu_exc})."
            ) from torch_npu_exc
        return {device_id: fallback for device_id in device_ids}

    missing = [device_id for device_id in device_ids if device_id not in detected]
    if missing:
        raise RuntimeError(
            "无法自动识别 Ascend 架构: npu-smi 未返回配置中的 device "
            f"{missing}; 已识别 device={sorted(detected)}."
        )
    return {device_id: detected[device_id] for device_id in device_ids}


async def run_one_case(case_path: Path, device_id: int, cfg: _RunCfg,
                       semaphore: Any) -> CaseRunRecord:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    logger.info("[%s] case start: device=%s source=%s", case_path.stem, device_id, case_path)

    logger.info("[%s] loading case + probing inputs", case_path.stem)
    case = case_loader.load_case(case_path, case_id=case_path.stem)
    op_name = case.op_name
    report_subdir = f"{case.level}/{op_name}" if cfg.use_level_dirs and case.level else op_name
    workdir_root = (
        f"{cfg.workdir_root}/{case.level}"
        if cfg.use_level_dirs and case.level else cfg.workdir_root
    )
    op_workdir = cfg.pypto_repo_root / workdir_root
    op_dir = op_workdir / op_name
    case_report_dir = cfg.report_dir / report_subdir
    case_report_dir.mkdir(parents=True, exist_ok=True)
    _write_case_phase(
        case_report_dir,
        op_name=op_name,
        case_id=case.case_id,
        phase="prepare",
        status="running",
        message="case loaded; writing SPEC/task_desc",
    )
    logger.info("[%s] case loaded: op=%s report_dir=%s", case.case_id, op_name, case_report_dir)

    # 1) 写 SPEC + task_desc
    logger.info("[%s] writing SPEC/task_desc into %s", case.case_id, op_workdir)
    case_loader.write_spec(case, op_workdir)
    case_loader.write_task_desc(case, op_workdir)

    # 2) Pypto 7-stage 工作流 (占设备号槽位)
    record = CaseRunRecord(
        op_name=op_name,
        case_id=case.case_id,
        source_file=case.source_file,
        level=case.level,
        report_subdir=report_subdir,
        started_at=started_at,
    )

    async with semaphore:
        if cfg.skip_pypto_gen:
            logger.info("[%s] skip pypto generation (--skip-pypto-gen)", case.case_id)
            pypto_log = case_report_dir / "pypto_run.log"
            pypto_log.write_text(
                "[pypto workflow skipped] --skip-pypto-gen\n",
                encoding="utf-8",
            )
            session_export = OpencodeExportResult(
                status="skipped",
                message="--skip-pypto-gen, 本次没有新的 OpenCode session 可导出.",
            )
            append_export_result_to_log(pypto_log, session_export, label="pypto")
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="pypto",
                status="skipped",
                message="--skip-pypto-gen",
            )
            pypto_result = PyptoRunResult(
                op_name=op_name,
                status=PyptoRunStatus.SKIPPED,
                workdir=op_dir,
                log_file=pypto_log,
                message="--skip-pypto-gen, 跳过 pypto 生成阶段.",
                opencode_session_export_message=session_export.message,
            )
        else:
            pypto_log = case_report_dir / "pypto_run.log"
            logger.info("[%s] launching pypto workflow; log=%s", case.case_id, pypto_log)
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="pypto",
                status="running",
                message=f"log={pypto_log}",
            )
            pypto_result = await asyncio.to_thread(
                run_pypto_workflow,
                op_name=op_name,
                pypto_repo_root=cfg.pypto_repo_root,
                workdir_root=workdir_root,
                opencode_bin=cfg.opencode_bin,
                opencode_model=cfg.opencode_model,
                agent=cfg.pypto_agent,
                timeout_sec=cfg.pypto_timeout,
                device_id=device_id,
                log_file=pypto_log,
                output_format=cfg.pypto_output_format,
                incomplete_workflow_retry=cfg.incomplete_workflow_retry,
                skip_if_done=not cfg.force_regen,
                task_desc_rel=f"{workdir_root}/{op_name}/task_desc.py",
                case_init_args_repr=case.init_args_repr,
                case_init_source=case.init_source,
                case_forward_source=case.forward_source,
                stop_event=_stop_event,
            )
            status_value = pypto_result.status.value
            finish_log = (
                f"[{case.case_id}] pypto workflow finished: "
                f"status={status_value} duration={pypto_result.duration_sec:.1f}s "
                f"message={pypto_result.message}"
            )
            logger.info("%s", _color_pypto_finish_log(status_value, finish_log))

        record.pypto_status = pypto_result.status.value
        record.pypto_message = pypto_result.message
        record.pypto_duration_sec = pypto_result.duration_sec
        record.pypto_log_file = str(pypto_result.log_file) if pypto_result.log_file else None
        record.pypto_attempt_log_files = [str(p) for p in pypto_result.attempt_log_files]
        record.pypto_retry_count = pypto_result.retry_count
        record.pypto_session_id = pypto_result.opencode_session_id
        record.pypto_session_md_file = (
            str(pypto_result.opencode_session_md_file)
            if pypto_result.opencode_session_md_file else None
        )
        record.pypto_session_export_message = pypto_result.opencode_session_export_message
        record.pypto_artifacts = {k: str(v) for k, v in pypto_result.artifacts.items()}
        try:
            copied_custom_dir = _copy_pypto_custom_to_report(op_dir, case_report_dir, op_name)
            if copied_custom_dir is not None:
                record.pypto_artifacts["report_custom_dir"] = str(copied_custom_dir)
                logger.info("[%s] copied pypto custom artifacts to %s",
                            case.case_id, copied_custom_dir)
        except Exception as exc:
            record.pypto_artifacts["report_custom_copy_error"] = str(exc)
            logger.warning("[%s] failed to copy pypto custom artifacts: %s",
                           case.case_id, exc)

        if not pypto_result.ok:
            record.overall_status = "pypto_failed"
            record.finished_at = dt.datetime.now().isoformat(timespec="seconds")
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="done",
                status="pypto_failed",
                message=pypto_result.message,
                pypto_status=pypto_result.status.value,
            )
            write_case_result(record, cfg.report_dir)
            logger.warning("[%s] case stop after pypto failure: status=%s",
                           case.case_id, record.pypto_status)
            return record

        # 3) KernelVerifier (仍占设备号槽位避免冲突)
        verifier_log = case_report_dir / "verifier.log"
        verifier_arch = cfg.arch_by_device.get(device_id, cfg.arch)
        try:
            logger.info("[%s] launching verifier; mode=%s log=%s",
                        case.case_id, cfg.verifier_mode, verifier_log)
            _write_case_phase(
                case_report_dir,
                op_name=op_name,
                case_id=case.case_id,
                phase="verifier",
                status="running",
                message=f"mode={cfg.verifier_mode}; log={verifier_log}",
                pypto_status=pypto_result.status.value,
            )
            verifier_result = await run_verifier(
                op_name=op_name,
                op_dir=op_dir,
                task_desc=case.task_desc,
                arch=verifier_arch,
                backend=cfg.backend,
                framework=cfg.framework,
                device_id=device_id,
                log_dir=cfg.log_dir,
                task_id=f"benchmark_{op_name}_{int(time.time()*1000)}",
                verify_timeout=cfg.verify_timeout,
                extra_config=cfg.extra_verifier_config,
                log_file=verifier_log,
                mode=cfg.mode,
                verifier_mode=cfg.verifier_mode,
                opencode_bin=cfg.opencode_bin,
                opencode_model=cfg.opencode_model,
                validator_agent=cfg.validator_agent,
                skill_timeout_sec=cfg.skill_timeout_sec,
                skill_retry=cfg.skill_retry,
                skill_retry_interval_sec=cfg.skill_retry_interval_sec,
            )
            logger.info("[%s] verifier finished: status=%s duration=%.1fs correctness=%s",
                        case.case_id, verifier_result.status.value,
                        verifier_result.duration_sec, verifier_result.correctness)
        except Exception as e:
            logger.exception("[%s] verifier raised exception: %s", case.case_id, e)
            verifier_result = VerifierResult(
                op_name=op_name,
                status=VerifierStatus.ERROR,
                message=f"run_verifier 抛异常: {e}",
                failure_category="system_error",
            )

    record.verifier_status = verifier_result.status.value
    record.verifier_message = verifier_result.message
    record.verifier_duration_sec = verifier_result.duration_sec
    record.verifier_log_file = str(verifier_result.log_file) if verifier_result.log_file else None
    record.verifier_session_id = verifier_result.opencode_session_id
    record.verifier_session_md_file = (
        str(verifier_result.opencode_session_md_file)
        if verifier_result.opencode_session_md_file else None
    )
    record.verifier_session_export_message = verifier_result.opencode_session_export_message
    record.correctness = verifier_result.correctness
    record.failure_category = verifier_result.failure_category
    record.perf_gen_time_us = verifier_result.perf_gen_time_us
    record.perf_base_time_us = verifier_result.perf_base_time_us
    record.perf_speedup = verifier_result.perf_speedup
    record.perf_roofline_time_us = verifier_result.perf_roofline_time_us
    record.perf_roofline_speedup = verifier_result.perf_roofline_speedup
    record.perf_message = verifier_result.perf_message
    record.overall_status = derive_overall_status(
        pypto_ok=pypto_result.ok,
        verifier_status=verifier_result.status.value,
        correctness=verifier_result.correctness,
    )
    record.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    _write_case_phase(
        case_report_dir,
        op_name=op_name,
        case_id=case.case_id,
        phase="done",
        status=record.overall_status,
        message=verifier_result.message,
        pypto_status=pypto_result.status.value,
        verifier_status=verifier_result.status.value,
    )
    write_case_result(record, cfg.report_dir)
    logger.info("[%s] case finished: overall=%s pypto=%s verifier=%s",
                case.case_id, record.overall_status,
                record.pypto_status, record.verifier_status)
    return record


# ────────────────────────────────────────────────────────────
# 主调度
# ────────────────────────────────────────────────────────────

async def run_batch(case_paths: List[Path], devices: List[int], concurrency: int,
                    cfg: _RunCfg) -> List[CaseRunRecord]:
    if not case_paths:
        return []
    if not devices:
        raise ValueError("devices 列表不能为空.")
    if concurrency < 1:
        raise ValueError(f"concurrency 必须 >= 1, 收到 {concurrency}")

    # 并发仅由 concurrency 控制; device 按 case 索引轮询 (idx % len(devices)),
    # 与 NPU 侧/工作流内部的占卡策略解耦, 不在此处用 len(devices) 夹逼并发度.
    semaphore = asyncio.Semaphore(concurrency)

    class _AcquiredSlot:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    async def _wrapper(idx: int, path: Path) -> CaseRunRecord:
        device_id = devices[idx % len(devices)]
        try:
            logger.info("[%s] waiting for execution slot (device=%s)", path.stem, device_id)
            async with semaphore:
                logger.info("[%s] acquired execution slot", path.stem)
                return await run_one_case(path, device_id, cfg, _AcquiredSlot())
        except Exception as e:
            logger.exception(f"[{path.name}] run_one_case 异常: {e}")
            op_name = derive_op_name(path.parent.name)
            return CaseRunRecord(
                op_name=op_name,
                case_id=path.parent.name,
                source_file=str(path),
                pypto_status="exception",
                pypto_message=str(e),
                overall_status="pypto_failed",
                started_at=dt.datetime.now().isoformat(timespec="seconds"),
                finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            )

    coros = [_wrapper(i, p) for i, p in enumerate(case_paths)]
    return await asyncio.gather(*coros)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def _resolve_pypto_repo_root(repo_root: Optional[Path]) -> Path:
    if repo_root:
        return repo_root.resolve()
    return DEFAULT_PYPTO_REPO_ROOT.resolve()


def _validate_pypto_from_yaml(pypto_yaml: Dict[str, Any]) -> None:
    """与 ``_build_cfg`` 一致的 PyPTO 仓根校验, 供后台 fork 前 fail-fast."""
    repo_root_opt = _optional_path(pypto_yaml.get("repo_root"))
    pypto_repo_root = _resolve_pypto_repo_root(repo_root_opt)
    _validate_pypto_repo_root(pypto_repo_root, is_default=repo_root_opt is None)


def preflight_background_run(config_path: Path) -> None:
    """后台 ``run`` fork 子进程前调用, 覆盖 ``state.json`` 写入前的失败点.

    **须在** ``pre_resolve_state_dir`` **之后**调用: 未固定 ``output.root_dir`` 时父进程会注入
    ``_BENCHMARK_ARTIFACT_ROOT_DIR``, ``_build_cfg`` 才能与 fork 子进程共用同一路径.

    - ``dry_run``: 仅 ``_build_cfg``（含 PyPTO 仓根校验）
    - 非 dry: 另含 KernelBench 目录 / 用例解析 / NPU arch 检测等, 与 ``run_from_config`` 一致
    """
    yaml_cfg = load_yaml_config(config_path)
    cfg = _build_cfg(yaml_cfg)
    if bool(yaml_cfg.get("dry_run", False)):
        return
    _assert_run_batch_prerequisites(yaml_cfg, cfg)


def _validate_pypto_repo_root(pypto_repo_root: Path, *, is_default: bool) -> None:
    """Fail fast unless ``pypto_repo_root`` is a plausible PyPTO checkout (``.opencode/``)."""
    dl_hint = "bash benchmark/scripts/download_pypto.sh"
    if not pypto_repo_root.exists():
        if is_default:
            raise SystemExit(
                f"PyPTO 默认仓库路径不存在 ({pypto_repo_root}). "
                "请先下载 PyPTO 仓库。\n"
                f"  {dl_hint}"
            )
        raise SystemExit(f"配置的 pypto.repo_root 不存在: {pypto_repo_root}")
    if not pypto_repo_root.is_dir():
        if is_default:
            raise SystemExit(
                f"PyPTO 默认仓库路径存在但不是目录: {pypto_repo_root}\n"
                f"  {dl_hint}"
            )
        raise SystemExit(f"配置的 pypto.repo_root 不是目录: {pypto_repo_root}")
    opencode_dir = pypto_repo_root / ".opencode"
    if not opencode_dir.is_dir():
        msg = (
            f"路径存在但缺少 `.opencode/`，不是合法的 PyPTO 仓库: {pypto_repo_root}"
            if not is_default
            else (
                f"默认 PyPTO 路径不完整（缺少 `.opencode/`）: {pypto_repo_root}\n"
                f"  {dl_hint}"
            )
        )
        raise SystemExit(msg)


def _optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _build_artifact_root(output_yaml: Dict[str, Any]) -> Tuple[Path, str]:
    root_dir = _optional_path(output_yaml.get("root_dir"))
    if root_dir is not None:
        resolved = root_dir.resolve()
        return resolved, resolved.name

    env_root = os.environ.get(BENCHMARK_ARTIFACT_ROOT_ENV, "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        return p, p.name

    run_subdir = f"Task_{uuid.uuid4().hex}"
    base_dir = _optional_path(output_yaml.get("base_dir")) or Path("benchmark_runs")
    artifact = (base_dir / run_subdir).resolve()
    return artifact, run_subdir


def pre_resolve_state_dir(config_path: Path) -> Path:
    """在 fork 子进程之前解析 ``monitor_state_dir`` (即 ``state.json`` 所在目录)。

    YAML 若未指定 ``output.root_dir``，则生成 ``base_dir/Task_<uuid>`` 并设置环境变量
    ``_BENCHMARK_ARTIFACT_ROOT_DIR``，供子进程与父进程对齐同一 artifact 路径。
    """
    yaml_cfg = load_yaml_config(config_path)
    output_yaml = yaml_cfg.get("output", {}) or {}

    root_dir = _optional_path(output_yaml.get("root_dir"))
    if root_dir is not None:
        return root_dir.resolve() / "state"

    env_root = os.environ.get(BENCHMARK_ARTIFACT_ROOT_ENV, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve() / "state"

    run_subdir = f"Task_{uuid.uuid4().hex}"
    base_dir = _optional_path(output_yaml.get("base_dir")) or Path("benchmark_runs")
    artifact_root = (base_dir / run_subdir).resolve()
    os.environ[BENCHMARK_ARTIFACT_ROOT_ENV] = str(artifact_root)
    return artifact_root / "state"


def _build_cfg(yaml_cfg: Dict[str, Any]) -> _RunCfg:
    pypto_yaml = yaml_cfg.get("pypto", {}) or {}
    repo_root_opt = _optional_path(pypto_yaml.get("repo_root"))
    pypto_repo_root = _resolve_pypto_repo_root(repo_root_opt)
    _validate_pypto_from_yaml(pypto_yaml)
    verifier_yaml = yaml_cfg.get("verifier", {}) or {}
    monitor_yaml = yaml_cfg.get("monitor", {}) or {}
    output_yaml = yaml_cfg.get("output", {}) or {}
    artifact_root_dir, run_subdir = _build_artifact_root(output_yaml)
    extra_verifier_config: Dict[str, Any] = {}
    verify_rtol = verifier_yaml.get("verify_rtol")
    verify_atol = verifier_yaml.get("verify_atol")
    if verify_rtol is not None:
        extra_verifier_config["verify_rtol"] = float(verify_rtol)
    if verify_atol is not None:
        extra_verifier_config["verify_atol"] = float(verify_atol)
    keep_artifacts = bool(verifier_yaml.get("keep_artifacts", False))
    extra_verifier_config["keep_artifacts"] = bool(keep_artifacts)

    return _RunCfg(
        pypto_repo_root=pypto_repo_root,
        artifact_root_dir=artifact_root_dir,
        run_subdir=run_subdir,
        workdir_root=pypto_yaml.get("workdir_root", "custom") or "custom",
        opencode_bin=pypto_yaml.get("opencode_bin", "") or "",
        opencode_model=pypto_yaml.get("opencode_model", "") or "",
        pypto_agent=pypto_yaml.get("agent", "pypto-op-orchestrator") or "pypto-op-orchestrator",
        pypto_timeout=int(pypto_yaml.get("timeout_sec", 1800) or 1800),
        pypto_output_format=pypto_yaml.get("output_format", "default") or "default",
        incomplete_workflow_retry=(
            int(pypto_yaml.get("incomplete_workflow_retry", 1) or 0)
        ),
        arch="",
        arch_by_device={},
        backend=verifier_yaml.get("backend", "ascend") or "ascend",
        framework=verifier_yaml.get("framework", "torch") or "torch",
        verify_timeout=int(verifier_yaml.get("verify_timeout", 900) or 900),
        log_dir=artifact_root_dir / "logs",
        report_dir=artifact_root_dir / "report",
        mode=verifier_yaml.get("mode", "correctness") or "correctness",
        skip_pypto_gen=bool(pypto_yaml.get("skip_pypto_gen", False)),
        force_regen=bool(pypto_yaml.get("force_regen", False)),
        use_level_dirs=False,
        extra_verifier_config=extra_verifier_config,
        verifier_mode=verifier_yaml.get("verifier_mode", "opencode") or "opencode",
        validator_agent=verifier_yaml.get("validator_agent", "pypto-kernel-validator") or "pypto-kernel-validator",
        skill_timeout_sec=int(verifier_yaml.get("skill_timeout_sec", 1800) or 1800),
        skill_retry=int(verifier_yaml.get("skill_retry", 2) or 0),
        skill_retry_interval_sec=(
            int(verifier_yaml.get("skill_retry_interval_sec", 600) or 0)
        ),
        monitor_state_dir=artifact_root_dir / "state",
        monitor_poll_sec=int(monitor_yaml.get("poll_sec", 2) or 2),
    )


def _assert_run_batch_prerequisites(yaml_cfg: Dict[str, Any], cfg: _RunCfg) -> Tuple[Path, List[Path], List[str], List[int], int]:
    """``monitor.write_state`` 之前会触发的校验（bench / cases / arch）。失败则 ``SystemExit``。"""
    cfg.use_level_dirs = True
    yaml_bench = (yaml_cfg.get("bench_dir") or "").strip()
    if yaml_bench:
        bench_dir = Path(yaml_bench).expanduser()
    else:
        bench_dir = Path(__file__).resolve().parent / ".cache" / "KernelBench" / "KernelBench"
    if not bench_dir.is_absolute():
        cwd_candidate = (Path.cwd() / bench_dir).resolve()
        repo_candidate = (cfg.pypto_repo_root / bench_dir).resolve()
        if cwd_candidate.exists():
            bench_dir = cwd_candidate
        elif repo_candidate.exists():
            bench_dir = repo_candidate
        else:
            bench_dir = cwd_candidate

    try:
        cases_text = str(yaml_cfg.get("cases") or "")
        cases_by_level = parse_cases_by_level(cases_text)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    levels = list(cases_by_level.keys())
    limit = yaml_cfg.get("limit")
    case_limit = int(limit) if limit is not None else None

    case_paths: List[Path] = []
    for level in levels:
        level_dir = bench_dir / level
        if not level_dir.exists():
            raise SystemExit(
                f"level dir 不存在: {level_dir}\n"
                f"  bench_dir = {bench_dir}\n"
                f"  level     = {level}\n"
                f"请先运行: bash benchmark/scripts/download_kernelbench.sh"
            )
        requested = cases_by_level.get(level)
        case_paths.extend(discover_cases(level_dir, requested=requested, limit=case_limit))
    if not case_paths:
        raise SystemExit(f"未发现可执行用例 (bench_dir={bench_dir}, levels={levels})")

    device_cfg = yaml_cfg.get("devices") or [0]
    if isinstance(device_cfg, str):
        devices = parse_csv_int_list(device_cfg)
    else:
        devices = [int(item) for item in device_cfg]
    concurrency = int(yaml_cfg.get("concurrency") or len(devices))
    if cfg.backend == "ascend":
        try:
            cfg.arch_by_device = detect_ascend_arches(devices)
            cfg.arch = ",".join(
                f"{device_id}:{arch}"
                for device_id, arch in sorted(cfg.arch_by_device.items())
            )
        except RuntimeError as e:
            raise SystemExit(str(e)) from e

    return bench_dir, case_paths, levels, devices, concurrency


def _setup_logging(level: str) -> None:
    norm = "WARNING" if level == "WARN" else level
    logging.basicConfig(
        level=getattr(logging, norm),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_stop_event: Optional[threading.Event] = None


def _signal_handler(signum: int, _frame: Any) -> None:
    """SIGINT/SIGTERM/SIGHUP 转为 KeyboardInterrupt, 并通知子线程停止."""
    global _stop_event
    if _stop_event is not None:
        _stop_event.set()
    raise KeyboardInterrupt()


def _drain_task_exception(task: "asyncio.Task[Any]") -> None:
    """Mark a finished task exception as retrieved to avoid asyncio shutdown noise."""
    if not task.done() or task.cancelled():
        return
    try:
        task.exception()
    except (KeyboardInterrupt, SystemExit):
        pass


def _run_interruptible(coro: Any) -> Any:
    """Run a coroutine and cleanly drain the main task after KeyboardInterrupt."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        task = loop.create_task(coro)
        try:
            return loop.run_until_complete(task)
        except KeyboardInterrupt:
            _drain_task_exception(task)
            if not task.done():
                task.cancel()
                try:
                    loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                except KeyboardInterrupt:
                    pass
                _drain_task_exception(task)
            raise
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@dataclass
class _MonitorStateWorker:
    state: Dict[str, Any]
    report_keys: List[str]
    seen_pids: Dict[Tuple[str, str], Tuple[int, str]]
    stop_event: threading.Event
    thread: threading.Thread


def _empty_monitor_phase() -> Dict[str, Any]:
    return {
        "pid": None,
        "started_at": None,
        "ended_at": None,
        "duration_sec": 0.0,
        "status": None,
    }


def _build_initial_monitor_state(case_paths: List[Path], cfg: _RunCfg) -> tuple[Dict[str, Any], List[str]]:
    operators: List[Dict[str, Any]] = []
    report_keys: List[str] = []
    for path in case_paths:
        level = path.parent.name
        op_name = derive_op_name(path.stem)
        report_key = f"{level}/{op_name}" if cfg.use_level_dirs else op_name
        report_keys.append(report_key)
        operators.append({
            "case_id": path.stem,
            "level": level,
            "report_key": report_key,
            "op_name": op_name,
            "phase": "pending",
            "phase_status": "pending",
            "phase_message": "",
            "opencode_pid": None,
            "phases": {
                "pypto": _empty_monitor_phase(),
                "verifier": _empty_monitor_phase(),
            },
            "started_at": None,
            "ended_at": None,
            "duration_sec": 0.0,
            "dev_status": "未开始",
            "result_status": None,
            "pypto_status": None,
        })

    state = {
        "main_pid": os.getpid(),
        "main_status": "正在进行",
        "main_exit_code": None,
        "artifact_root_dir": str(cfg.artifact_root_dir),
        "run_subdir": cfg.run_subdir,
        "started_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [path.stem for path in case_paths],
        "timeout_sec": cfg.pypto_timeout,
        "report_dir": str(cfg.report_dir),
        "log_dir": str(cfg.log_dir),
        "operators": operators,
    }
    return state, report_keys


def _start_monitor_state_worker(case_paths: List[Path], cfg: _RunCfg) -> _MonitorStateWorker:
    monitor.configure_state_dir(cfg.monitor_state_dir)
    state, report_keys = _build_initial_monitor_state(case_paths, cfg)
    seen_pids: Dict[Tuple[str, str], Tuple[int, str]] = {}
    stop_event = threading.Event()
    monitor.write_state(state)

    def _worker() -> None:
        while not stop_event.wait(cfg.monitor_poll_sec):
            monitor.refresh_run_state(
                state,
                root_pid=os.getpid(),
                report_dir=cfg.report_dir,
                report_keys=report_keys,
                timeout_sec=cfg.pypto_timeout,
                seen_pids=seen_pids,
            )
            monitor.write_state(state)

    thread = threading.Thread(target=_worker, name="benchmark-monitor-state", daemon=True)
    thread.start()
    return _MonitorStateWorker(
        state=state,
        report_keys=report_keys,
        seen_pids=seen_pids,
        stop_event=stop_event,
        thread=thread,
    )


def _finish_monitor_state_worker(
    worker: Optional[_MonitorStateWorker],
    cfg: _RunCfg,
    exit_code: int,
) -> None:
    if worker is None:
        return
    worker.stop_event.set()
    worker.thread.join(timeout=5)
    monitor.refresh_run_state(
        worker.state,
        root_pid=os.getpid(),
        report_dir=cfg.report_dir,
        report_keys=worker.report_keys,
        timeout_sec=cfg.pypto_timeout,
        seen_pids=worker.seen_pids,
        exit_code=exit_code,
    )
    monitor.write_state(worker.state)


def run_from_config(config_path: Path) -> int:
    global _stop_event
    _stop_event = threading.Event()

    # SIGINT: Ctrl+C; SIGTERM: 外部进程管理器可能发送; SIGHUP: 外层 shell 退出时发送.
    # 这些信号都转为 KeyboardInterrupt 让主协程有机会取消.
    # 同时 set _stop_event, 让 asyncio.to_thread 里的子线程从 sleep 中醒来.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)

    try:
        yaml_cfg = load_yaml_config(config_path)
        _setup_logging(str(yaml_cfg.get("log_level", "INFO") or "INFO"))

        cfg = _build_cfg(yaml_cfg)
        if bool(yaml_cfg.get("dry_run", False)):
            logger.info("dry-run: config=%s", config_path)
            logger.info("dry-run: report_dir=%s log_dir=%s mode=%s", cfg.report_dir, cfg.log_dir, cfg.mode)
            logger.info("dry-run: monitor_state_dir=%s", cfg.monitor_state_dir)
            if os.environ.get(_BENCHMARK_BACKGROUND_CHILD_ENV) == "1":
                cfg.monitor_state_dir.mkdir(parents=True, exist_ok=True)
                monitor.configure_state_dir(cfg.monitor_state_dir)
                now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dry_state: Dict[str, Any] = {
                    "main_pid": os.getpid(),
                    "main_status": "已完成",
                    "main_exit_code": 0,
                    "artifact_root_dir": str(cfg.artifact_root_dir),
                    "run_subdir": cfg.run_subdir,
                    "started_at": now,
                    "updated_at": now,
                    "cases": [],
                    "timeout_sec": cfg.pypto_timeout,
                    "report_dir": str(cfg.report_dir),
                    "log_dir": str(cfg.log_dir),
                    "operators": [],
                }
                monitor.write_state(dry_state)
            return 0

        bench_dir, case_paths, levels, devices, concurrency = _assert_run_batch_prerequisites(yaml_cfg, cfg)

        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"pypto_repo_root  = {cfg.pypto_repo_root}")
        logger.info(f"artifact_root    = {cfg.artifact_root_dir}")
        logger.info(f"run_subdir       = {cfg.run_subdir}")
        logger.info(f"bench_dir        = {bench_dir}")
        logger.info(f"levels           = {levels}")
        logger.info(f"cases            = {[p.stem for p in case_paths]}")
        logger.info(f"devices          = {devices}, concurrency = {concurrency}")
        logger.info(f"arch / backend   = {cfg.arch} / {cfg.backend}")
        logger.info(f"report_dir       = {cfg.report_dir}")
        logger.info(f"monitor_state_dir= {cfg.monitor_state_dir}")
        print(f"monitor_state_dir: {cfg.monitor_state_dir}", flush=True)
        print(f"monitor_command: {sys.executable} -m benchmark monitor {cfg.monitor_state_dir}", flush=True)

        monitor_worker = _start_monitor_state_worker(case_paths, cfg)
        exit_code = 1
        try:
            try:
                records = _run_interruptible(run_batch(case_paths, devices, concurrency, cfg))
            except KeyboardInterrupt:
                logger.warning("收到中断信号, 正在清理子进程并退出.")
                exit_code = 130
                return exit_code

            summary_paths = write_summary(
                records, cfg.report_dir,
                meta={
                    "bench_dir": str(bench_dir),
                    "level": ",".join(levels),
                    "levels": levels,
                    "kernelbench_branch": "pypto-supported-21fbe",
                    "arch": cfg.arch,
                    "backend": cfg.backend,
                    "framework": cfg.framework,
                    "mode": cfg.mode,
                    "verifier_mode": cfg.verifier_mode,
                    "validator_agent": cfg.validator_agent if cfg.verifier_mode == "opencode" else None,
                    "artifact_root_dir": str(cfg.artifact_root_dir),
                    "run_subdir": cfg.run_subdir,
                    "devices": devices,
                    "concurrency": concurrency,
                    "pypto_repo_root": str(cfg.pypto_repo_root),
                },
            )

            success_n = sum(1 for r in records if r.succeeded)
            total = len(records)
            logger.info(f"完成: {success_n}/{total} 通过")
            logger.info(f"summary: {summary_paths['md']}")
            exit_code = 0 if success_n == total else 1
            return exit_code
        finally:
            _finish_monitor_state_worker(monitor_worker, cfg, exit_code)
    finally:
        if _stop_event is not None:
            _stop_event.set()
        cleanup_registered_process_groups()


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> int:
    return run_from_config(Path(config_path))
