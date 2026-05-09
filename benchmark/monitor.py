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
"""算子运行监控看板.

用法:
    python3 -m benchmark monitor <state_dir>

设计要点:
- monitor 只读取 benchmark run 写出的 ``state.json``.
- benchmark 运行配置不再通过 monitor CLI 传入.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_wcswidth():
    install_cmd = [sys.executable, "-m", "pip", "install", "wcwidth"]

    def _manual_install_hint() -> str:
        return (
            f"请手动执行: {' '.join(install_cmd)}\n"
            f"如果当前 Python 没有 pip, 请先执行: {sys.executable} -m ensurepip --upgrade"
        )

    try:
        from wcwidth import wcswidth as imported_wcswidth
        return imported_wcswidth
    except ModuleNotFoundError as first_error:
        print(
            "[benchmark.monitor] Python package 'wcwidth' not found; "
            f"trying automatic install: {' '.join(install_cmd)}",
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as install_error:
            raise SystemExit(
                "缺少 Python 依赖 wcwidth, 且自动安装启动失败.\n"
                f"{_manual_install_hint()}\n"
                f"原始 import 错误: {first_error}\n"
                f"自动安装错误: {install_error}"
            ) from install_error

        if proc.returncode != 0:
            install_output = "\n".join(
                part.strip()
                for part in (proc.stdout, proc.stderr)
                if part and part.strip()
            )
            if len(install_output) > 2000:
                install_output = install_output[-2000:]
            raise SystemExit(
                "缺少 Python 依赖 wcwidth, 且自动安装失败.\n"
                f"{_manual_install_hint()}\n"
                f"pip 退出码: {proc.returncode}\n"
                f"pip 输出:\n{install_output}"
            ) from first_error

        try:
            from wcwidth import wcswidth as installed_wcswidth
            return installed_wcswidth
        except ModuleNotFoundError as second_error:
            raise SystemExit(
                "自动安装 wcwidth 已返回成功, 但当前 Python 仍无法 import.\n"
                f"{_manual_install_hint()}\n"
                f"import 错误: {second_error}"
            ) from second_error


wcswidth = _load_wcswidth()


DEFAULT_STATE_DIR = "/tmp/pypto-benchmark-monitor"
_POLL_SEC = 2
_OVER_THRESHOLD = 5400
_CONFIGURED_STATE_DIR = Path(DEFAULT_STATE_DIR)

_OP_RE_TITLE = re.compile(r"pypto-bench:(pypto|verifier):([A-Za-z0-9_.:-]+):")
_OP_RE_WORKFLOW = re.compile(r"算子 `(\w+)`")
_OP_RE_VERIFIER = re.compile(r"op_name\s*=\s*(\w+)")


# ────────────────────────────────────────────────────────────
# 路径约定
# ────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    return _CONFIGURED_STATE_DIR


def configure_state_dir(state_dir: Path) -> None:
    global _CONFIGURED_STATE_DIR
    _CONFIGURED_STATE_DIR = Path(state_dir).expanduser()


def _state_file() -> Path:
    return _state_dir() / "state.json"


# ────────────────────────────────────────────────────────────
# 进程工具 (Linux /proc)
# ────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _read_ppid(pid: int) -> int:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        i = data.rfind(")")
        if i < 0:
            return 0
        return int(data[i + 2:].split()[1])
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return 0


def _descendants(root: int) -> List[int]:
    """BFS 获取 root 的全部后代 PID."""
    ppid_map: Dict[int, int] = {}
    try:
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                pid = int(entry.name)
                ppid = _read_ppid(pid)
                if ppid:
                    ppid_map[pid] = ppid
    except OSError:
        return []

    result: List[int] = []
    queue = [root]
    visited = {root}
    while queue:
        parent = queue.pop(0)
        for pid, ppid in ppid_map.items():
            if ppid == parent and pid not in visited:
                visited.add(pid)
                result.append(pid)
                queue.append(pid)
    return result


def _detect_opencode_phase(cmd: str) -> str:
    if "pypto-bench:verifier:" in cmd or "pypto-kernel-validator" in cmd:
        return "verifier"
    if "pypto-bench:pypto:" in cmd or "pypto-op-orchestrator" in cmd:
        return "pypto"
    return "unknown"


def _find_opencode_procs(root_pid: int) -> Dict[Tuple[str, str], int]:
    """扫描 root_pid 的后代, 返回 {(op_name, phase): pid}."""
    out: Dict[Tuple[str, str], int] = {}
    for pid in _descendants(root_pid):
        cmd = _read_cmdline(pid)
        if "opencode" not in cmd:
            continue
        title_match = _OP_RE_TITLE.search(cmd)
        if title_match:
            phase, op_name = title_match.groups()
            out[(op_name, phase)] = pid
            continue

        phase = _detect_opencode_phase(cmd)
        m = _OP_RE_WORKFLOW.search(cmd) or _OP_RE_VERIFIER.search(cmd)
        if m and phase != "unknown":
            out[(m.group(1), phase)] = pid
    return out


# ────────────────────────────────────────────────────────────
# 状态文件 I/O
# ────────────────────────────────────────────────────────────

def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_state() -> Optional[Dict[str, Any]]:
    p = _state_file()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: Dict[str, Any]) -> None:
    p = _state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")
    tmp.rename(p)


def write_state(state: Dict[str, Any]) -> None:
    _write_state(state)


# ────────────────────────────────────────────────────────────
# 开发状态判定
# ────────────────────────────────────────────────────────────

def _elapsed_since(ts: str) -> float:
    try:
        return (dt.datetime.now() - dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        try:
            return (dt.datetime.now() - dt.datetime.fromisoformat(ts)).total_seconds()
        except ValueError:
            return 0.0


def _compute_dev_status(op: Dict[str, Any], timeout_sec: int) -> str:
    """基于进程与报告信息计算算子开发状态."""
    rs = op.get("result_status")
    if rs == "success":
        return "已完成"
    if rs == "pypto_failed":
        if op.get("pypto_status") == "timeout":
            return "PyPTO超时"
        return "PyPTO失败"
    if rs == "verify_failed":
        return "Verifier失败"
    if rs == "baseline_failed":
        return "Baseline失败"
    if rs == "verify_error":
        return "Verifier异常"

    phases = op.get("phases") or {}
    for phase, label in (("verifier", "Verifier验证中"), ("pypto", "PyPTO生成中")):
        info = phases.get(phase) if isinstance(phases, dict) else None
        if not isinstance(info, dict):
            continue
        pid = info.get("pid")
        started = info.get("started_at")
        elapsed = _elapsed_since(started) if started else 0.0
        if pid and _pid_alive(pid):
            if phase == "pypto" and elapsed >= timeout_sec:
                return "PyPTO超时"
            if phase == "pypto" and elapsed >= _OVER_THRESHOLD:
                return "PyPTO超过5400s"
            return label

    pid = op.get("opencode_pid")
    started = op.get("started_at")
    phase = op.get("phase")
    phase_status = op.get("phase_status")
    if phase == "prepare" and phase_status == "running":
        return "准备中"
    if phase == "pypto":
        if phase_status == "skipped":
            return "PyPTO跳过"
        if phase_status == "running":
            return "PyPTO生成中"
    if phase == "verifier" and phase_status == "running":
        return "Verifier验证中"

    if not pid and not started:
        return "未开始"

    elapsed = _elapsed_since(started) if started else 0.0
    if pid and _pid_alive(pid):
        if elapsed >= timeout_sec:
            return "超时"
        if elapsed >= _OVER_THRESHOLD:
            return "未超时但已超过5400s"
        return "正在进行"

    return "正在进行"


# ────────────────────────────────────────────────────────────
# 报告目录扫描
# ────────────────────────────────────────────────────────────

def _scan_reports(report_dir: str, report_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """扫描 report_dir 下各算子的 result.json."""
    out: Dict[str, Dict[str, Any]] = {}
    rd = Path(report_dir)
    if not rd.exists():
        return out
    for key in report_keys:
        rf = rd / key / "result.json"
        if rf.exists():
            try:
                out[key] = json.loads(rf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def _scan_phase_states(report_dir: str, report_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """扫描 report_dir 下各算子的 phase_state.json."""
    out: Dict[str, Dict[str, Any]] = {}
    rd = Path(report_dir)
    if not rd.exists():
        return out
    for key in report_keys:
        pf = rd / key / "phase_state.json"
        if pf.exists():
            try:
                out[key] = json.loads(pf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def _empty_phase() -> Dict[str, Any]:
    return {
        "pid": None,
        "started_at": None,
        "ended_at": None,
        "duration_sec": 0.0,
        "status": None,
    }


def _ensure_phase(op: Dict[str, Any], phase: str) -> Dict[str, Any]:
    phases = op.setdefault("phases", {})
    if phase not in phases or not isinstance(phases.get(phase), dict):
        phases[phase] = _empty_phase()
    return phases[phase]


def _update_phase_pid(op: Dict[str, Any], phase: str, pid: int, first_seen: str) -> None:
    info = _ensure_phase(op, phase)
    if info.get("pid") != pid:
        info["pid"] = pid
        info["started_at"] = first_seen
        info["ended_at"] = None
        info["duration_sec"] = 0.0
    if _pid_alive(pid):
        info["duration_sec"] = round(_elapsed_since(info.get("started_at") or first_seen), 1)
        info["status"] = "running"
    elif not info.get("ended_at"):
        info["ended_at"] = _now()


# ────────────────────────────────────────────────────────────
# 状态聚合
# ────────────────────────────────────────────────────────────

def refresh_run_state(
    state: Dict[str, Any],
    *,
    root_pid: int,
    report_dir: Path,
    report_keys: List[str],
    timeout_sec: int,
    seen_pids: Dict[Tuple[str, str], Tuple[int, str]],
    exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    """从 report 文件和进程树刷新一次 benchmark run 状态."""
    if exit_code is None:
        for key, pid in _find_opencode_procs(root_pid).items():
            if key not in seen_pids or seen_pids[key][0] != pid:
                seen_pids[key] = (pid, _now())

    reports = _scan_reports(str(report_dir), report_keys)
    phase_states = _scan_phase_states(str(report_dir), report_keys)

    for op in state.get("operators", []):
        name = op["op_name"]
        report_key = op.get("report_key") or name
        if report_key in phase_states:
            ps = phase_states[report_key]
            phase = ps.get("phase", op.get("phase", "pending"))
            phase_status = ps.get("status", op.get("phase_status", ""))
            updated_at = ps.get("updated_at")
            op["phase"] = phase
            op["phase_status"] = phase_status
            op["phase_message"] = ps.get("message", "")
            if updated_at and not op["started_at"]:
                op["started_at"] = updated_at
            if phase in ("pypto", "verifier"):
                phase_info = _ensure_phase(op, phase)
                if updated_at and not phase_info.get("started_at"):
                    phase_info["started_at"] = updated_at
                if phase_status:
                    phase_info["status"] = phase_status
            if phase == "verifier" and updated_at:
                pypto_phase = _ensure_phase(op, "pypto")
                if pypto_phase.get("started_at") and not pypto_phase.get("ended_at"):
                    pypto_phase["ended_at"] = updated_at
            if phase == "done" and updated_at:
                for done_phase in ("verifier", "pypto"):
                    phase_info = _ensure_phase(op, done_phase)
                    if phase_info.get("started_at") and not phase_info.get("ended_at"):
                        phase_info["ended_at"] = updated_at
                        break
            if ps.get("pypto_status"):
                op["pypto_status"] = ps["pypto_status"]
            if ps.get("verifier_status"):
                op["verifier_status"] = ps["verifier_status"]

        active_pid: Optional[int] = None
        for phase in ("pypto", "verifier"):
            key = (name, phase)
            if key in seen_pids:
                pid, first = seen_pids[key]
                _update_phase_pid(op, phase, pid, first)
                if _pid_alive(pid):
                    active_pid = pid
                if not op["started_at"]:
                    op["started_at"] = first
        op["opencode_pid"] = active_pid

        if report_key in reports:
            r = reports[report_key]
            op["result_status"] = r.get("overall_status", "")
            op["pypto_status"] = r.get("pypto_status", "")
            op["verifier_status"] = r.get("verifier_status", "")
            if r.get("finished_at") and not op["ended_at"]:
                op["ended_at"] = r["finished_at"]
            if r.get("started_at") and not op["started_at"]:
                op["started_at"] = r["started_at"]
            dur = (r.get("pypto_duration_sec") or 0) + (r.get("verifier_duration_sec") or 0)
            if dur > op["duration_sec"]:
                op["duration_sec"] = round(dur, 1)
            pypto_phase = _ensure_phase(op, "pypto")
            verify_phase = _ensure_phase(op, "verifier")
            if r.get("pypto_duration_sec") is not None:
                pypto_phase["duration_sec"] = round(r.get("pypto_duration_sec") or 0, 1)
            if r.get("verifier_duration_sec") is not None:
                verify_phase["duration_sec"] = round(r.get("verifier_duration_sec") or 0, 1)
            if r.get("pypto_status"):
                pypto_phase["status"] = r["pypto_status"]
            if r.get("verifier_status"):
                verify_phase["status"] = r["verifier_status"]
            if r.get("finished_at"):
                if r.get("verifier_status") and verify_phase.get("started_at") and not verify_phase.get("ended_at"):
                    verify_phase["ended_at"] = r["finished_at"]
                elif r.get("pypto_status") and pypto_phase.get("started_at") and not pypto_phase.get("ended_at"):
                    pypto_phase["ended_at"] = r["finished_at"]
            if r.get("finished_at") and r.get("overall_status"):
                op["phase"] = "done"
                op["phase_status"] = r["overall_status"]

        phase_total = 0.0
        for phase in ("pypto", "verifier"):
            info = _ensure_phase(op, phase)
            phase_total += float(info.get("duration_sec") or 0.0)
        if phase_total > op["duration_sec"]:
            op["duration_sec"] = round(phase_total, 1)

        op["dev_status"] = _compute_dev_status(op, timeout_sec)

    if exit_code is not None:
        state["main_exit_code"] = exit_code
        state["main_status"] = "已完成" if exit_code == 0 else "未完成"

    state["updated_at"] = _now()
    return state


# ────────────────────────────────────────────────────────────
# 看板渲染
# ────────────────────────────────────────────────────────────

_MAIN_STATUS_LABEL = {
    "正在进行": "正在进行",
    "预检查中": "预检查中",
    "已完成": "已完成",
    "未完成": "未完成",
}

_PHASE_LABEL = {
    "pending": "未开始",
    "prepare": "准备",
    "pypto": "PyPTO",
    "verifier": "Verifier",
    "done": "完成",
}

_COL_W = {
    "idx": 4,
    "op": 24,
    "phase": 10,
    "phase_start": 17,
    "phase_time": 16,
    "dur": 9,
    "status": 22,
}


def _cell_width(value: Any) -> int:
    width = wcswidth(str(value))
    return width if width >= 0 else len(str(value))


def _clip_cell(value: Any, width: int) -> str:
    text = str(value)
    if _cell_width(text) <= width:
        return text
    suffix = "…"
    suffix_width = _cell_width(suffix)
    out: List[str] = []
    used = 0
    for ch in text:
        ch_width = _cell_width(ch)
        if used + ch_width + suffix_width > width:
            break
        out.append(ch)
        used += ch_width
    return "".join(out) + suffix


def _pad_cell(value: Any, width: int, *, align: str = "<") -> str:
    text = _clip_cell(value, width)
    pad = max(0, width - _cell_width(text))
    if align == ">":
        return " " * pad + text
    return text + " " * pad


def _format_monitor_time(value: Any) -> str:
    if not value:
        return "—"
    text = str(value)
    for parser in (
        dt.datetime.fromisoformat,
        lambda item: dt.datetime.strptime(item, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parser(text)
            return parsed.strftime("%m-%d %H:%M:%S")
        except ValueError:
            continue
    return text


def _phase_start(phases: Dict[str, Any], phase: str) -> str:
    info = phases.get(phase) if isinstance(phases, dict) else None
    if not isinstance(info, dict):
        return "—"
    return _format_monitor_time(info.get("started_at"))


def _phase_elapsed_or_end(phases: Dict[str, Any], phase: str) -> str:
    info = phases.get(phase) if isinstance(phases, dict) else None
    if not isinstance(info, dict):
        return "—"
    ended_at = info.get("ended_at")
    if ended_at:
        return _format_monitor_time(ended_at)
    duration = float(info.get("duration_sec") or 0.0)
    started_at = info.get("started_at")
    if info.get("status") == "running" and started_at:
        duration = max(duration, _elapsed_since(str(started_at)))
    if duration > 0:
        return f"{duration:.1f}s"
    return "—"


def _render_dashboard(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    width = 160
    lines.append("")
    lines.append("=" * width)
    lines.append("  算子运行监控看板")
    lines.append("=" * width)
    lines.append("")

    mp = state.get("main_pid") or "—"
    ms = _MAIN_STATUS_LABEL.get(state.get("main_status", ""), state.get("main_status", "—"))
    ec = state.get("main_exit_code")
    sa = state.get("started_at", "—")
    ua = state.get("updated_at", "—")

    lines.append(f"  主进程 PID: {mp}    状态: {ms}    退出码: {ec if ec is not None else '—'}")
    lines.append(f"  启动时间:   {sa}    更新时间: {ua}")
    preflight_msg = state.get("preflight_message")
    if preflight_msg:
        lines.append(f"  预检查:     {preflight_msg}")
    lines.append("")

    ops = state.get("operators", [])
    if not ops:
        lines.append("  (无算子记录)")
    else:
        w = _COL_W
        hdr = (
            f"  {_pad_cell('#', w['idx'], align='>')}"
            f"  {_pad_cell('Operator', w['op'])}"
            f"  {_pad_cell('Phase', w['phase'])}"
            f"  {_pad_cell('PyPTO Start', w['phase_start'])}"
            f"  {_pad_cell('PyPTO Elap/End', w['phase_time'])}"
            f"  {_pad_cell('Verify Start', w['phase_start'])}"
            f"  {_pad_cell('Verify Elap/End', w['phase_time'])}"
            f"  {_pad_cell('Sec', w['dur'], align='>')}"
            f"  {_pad_cell('Status', w['status'])}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (width - 2))

        for i, op in enumerate(ops, 1):
            phases = op.get("phases") or {}
            phase_s = _PHASE_LABEL.get(op.get("phase", ""), op.get("phase", "—"))
            pypto_start = _phase_start(phases, "pypto")
            pypto_elapsed = _phase_elapsed_or_end(phases, "pypto")
            verifier_start = _phase_start(phases, "verifier")
            verifier_elapsed = _phase_elapsed_or_end(phases, "verifier")
            dur_s = f"{op.get('duration_sec', 0):.1f}"
            dev_s = op.get("dev_status", "未知")
            nm = op.get("op_name", "?")
            if op.get("level"):
                nm = f"{op['level']}/{nm}"
            lines.append(
                f"  {_pad_cell(i, w['idx'], align='>')}"
                f"  {_pad_cell(nm, w['op'])}"
                f"  {_pad_cell(phase_s, w['phase'])}"
                f"  {_pad_cell(pypto_start, w['phase_start'])}"
                f"  {_pad_cell(pypto_elapsed, w['phase_time'])}"
                f"  {_pad_cell(verifier_start, w['phase_start'])}"
                f"  {_pad_cell(verifier_elapsed, w['phase_time'])}"
                f"  {_pad_cell(dur_s, w['dur'], align='>')}"
                f"  {_pad_cell(dev_s, w['status'])}"
            )

    lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI: monitor
# ────────────────────────────────────────────────────────────

def cmd_monitor(state_dir: Path) -> int:
    configure_state_dir(state_dir)
    state = _read_state()
    if state is None:
        print(f"无监控状态, 请确认目录包含 state.json: {_state_file()}", file=sys.stderr)
        return 1

    try:
        while True:
            if sys.stdout.isatty():
                os.system("clear")
            state = _read_state()
            if state is None:
                print("监控状态丢失.")
                break
            print(_render_dashboard(state))
            if state.get("main_status") in ("已完成", "未完成"):
                print("  主进程已结束, 看板停止刷新.")
                break
            time.sleep(_POLL_SEC)
    except KeyboardInterrupt:
        pass
    return 0


def main(state_dir: Path = Path(DEFAULT_STATE_DIR)) -> int:
    return cmd_monitor(Path(state_dir))
