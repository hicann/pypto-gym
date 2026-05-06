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
"""Export an OpenCode session JSON transcript to Markdown.

This is a Python replacement for the local NodeJS ``opencode-export-md``
helper.  The benchmark runners use it after each ``opencode run`` finishes so
the live log remains tail-able during execution and the final report keeps a
readable Markdown transcript per case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SESSION_ID_RE = re.compile(r"\bses_[A-Za-z0-9]+\b")
DEFAULT_EXPORT_TIMEOUT_SEC = 120


@dataclass
class OpencodeExportResult:
    """Result of one OpenCode transcript export attempt."""

    session_id: Optional[str] = None
    markdown_file: Optional[Path] = None
    status: str = "skipped"
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "exported" and self.markdown_file is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "markdown_file": str(self.markdown_file) if self.markdown_file else None,
            "status": self.status,
            "message": self.message,
        }


def make_session_title(op_name: str, phase: str) -> str:
    """Return a unique and searchable OpenCode session title."""
    safe_op = _safe_title_part(op_name, limit=64)
    safe_phase = _safe_title_part(phase, limit=32)
    millis = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    return (
        f"pypto-bench:{safe_phase}:{safe_op}:"
        f"{os.getpid()}:{threading.get_ident()}:{millis}:{unique}"
    )


def find_session_ids(text: str) -> List[str]:
    """Find unique session ids while preserving their first-seen order."""
    out: List[str] = []
    seen = set()
    for match in SESSION_ID_RE.finditer(text or ""):
        sid = match.group(0)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def export_session_from_log(
    *,
    log_file: Optional[Path],
    output_file: Path,
    session_title: str = "",
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    allow_latest_fallback: bool = False,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> OpencodeExportResult:
    """Resolve a session for one runner log and export it to Markdown.

    ``session_title`` is preferred because benchmark cases can run in parallel.
    ``allow_latest_fallback`` is intentionally false for benchmark automation.
    """
    session_id = resolve_session_id(
        log_file=log_file,
        session_title=session_title,
        opencode_bin=opencode_bin,
        cwd=cwd,
        allow_latest_fallback=allow_latest_fallback,
        timeout_sec=timeout_sec,
    )
    if not session_id:
        return OpencodeExportResult(
            status="skipped",
            message="未能解析 OpenCode session id, 跳过 Markdown 导出.",
        )

    return export_session_to_markdown(
        session_id=session_id,
        output_file=output_file,
        opencode_bin=opencode_bin,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )


def resolve_session_id(
    *,
    log_file: Optional[Path] = None,
    session_title: str = "",
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    allow_latest_fallback: bool = False,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    """Resolve a session id by unique title, then by log contents."""
    if session_title:
        by_title = resolve_session_id_by_title(
            session_title,
            opencode_bin=opencode_bin,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        if by_title:
            return by_title

    if log_file is not None and log_file.exists():
        try:
            ids = find_session_ids(log_file.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            ids = []
        if ids:
            return ids[0]

    if allow_latest_fallback:
        return latest_session_id(
            opencode_bin=opencode_bin,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
    return None


def resolve_session_id_by_title(
    session_title: str,
    *,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    """Find a session id whose OpenCode title matches ``session_title``."""
    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "session", "list"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        stdout_text = _decode_process_output(completed.stdout)
        for line in stdout_text.splitlines():
            if session_title in line:
                ids = find_session_ids(line)
                if ids:
                    return ids[0]
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    return _resolve_session_id_by_title_from_db(session_title, cwd=cwd)


def latest_session_id(
    *,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    """Return the first session id from ``opencode session list``."""
    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "session", "list"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    ids = find_session_ids(_decode_process_output(completed.stdout))
    return ids[0] if ids else None


def export_session_to_markdown(
    *,
    session_id: str,
    output_file: Path,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    raw_json_file: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> OpencodeExportResult:
    """Render one OpenCode session transcript as Markdown."""
    data = _load_session_export_from_db(session_id)
    if data is not None:
        return _write_session_markdown(
            session_id=session_id,
            data=data,
            output_file=output_file,
            raw_json_file=raw_json_file,
            message="Markdown transcript exported from OpenCode sqlite storage.",
        )

    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "export", session_id],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode 可执行未找到: {exc}",
        )
    except subprocess.TimeoutExpired:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export 超时 (>{timeout_sec}s): {session_id}",
        )
    except OSError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export 启动失败: {exc}",
        )

    if completed.returncode != 0:
        detail = (
            _decode_process_output(completed.stderr)
            or _decode_process_output(completed.stdout)
            or ""
        ).strip()
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export 失败 code={completed.returncode}: {detail[:500]}",
        )

    stdout_text = _decode_process_output(completed.stdout)
    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export JSON 解析失败: {exc}",
        )

    return _write_session_markdown(
        session_id=session_id,
        data=data,
        output_file=output_file,
        raw_json_file=raw_json_file,
        message="Markdown transcript exported.",
    )


def _write_session_markdown(
    *,
    session_id: str,
    data: Dict[str, Any],
    output_file: Path,
    raw_json_file: Optional[Path],
    message: str,
) -> OpencodeExportResult:
    try:
        if raw_json_file is not None:
            raw_json_file.parent.mkdir(parents=True, exist_ok=True)
            raw_json_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        output_file = output_file.resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(render_transcript(data), encoding="utf-8")
    except OSError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"Markdown 写入失败: {exc}",
        )

    return OpencodeExportResult(
        session_id=session_id,
        markdown_file=output_file,
        status="exported",
        message=message,
    )


def render_transcript(data: Dict[str, Any]) -> str:
    """Render OpenCode export JSON as Markdown."""
    info = data.get("info") or {}
    messages = data.get("messages") or []
    title = info.get("title") or info.get("id") or "OpenCode Session"
    session_id = info.get("id") or ""
    time_block = info.get("time") or {}

    lines: List[str] = [
        f"# {title}",
        "",
        f"**Session ID:** {session_id}",
        "",
        f"**Created:** {_format_timestamp(time_block.get('created'))}",
        "",
        f"**Updated:** {_format_timestamp(time_block.get('updated'))}",
        "",
        "---",
        "",
    ]

    for msg in messages:
        msg_info = msg.get("info") or {}
        parts = msg.get("parts") or []
        lines.append(_format_message(msg_info, parts).rstrip())
        lines.extend(["---", ""])

    return "\n".join(lines).rstrip() + "\n"


def append_export_result_to_log(
    log_file: Optional[Path],
    result: OpencodeExportResult,
    *,
    label: str,
) -> None:
    """Append the export status to a live runner log without affecting results."""
    if log_file is None:
        return
    _write_export_result_sidecar(log_file, result, label=label)
    try:
        with log_file.open("a", encoding="utf-8") as handle:
            if result.ok:
                handle.write(
                    f"\n[opencode export:{label}] session_id={result.session_id} "
                    f"markdown={result.markdown_file}\n"
                )
            else:
                handle.write(
                    f"\n[opencode export:{label}] status={result.status} "
                    f"session_id={result.session_id or '<unknown>'} "
                    f"message={result.message}\n"
                )
    except OSError:
        pass


def _write_export_result_sidecar(
    log_file: Path,
    result: OpencodeExportResult,
    *,
    label: str,
) -> None:
    """Write visible export diagnostics next to pypto/verifier logs."""
    try:
        payload = result.to_dict()
        payload["label"] = label
        payload["log_file"] = str(log_file)
        status_json = log_file.parent / f"{label}_session_export.json"
        status_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        status_log = log_file.parent / f"{label}_session_export.log"
        status_log.write_text(
            "\n".join(
                [
                    f"label={label}",
                    f"status={result.status}",
                    f"session_id={result.session_id or ''}",
                    f"markdown_file={result.markdown_file or ''}",
                    f"message={result.message}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _format_message(info: Dict[str, Any], parts: List[Dict[str, Any]]) -> str:
    role = info.get("role")
    chunks: List[str] = []
    if role == "user":
        chunks.append("## User\n")
    else:
        chunks.append(_assistant_header(info))

    for part in parts:
        rendered = _format_part(part)
        if rendered:
            chunks.append(rendered)
    return "\n".join(chunk.rstrip() for chunk in chunks if chunk is not None) + "\n"


def _assistant_header(info: Dict[str, Any]) -> str:
    pieces: List[str] = []
    agent = info.get("agent")
    model = info.get("modelID")
    duration = _message_duration(info.get("time") or {})
    if agent:
        pieces.append(_titlecase(str(agent)))
    if model:
        pieces.append(str(model))
    if duration:
        pieces.append(duration)
    if pieces:
        return f"## Assistant ({' - '.join(pieces)})\n"
    return "## Assistant\n"


def _format_part(part: Dict[str, Any]) -> str:
    part_type = part.get("type")
    if part_type == "text" and not part.get("synthetic"):
        return str(part.get("text") or "") + "\n"

    if part_type == "reasoning":
        text = str(part.get("text") or "").strip()
        return f"_Thinking:_\n\n{text}\n" if text else ""

    if part_type == "tool":
        tool_name = part.get("tool") or "(unknown)"
        state = part.get("state") or {}
        chunks = [f"**Tool: {tool_name}**\n"]
        if "input" in state and state.get("input") is not None:
            input_text = json.dumps(state.get("input"), indent=2, ensure_ascii=False)
            chunks.append("**Input:**\n")
            chunks.append(_fenced(input_text, lang="json"))
        if state.get("status") == "completed" and state.get("output") is not None:
            chunks.append("**Output:**\n")
            chunks.append(_fenced(_stringify_block(state.get("output"))))
        if state.get("status") == "error" and state.get("error") is not None:
            chunks.append("**Error:**\n")
            chunks.append(_fenced(_stringify_block(state.get("error"))))
        return "\n".join(chunk.rstrip() for chunk in chunks) + "\n"

    return ""


def _fenced(text: str, lang: str = "") -> str:
    longest = 0
    for match in re.finditer(r"`+", text or ""):
        longest = max(longest, len(match.group(0)))
    fence = "`" * max(3, longest + 1)
    suffix = lang if lang else ""
    return f"{fence}{suffix}\n{text}\n{fence}\n"


def _decode_process_output(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return bytes(data).decode("utf-8", errors="replace")


def _stringify_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _message_duration(time_block: Dict[str, Any]) -> str:
    created = time_block.get("created")
    completed = time_block.get("completed")
    if not isinstance(created, (int, float)) or not isinstance(completed, (int, float)):
        return ""
    return f"{(completed - created) / 1000:.1f}s"


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) > 100000000000 else value
        try:
            return dt.datetime.fromtimestamp(seconds).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
        except (OverflowError, OSError, ValueError):
            return str(value)
    return str(value)


def _titlecase(value: str) -> str:
    out: List[str] = []
    for part in re.split(r"([\s_-]+)", value):
        if not part or re.fullmatch(r"[\s_-]+", part):
            out.append(part)
        else:
            out.append(part[0].upper() + part[1:])
    return "".join(out)


def _safe_title_part(value: str, *, limit: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value or "unknown").strip("_")
    return (cleaned or "unknown")[:limit]


def _resolve_opencode(opencode_bin: str = "") -> str:
    if opencode_bin:
        candidate = Path(opencode_bin).expanduser()
        if candidate.exists():
            return str(candidate)
        found = shutil.which(opencode_bin)
        if found:
            return found
        raise FileNotFoundError(opencode_bin)
    found = shutil.which("opencode")
    if not found:
        raise FileNotFoundError("opencode")
    return found


def _load_session_export_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Build an OpenCode export transcript from sqlite storage."""
    for db_path in _opencode_db_paths():
        data = _load_session_export_from_db_path(db_path, session_id)
        if data is not None:
            return data
    return None


def _load_session_export_from_db_path(
    db_path: Path,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
            session = conn.execute(
                "select id, title, directory, version, time_created, time_updated "
                "from session where id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None

            message_rows = conn.execute(
                "select id, data from message where session_id = ? "
                "order by time_created asc, id asc",
                (session_id,),
            ).fetchall()
            part_rows = conn.execute(
                "select message_id, data from part where session_id = ? "
                "order by time_created asc, id asc",
                (session_id,),
            ).fetchall()
    except sqlite3.Error:
        return None

    parts_by_message: Dict[str, List[Dict[str, Any]]] = {}
    for message_id, part_text in part_rows:
        part_data = _loads_json_object(part_text)
        if part_data is not None:
            parts_by_message.setdefault(str(message_id), []).append(part_data)

    messages: List[Dict[str, Any]] = []
    for message_id, message_text in message_rows:
        message_info = _loads_json_object(message_text)
        if message_info is None:
            continue
        message_info.setdefault("id", str(message_id))
        messages.append(
            {
                "info": message_info,
                "parts": parts_by_message.get(str(message_id), []),
            }
        )

    sid, title, directory, version, created, updated = session
    return {
        "info": {
            "id": sid,
            "title": title,
            "directory": directory,
            "version": version,
            "time": {
                "created": created,
                "updated": updated,
            },
        },
        "messages": messages,
    }


def _loads_json_object(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _resolve_session_id_by_title_from_db(
    session_title: str,
    *,
    cwd: Optional[Path],
) -> Optional[str]:
    for db_path in _opencode_db_paths():
        session_id = _resolve_session_id_by_title_from_db_path(
            db_path,
            session_title,
            cwd=cwd,
        )
        if session_id:
            return session_id
    return None


def _resolve_session_id_by_title_from_db_path(
    db_path: Path,
    session_title: str,
    *,
    cwd: Optional[Path],
) -> Optional[str]:
    if not db_path.exists():
        return None
    cwd_resolved = str(cwd.resolve()) if cwd else ""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
            rows = conn.execute(
                "select id, directory from session where title = ? "
                "order by time_created desc limit 20",
                (session_title,),
            ).fetchall()
    except sqlite3.Error:
        return None

    if not rows:
        return None
    if cwd_resolved:
        for session_id, directory in rows:
            if str(Path(directory).resolve()) == cwd_resolved:
                return str(session_id)
    return str(rows[0][0])


def _opencode_db_paths() -> List[Path]:
    explicit = os.environ.get("OPENCODE_DB")
    if explicit:
        return [Path(explicit).expanduser()]

    candidates: List[Path] = []
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        candidates.append(Path(data_home).expanduser() / "opencode" / "opencode.db")
    candidates.append(Path.home() / ".local" / "share" / "opencode" / "opencode.db")

    out: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _opencode_db_path() -> Optional[Path]:
    paths = _opencode_db_paths()
    return paths[0] if paths else None


def _parse_compatible_args(
    positional: Sequence[str],
) -> tuple[Optional[str], Optional[Path]]:
    if len(positional) > 2:
        raise ValueError("too many arguments")
    if not positional:
        return None, None
    first = positional[0]
    if first.endswith(".md"):
        return None, Path(first)
    session_id = first
    output_file = Path(positional[1]) if len(positional) == 2 else None
    return session_id, output_file


def _main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export an OpenCode session transcript to Markdown.",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Compatible forms: [sessionID] [output.md], [output.md], or no args.",
    )
    parser.add_argument("--from-log", type=Path, default=None,
                        help="Resolve session id from a runner log file.")
    parser.add_argument("--title", default="",
                        help="Resolve session id by exact OpenCode session title.")
    parser.add_argument("--opencode-bin", default="", help="OpenCode executable.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(),
                        help="Working directory for opencode commands.")
    parser.add_argument("--allow-latest-fallback", action="store_true",
                        help="Allow latest-session fallback when --from-log/--title miss.")
    parser.add_argument("--raw-json-out", type=Path, default=None,
                        help="Optional path to also write formatted export JSON.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_EXPORT_TIMEOUT_SEC)
    ns = parser.parse_args(argv)

    try:
        session_id, output_file = _parse_compatible_args(ns.args)
    except ValueError as exc:
        parser.error(str(exc))

    if ns.from_log is not None:
        output_file = output_file or Path("session-from-log.md")
        result = export_session_from_log(
            log_file=ns.from_log,
            output_file=output_file,
            session_title=ns.title,
            opencode_bin=ns.opencode_bin,
            cwd=ns.cwd,
            allow_latest_fallback=ns.allow_latest_fallback,
            timeout_sec=ns.timeout_sec,
        )
    else:
        session_id = session_id or resolve_session_id(
            session_title=ns.title,
            opencode_bin=ns.opencode_bin,
            cwd=ns.cwd,
            allow_latest_fallback=True,
            timeout_sec=ns.timeout_sec,
        )
        if not session_id:
            print("No OpenCode sessions found.", file=sys.stderr)
            return 1
        output_file = output_file or Path(f"session-{session_id[:8]}.md")
        result = export_session_to_markdown(
            session_id=session_id,
            output_file=output_file,
            opencode_bin=ns.opencode_bin,
            cwd=ns.cwd,
            raw_json_file=ns.raw_json_out,
            timeout_sec=ns.timeout_sec,
        )

    if result.ok:
        print(result.markdown_file)
        return 0
    print(result.message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(_main_cli())
