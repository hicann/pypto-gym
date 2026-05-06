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
"""Unified public CLI for benchmark."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from benchmark import monitor, run_kernelbench


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=run_kernelbench.DEFAULT_CONFIG_PATH,
        help="YAML config path, for example: configs/relu.yaml",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="KernelBench x pypto unified benchmark CLI",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run", help="run benchmark from YAML config")
    _add_config_arg(run_parser)
    run_parser.add_argument(
        "--foreground",
        action="store_true",
        help="前台运行评测（原行为）；默认后台 detached 并自动打开 monitor TUI",
    )

    monitor_parser = subcommands.add_parser("monitor", help="view benchmark monitor state directory")
    monitor_parser.add_argument(
        "state_dir",
        type=Path,
        help="Directory containing state.json written by `python -m benchmark run`",
    )
    return parser


def _wait_for_state_json(state_json: Path, *, timeout_sec: float = 10.0,
                         poll_sec: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if state_json.exists():
            return True
        time.sleep(poll_sec)
    return False


def _read_log_tail(path: Path, *, max_bytes: int = 8000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")
    return data[-max_bytes:].decode("utf-8", errors="replace")


def _describe_fork_child_status(pid: int) -> str:
    try:
        harvested, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return "子进程: waitpid 不可用 (可能已被回收)."
    if harvested == 0:
        return "子进程: 仍在运行或尚未可收集状态."
    if os.WIFEXITED(status):
        return f"子进程已退出, exit_code={os.WEXITSTATUS(status)}."
    if os.WIFSIGNALED(status):
        return f"子进程被信号终止: {os.WTERMSIG(status)}."
    return f"子进程 wait 状态: {status!r}."


def _child_run_detached_benchmark(config_path: Path, log_dir: Path) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    out_fd = os.open(
        str(log_dir / "benchmark.out"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    err_fd = os.open(
        str(log_dir / "benchmark.err"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(devnull_fd, 0)
        os.dup2(out_fd, 1)
        os.dup2(err_fd, 2)
    finally:
        for fd in (devnull_fd, out_fd, err_fd):
            if fd > 2:
                try:
                    os.close(fd)
                except OSError:
                    pass

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)

    os.environ[run_kernelbench._BENCHMARK_BACKGROUND_CHILD_ENV] = "1"

    exit_code = 1
    try:
        exit_code = run_kernelbench.run_from_config(config_path)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            exit_code = code
        elif code is None:
            exit_code = 0
        else:
            exit_code = 1
            # 捕获 SystemExit 会抑制解释器默认打印; 已 dup 到 benchmark.err, 须显式写出原因.
            msg = str(code).strip() if isinstance(code, str) else ""
            if not msg and exc.args:
                msg = str(exc.args[0]).strip()
            if msg:
                print(msg, file=sys.stderr, flush=True)
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    os._exit(exit_code)


def _handle_run(config_path: Path, *, foreground: bool) -> int:
    if foreground:
        return run_kernelbench.run_from_config(config_path)

    if not hasattr(os, "fork"):
        print("错误: 当前平台不支持 os.fork，请使用 --foreground。", file=sys.stderr)
        return 2

    state_dir = run_kernelbench.pre_resolve_state_dir(config_path)
    log_dir = state_dir.parent / "logs"

    try:
        run_kernelbench.preflight_background_run(config_path)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        if code is not None:
            print(code, file=sys.stderr)
        return 1

    pid = os.fork()
    if pid < 0:
        print("错误: fork 失败。", file=sys.stderr)
        return 2

    if pid == 0:
        _child_run_detached_benchmark(config_path, log_dir)

    if hasattr(signal, "SIGCHLD"):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    state_json = state_dir / "state.json"
    if not _wait_for_state_json(state_json):
        child_note = _describe_fork_child_status(pid)
        err_tail = _read_log_tail(log_dir / "benchmark.err")
        out_tail = _read_log_tail(log_dir / "benchmark.out")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        print(f"错误: 等待 state.json 超时（超过 10s）: {state_json}", file=sys.stderr)
        print(child_note, file=sys.stderr)
        if err_tail.strip():
            print(f"--- logs/benchmark.err (tail) ---\n{err_tail}", file=sys.stderr)
        else:
            print("--- logs/benchmark.err: (空) ---", file=sys.stderr)
        if out_tail.strip():
            print(f"--- logs/benchmark.out (tail) ---\n{out_tail}", file=sys.stderr)
        else:
            print("--- logs/benchmark.out: (空) ---", file=sys.stderr)
        print(
            f"提示: 若子进程在意想不到处失败, 可用前台重放查看完整输出:\n"
            f"  {sys.executable} -m benchmark run --config {config_path} --foreground",
            file=sys.stderr,
        )
        print(f"日志目录: {log_dir}", file=sys.stderr)
        return 1

    rc = monitor.main(state_dir)
    reconnect_cmd = f"{sys.executable} -m benchmark monitor {state_dir}"
    print(f"如需重连 monitor: {reconnect_cmd}", flush=True)
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _handle_run(args.config, foreground=args.foreground)
    if args.command == "monitor":
        return monitor.main(args.state_dir)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
