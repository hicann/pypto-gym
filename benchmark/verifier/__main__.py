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
"""统一 verifier CLI — 给 ``.opencode/skills/pypto-kernel-validate`` 直接调用.

子命令:
    cheat-check    仅跑机械层反作弊检测 (无需 NPU 设备).
    verify         机械检测 + 精度 (+ 可选性能), 输出统一 JSON 报告.

设计意图:
    skill 引导大模型按以下流程做验证:
        1. ``cheat-check`` 拿机械层报告
        2. **大模型亲自语义审阅** (skill 中详述; 本 CLI 不参与)
        3. ``verify`` 跑精度/性能 (CHEAT 已确诊则跳过)
        4. 大模型整合三方结论, 给出 final_verdict

    本 CLI 故意不做 "自动调 LLM 复审", 因为语义审阅是 skill 的核心价值,
    必须由调用方的 agent 在 skill 上下文里亲自完成.

JSON 报告 schema (verify 子命令):
    {
      "op_name": "...",
      "op_dir": "...",
      "task_id": "...",
      "mode": "correctness | performance | full",
      "cheat_check": { ... },          # cheat_detector.detect_cheats 输出
      "correctness": {
        "status": "passed | failed | error | skipped",
        "duration_sec": <float>,
        "log_excerpt": "...",
        "log_file": "..."              # 完整日志路径 (调用方按需读)
      },
      "performance": {
        "status": "passed | failed | error | skipped",
        "gen_time_us": <float|null>,
        "base_time_us": <float|null>,
        "speedup": <float|null>,
        "log_excerpt": "..."
      },
      "verdict_machine": "pass | failed_cheat | failed_correctness | failed_performance | error"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.verifier import cheat_detector
from benchmark.verifier.config import load_config
from benchmark.verifier.kernel_verifier import KernelVerifier
from benchmark.verifier.manager import get_worker_manager, register_local_worker


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 子命令: cheat-check
# ────────────────────────────────────────────────────────────

def _cmd_cheat_check(args: argparse.Namespace) -> int:
    op_dir: Path = args.op_dir.resolve()
    op_name = args.op_name or op_dir.name
    if not op_dir.is_dir():
        logger.error("[verifier] op_dir 不存在或非目录: %s", op_dir)
        return 2

    report = cheat_detector.detect_cheats(op_dir, op_name)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        logger.info("[verifier] cheat-check 报告已写入: %s", args.json_out)
    sys.stdout.write(payload + "\n")
    return {"pass": 0, "suspicious": 0, "cheat": 1}.get(report.verdict, 0)


# ────────────────────────────────────────────────────────────
# 子命令: verify
# ────────────────────────────────────────────────────────────

_MODE_CHOICES = ("correctness", "performance", "full")


def _excerpt(text: str, max_chars: int = 4000) -> str:
    """裁剪日志, 留头尾各一半 — 避免 JSON 报告爆掉, 但保留首尾上下文."""
    if not text or len(text) <= max_chars:
        return text or ""
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-tail:]


def _collect_pypto_source_files(op_dir: Path, op_name: str) -> Dict[str, str]:
    """读取 verifier 需要的 PyPTO 源文件, 后续在 verify_dir 原样落盘.

    与 ``verifier_runner.collect_pypto_source_files`` 行为一致, 在此重复实现
    是为了让 verifier 子包具备完整的对外 CLI, 不反向依赖桥接层模块.
    """
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


async def _run_verify_async(args: argparse.Namespace) -> Dict[str, Any]:
    op_dir: Path = args.op_dir.resolve()
    op_name = args.op_name or op_dir.name
    task_desc_file: Path = args.task_desc.resolve()
    mode: str = args.mode

    output: Dict[str, Any] = {
        "op_name": op_name,
        "op_dir": str(op_dir),
        "task_id": args.task_id,
        "mode": mode,
        "cheat_check": None,
        "cheat_gate_warning": "",
        "correctness": {"status": "skipped"},
        "performance": {"status": "skipped"},
        "verdict_machine": "error",
    }

    if not op_dir.is_dir():
        output["error"] = f"op_dir 不存在或非目录: {op_dir}"
        return output
    if not task_desc_file.exists():
        output["error"] = f"task_desc 不存在: {task_desc_file}"
        return output

    cheat_report = cheat_detector.detect_cheats(op_dir, op_name)
    output["cheat_check"] = cheat_report.to_dict()
    if cheat_report.verdict == "cheat" and not args.no_cheat_gate:
        output["cheat_gate_warning"] = (
            "cheat_detector 给出 cheat verdict, 但机械层规则存在误报可能; "
            "当前按 warning 继续跑 correctness/performance, 由后续验证与语义层综合裁定."
        )

    try:
        source_files = _collect_pypto_source_files(op_dir, op_name)
    except FileNotFoundError as e:
        output["error"] = str(e)
        output["verdict_machine"] = "error"
        return output

    config = load_config("pypto", backend=args.backend)
    if args.log_dir:
        config["log_dir"] = str(args.log_dir.resolve())
    if args.keep_artifacts is not None:
        config["keep_artifacts"] = bool(args.keep_artifacts)
    config["verify_timeout"] = args.verify_timeout
    if args.verify_rtol is not None:
        config["verify_rtol"] = float(args.verify_rtol)
    if args.verify_atol is not None:
        config["verify_atol"] = float(args.verify_atol)

    manager = get_worker_manager()
    has = await manager.has_worker(backend=args.backend, arch=args.arch)
    if not has:
        await register_local_worker(
            [args.device_id], backend=args.backend, arch=args.arch,
        )
    worker = await manager.select(backend=args.backend, arch=args.arch)
    if worker is None:
        output["error"] = (
            f"WorkerManager 没有匹配 backend={args.backend}/arch={args.arch} 的 worker."
        )
        output["verdict_machine"] = "error"
        return output

    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc_file.read_text(encoding="utf-8"),
            task_id=args.task_id,
            framework=args.framework,
            dsl="pypto",
            backend=args.backend,
            arch=args.arch,
            config=config,
            worker=worker,
        )
        task_info = {"source_files": source_files}

        success, log_text = await verifier.run(
            task_info, current_step=0, device_id=args.device_id,
        )
        output["correctness"] = {
            "status": "passed" if success else "failed",
            "log_excerpt": _excerpt(log_text),
        }

        if mode in ("performance", "full"):
            if not success:
                output["performance"] = {
                    "status": "skipped",
                    "log_excerpt": "skipped: correctness failed.",
                }
            else:
                profile_settings: Dict[str, Any] = {}
                if args.profile_warmup is not None:
                    profile_settings["warmup_times"] = args.profile_warmup
                if args.profile_run is not None:
                    profile_settings["run_times"] = args.profile_run
                try:
                    perf = await verifier.run_profile(
                        task_info,
                        current_step=0,
                        device_id=args.device_id,
                        profile_settings=profile_settings,
                    )
                    perf_log = perf.get("log", "") if perf else ""
                    has_cheat_marker = "CHEAT_MULTI_KERNEL" in perf_log
                    perf_status = "passed" if (
                        perf and perf.get("gen_time") is not None and not has_cheat_marker
                    ) else "failed"
                    output["performance"] = {
                        "status": perf_status,
                        "gen_time_us": perf.get("gen_time") if perf else None,
                        "base_time_us": perf.get("base_time") if perf else None,
                        "speedup": perf.get("speedup") if perf else None,
                        "cheat_multi_kernel": has_cheat_marker,
                        "log_excerpt": _excerpt(perf_log),
                    }
                    if has_cheat_marker:
                        # 运行时 CHEAT 信号 ≥ 静态判定的 cheat verdict, 提升报告基调.
                        cheat_dict = output["cheat_check"]
                        if cheat_dict.get("verdict") == "pass":
                            cheat_dict["verdict"] = "cheat"
                        cheat_dict["checks"].append({
                            "name": "runtime_multi_kernel",
                            "status": "fail",
                            "level": "fatal",
                            "detail": (
                                "profile 运行时 stdout 出现 CHEAT_MULTI_KERNEL 标记 — "
                                "算子在 NPU 上被拆成多个 jit kernel."
                            ),
                            "extra": {},
                        })
                        cheat_dict["summary"] = (
                            "CHEAT (runtime): " + cheat_dict.get("summary", "")
                        )
                except Exception as pe:
                    output["performance"] = {
                        "status": "error",
                        "log_excerpt": f"run_profile 异常: {pe}",
                    }
    finally:
        try:
            await manager.release(worker)
        except Exception:
            logger.warning("release worker failed", exc_info=True)

    perf_block = output.get("performance") or {}
    if perf_block.get("cheat_multi_kernel"):
        output["verdict_machine"] = "failed_cheat"
    elif output["correctness"].get("status") != "passed":
        output["verdict_machine"] = "failed_correctness"
    elif mode in ("performance", "full") and output["performance"].get("status") not in (
        "passed", "skipped"
    ):
        output["verdict_machine"] = "failed_performance"
    else:
        output["verdict_machine"] = "pass"

    return output


def _cmd_verify(args: argparse.Namespace) -> int:
    if args.mode not in _MODE_CHOICES:
        logger.error(
            "[verifier] mode 必须是 %s, 实际: %r",
            _MODE_CHOICES,
            args.mode,
        )
        return 2
    result = asyncio.run(_run_verify_async(args))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        logger.info("[verifier] verify 报告已写入: %s", args.json_out)
    sys.stdout.write(payload + "\n")
    return 0 if result.get("verdict_machine") == "pass" else 1


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.verifier",
        description=__doc__.split("\n", 1)[0] if __doc__ else "",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    cc = sub.add_parser("cheat-check", help="仅跑机械层反作弊检测.")
    cc.add_argument("op_dir", type=Path)
    cc.add_argument("--op-name", default=None)
    cc.add_argument("--json-out", type=Path, default=None)
    cc.set_defaults(func=_cmd_cheat_check)

    v = sub.add_parser("verify", help="cheat-check + 精度 + (可选)性能.")
    v.add_argument("op_dir", type=Path)
    v.add_argument("--op-name", default=None)
    v.add_argument("--task-desc", type=Path, required=True,
                   help="KernelBench task_desc.py 路径 (case_loader 写出).")
    v.add_argument("--mode", choices=_MODE_CHOICES, default="correctness")
    v.add_argument("--arch", default="ascend910b4")
    v.add_argument("--backend", default="ascend")
    v.add_argument("--framework", default="torch")
    v.add_argument("--device-id", type=int, default=0)
    v.add_argument("--task-id", default="0")
    v.add_argument("--verify-timeout", type=int, default=900)
    v.add_argument("--verify-rtol", type=float, default=None,
                   help="verify 精度比较 rtol; 未传则用配置/默认值")
    v.add_argument("--verify-atol", type=float, default=None,
                   help="verify 精度比较 atol; 未传则用配置/默认值")
    v.add_argument("--profile-warmup", type=int, default=None)
    v.add_argument("--profile-run", type=int, default=None)
    v.add_argument("--log-dir", type=Path, default=None,
                   help="KernelVerifier 工作目录根; 缺省 ~/pypto_bench_logs/Task_<rand>.")
    v.add_argument("--keep-artifacts",
                   action=argparse.BooleanOptionalAction,
                   default=None,
                   help="保留 verify/profile 临时工作目录; 默认运行后自动清理.")
    v.add_argument("--json-out", type=Path, default=None,
                   help="统一 JSON 报告输出路径; 缺省仅打印到 stdout.")
    v.add_argument("--no-cheat-gate", action="store_true",
                   help="即便 cheat-check 判 cheat 也继续跑精度/性能 (调试用).")
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
