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
"""桥接层 ``KernelVerifier`` — pypto + ascend + torch 自包含验证.

调用范式:

    KernelVerifier(
        op_name, framework_code,
        task_id="0",
        framework="torch",
        dsl="pypto",
        backend="ascend",
        arch="ascend910b4",
        config=...,
        worker=...,
        bench_type="kernelbench",
    )
    success, log = await verifier.run(task_info, current_step=0, device_id=...)
    perf_dict   = await verifier.run_profile(task_info, current_step=0,
                                             device_id=..., profile_settings={...})

入参约束 (不符直接抛 ValueError):
    - dsl 只接受 ``"pypto"``
    - backend 只接受 ``"ascend"``
    - framework 只接受 ``"torch"``
    - bench_type 只接受 ``"kernelbench"``
    - worker 必须是本子包的 ``LocalWorker``

run_profile 返回 dict 字段:
    - gen_time          (us, float; 失败为 inf)
    - base_time         (us, float; 失败为 inf)
    - speedup           (float; base/gen, 任一无效则 0.0)
    - roofline_time     (None — 桥接层不算)
    - roofline_speedup  (0.0 — 桥接层不算)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from . import script_builder
from .local_worker import LocalWorker

logger = logging.getLogger(__name__)


_BASE_TIME_RE = re.compile(r"^BASE_TIME_US:\s*([0-9.eE+\-]+)\s*$", re.MULTILINE)
_GEN_TIME_RE = re.compile(r"^PROFILE_RESULT_GEN_US:\s*([0-9.eE+\-]+)\s*$", re.MULTILINE)
_VERIFY_PASS_RE = re.compile(r"^VERIFY_PASSED\s*$", re.MULTILINE)
_VERIFY_FAIL_RE = re.compile(r"^VERIFY_FAILED:?", re.MULTILINE)


class KernelVerifier:
    def __init__(
        self,
        op_name: str,
        framework_code: str,
        task_id: str = "0",
        framework: str = "torch",
        dsl: str = "pypto",
        backend: str = "ascend",
        arch: str = "ascend910b4",
        impl_func_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        worker: Optional[LocalWorker] = None,
        bench_type: str = "kernelbench",
    ) -> None:
        if dsl != "pypto":
            raise ValueError(
                f"pypto KernelVerifier 仅支持 dsl='pypto', 实际: {dsl!r}"
            )
        if backend != "ascend":
            raise ValueError(
                f"pypto KernelVerifier 仅支持 backend='ascend', 实际: {backend!r}"
            )
        if framework != "torch":
            raise ValueError(
                f"pypto KernelVerifier 仅支持 framework='torch', 实际: {framework!r}"
            )
        if bench_type != "kernelbench":
            raise ValueError(
                f"pypto KernelVerifier 仅支持 bench_type='kernelbench', 实际: {bench_type!r}"
            )
        if not config:
            raise ValueError("config is required for KernelVerifier")
        if worker is not None and not isinstance(worker, LocalWorker):
            raise ValueError(
                f"pypto KernelVerifier 只接受本子包的 LocalWorker, 实际: {type(worker).__name__}"
            )

        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        self.dsl = dsl
        self.backend = backend
        self.arch = arch.lower()
        self.task_id = task_id
        self.bench_type = bench_type
        self.impl_func_name = impl_func_name or "ModelNew"
        self.config: Dict[str, Any] = config
        self.log_dir: str = config.get("log_dir") or os.path.expanduser("~/pypto_bench_logs")
        self.worker: Optional[LocalWorker] = worker

    # ---------- 静态工具 ----------

    @staticmethod
    def _parse_time(log: str, pattern: re.Pattern[str]) -> float:
        m = pattern.search(log)
        if not m:
            return float("inf")
        try:
            v = float(m.group(1))
        except ValueError:
            return float("inf")
        if v <= 0 or v != v:  # NaN / 非正
            return float("inf")
        return v

    @staticmethod
    def _empty_profile_result(error: str = "") -> Dict[str, Any]:
        return {
            "gen_time": None,
            "base_time": None,
            "speedup": 0.0,
            "roofline_time": None,
            "roofline_speedup": 0.0,
            "roofline": None,
            "log": error,
        }

    # ---------- 公共 API: run / run_profile ----------

    async def run(
        self,
        task_info: Dict[str, Any],
        current_step: int = 0,
        device_id: int = -1,
    ) -> tuple[bool, str]:
        if not self._has_impl_source(task_info):
            return False, "task_info 缺少 coder_code/source_files, 无法验证"

        verify_dir = self._verify_dir(current_step)
        actual_device_id: Optional[int] = None
        try:
            actual_device_id = await self._acquire_device(device_id)
            self._write_source_artifacts(task_info, verify_dir)

            verify_script_name = f"verify_{self.op_name}.py"
            script_text = script_builder.build_verify_script(
                op_name=self.op_name,
                framework_filename=self._framework_filename(),
                device_id=actual_device_id,
                pypto_run_mode=int(self.config.get("pypto_run_mode", 0)),
                rtol=float(self.config.get("verify_rtol", 1e-3)),
                atol=float(self.config.get("verify_atol", 1e-3)),
            )
            (verify_dir / verify_script_name).write_text(script_text, encoding="utf-8")

            verify_timeout = int(self.config.get("verify_timeout", 900))
            # +30s buffer 防止 PyPTO autotune 卡在 sync IO 上误超时.
            success, log = await self._run_one_script(
                verify_script_name, str(verify_dir), verify_timeout + 30
            )
            if success:
                # 子进程 returncode=0 不一定代表语义通过, 必须看 stdout 标记.
                if _VERIFY_PASS_RE.search(log) and not _VERIFY_FAIL_RE.search(log):
                    return True, log
                # 没看到 VERIFY_PASSED 等同失败 (脚本可能因 numpy/torch 版本兼容问题 silently exit).
                return False, log + "\n[verifier] VERIFY_PASSED marker missing in stdout"
            return False, log
        finally:
            await self._release_device(actual_device_id)
            self._cleanup_verify_dir(verify_dir)

    async def run_profile(
        self,
        task_info: Dict[str, Any],
        current_step: int = 0,
        device_id: int = -1,
        profile_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile_settings = profile_settings or {}
        warmup_times = int(profile_settings.get(
            "warmup_times",
            self.config.get("profile_settings", {}).get("warmup_times", 5),
        ))
        run_times = int(profile_settings.get(
            "run_times",
            self.config.get("profile_settings", {}).get("run_times", 50),
        ))

        if not self._has_impl_source(task_info):
            return self._empty_profile_result(error="task_info 缺少 coder_code/source_files")

        verify_dir = self._verify_dir(current_step)
        actual_device_id: Optional[int] = None
        try:
            actual_device_id = await self._acquire_device(device_id)
            # profile 路径独立写一遍源文件 (覆盖即可), 调用方多次重跑也保证一致.
            self._write_source_artifacts(task_info, verify_dir)

            base_script_name = f"profile_{self.op_name}_base.py"
            gen_script_name = f"profile_{self.op_name}_generation.py"

            (verify_dir / base_script_name).write_text(
                script_builder.build_profile_base_script(
                    op_name=self.op_name,
                    framework_filename=self._framework_filename(),
                    device_id=actual_device_id,
                    warmup_times=warmup_times,
                    run_times=run_times,
                ),
                encoding="utf-8",
            )
            (verify_dir / gen_script_name).write_text(
                script_builder.build_profile_generation_script(
                    op_name=self.op_name,
                    framework_filename=self._framework_filename(),
                    device_id=actual_device_id,
                    pypto_run_mode=int(self.config.get("pypto_run_mode", 0)),
                ),
                encoding="utf-8",
            )

            base_timeout = int(self.config.get("verify_timeout", 900)) + 60
            gen_timeout = int(self.config.get("verify_timeout", 900)) + 600  # PyPTO autotune 慢

            base_ok, base_log = await self._run_one_script(
                base_script_name, str(verify_dir), base_timeout
            )
            base_time = self._parse_time(base_log, _BASE_TIME_RE) if base_ok else float("inf")
            if not base_ok:
                logger.warning(f"[{self.op_name}] base profile script failed; base_time=inf")

            gen_ok, gen_log = await self._run_one_script(
                gen_script_name, str(verify_dir), gen_timeout
            )
            gen_time = self._parse_time(gen_log, _GEN_TIME_RE) if gen_ok else float("inf")
            if not gen_ok:
                logger.warning(f"[{self.op_name}] generation profile script failed; gen_time=inf")

            speedup = (base_time / gen_time) if (
                base_time < float("inf") and gen_time > 0 and gen_time < float("inf")
            ) else 0.0

            logger.info(
                f"[{self.op_name}] profile done: base={base_time:.2f}us, "
                f"gen={gen_time:.2f}us, speedup={speedup:.4f}x"
            )
            return {
                "gen_time": None if gen_time == float("inf") else gen_time,
                "base_time": None if base_time == float("inf") else base_time,
                "speedup": speedup,
                "roofline_time": None,
                "roofline_speedup": 0.0,
                "roofline": None,
                "log": base_log + "\n---\n" + gen_log,
            }
        finally:
            await self._release_device(actual_device_id)
            self._cleanup_verify_dir(verify_dir)

    # ---------- 工件落盘 ----------

    def _verify_dir(self, current_step: int) -> Path:
        expanded = os.path.expanduser(self.log_dir)
        unique = f"Iteration{self.task_id}_Step{current_step:02d}_verify"
        d = Path(expanded) / self.op_name / unique
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _keep_artifacts(self) -> bool:
        value = self.config.get("keep_artifacts", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _cleanup_verify_dir(self, verify_dir: Path) -> None:
        if self._keep_artifacts():
            logger.info("[%s] keeping verifier artifacts: %s", self.op_name, verify_dir)
            return
        try:
            shutil.rmtree(verify_dir)
            self._cleanup_empty_op_dir(verify_dir.parent)
            logger.debug("[%s] cleaned verifier artifacts: %s", self.op_name, verify_dir)
        except FileNotFoundError:
            return
        except Exception:
            logger.warning("[%s] failed to clean verifier artifacts: %s", self.op_name, verify_dir,
                           exc_info=True)

    def _cleanup_empty_op_dir(self, op_dir: Path) -> None:
        try:
            op_dir.rmdir()
        except OSError:
            return

    def _framework_filename(self) -> str:
        return f"{self.op_name}_torch.py"

    def _pypto_impl_filename(self) -> str:
        return f"{self.op_name}_pypto_impl.py"

    def _has_impl_source(self, task_info: Dict[str, Any]) -> bool:
        source_files = task_info.get("source_files")
        return bool(task_info.get("coder_code") or (isinstance(source_files, dict) and source_files))

    def _write_source_artifacts(self, task_info: Dict[str, Any], verify_dir: Path) -> None:
        """把 framework_code 与 PyPTO 源文件写入 verify_dir.

        - ``<op>_torch.py``       — 原 KernelBench task_desc (含 Model / get_inputs /
          get_init_inputs).
        - ``source_files``        — 原样写入调用方提供的 ``{op}_impl.py`` /
          ``{op}_pypto_impl.py`` 等文件, 让 wrapper 中的本地 import 自然解析.
        - ``coder_code``          — 兼容旧调用方, 写入 ``{op}_pypto_impl.py``.
        """
        verify_dir.mkdir(parents=True, exist_ok=True)
        framework_file = verify_dir / self._framework_filename()
        framework_file.write_text(self.framework_code, encoding="utf-8")
        source_files = task_info.get("source_files")
        if isinstance(source_files, dict) and source_files:
            for rel_name, content in source_files.items():
                path = Path(str(rel_name))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"非法 verifier 源文件名: {rel_name!r}")
                target = verify_dir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content), encoding="utf-8")
        else:
            impl_code = str(task_info.get("coder_code") or "")
            impl_file = verify_dir / self._pypto_impl_filename()
            impl_file.write_text(impl_code, encoding="utf-8")
        logger.debug(
            "[%s] wrote %s (%d chars)",
            self.op_name,
            framework_file,
            len(self.framework_code),
        )
        logger.debug("[%s] wrote PyPTO source artifacts into %s", self.op_name, verify_dir)

    # ---------- device 管理 (LocalWorker 自带 device pool) ----------

    async def _acquire_device(self, requested_device_id: int) -> int:
        """LocalWorker 模式下从池里 acquire (忽略入参), 否则直接用入参."""
        if self.worker is not None:
            acquired = await self.worker.acquire_device()
            logger.info(f"[{self.op_name}] acquired NPU device {acquired} from worker pool")
            return acquired
        actual = requested_device_id if requested_device_id != -1 else 0
        logger.info(f"[{self.op_name}] using device {actual} (no worker, deprecated path)")
        return actual

    async def _release_device(self, device_id: Optional[int]) -> None:
        if device_id is None or self.worker is None:
            return
        try:
            await self.worker.release_device(device_id)
        except Exception as e:
            logger.warning(f"[{self.op_name}] release device {device_id} failed: {e}")

    # ---------- 内部工具 ----------

    async def _run_one_script(self, script_name: str, cwd: str, timeout: int) -> tuple[bool, str]:
        if self.worker is None:
            raise RuntimeError(
                f"[{self.op_name}] no worker bound; please call register_local_worker(...) "
                "and inject worker via WorkerManager.select() before run/run_profile."
            )
        return await self.worker.run_script(script_name, cwd=cwd, timeout=timeout)
