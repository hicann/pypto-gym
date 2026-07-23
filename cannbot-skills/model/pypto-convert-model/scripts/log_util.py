# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Logging utilities for the conversion experiment.

Each conversion gets its own log file at $WORKDIR/logs/<model_id>__<format>.log.
A master log records overall progress.

WORKDIR is $CONVERT_MODEL_WORKDIR or ~/.cache/pypto-convert-model. Logs live
outside the skill directory so the repo never accumulates run artifacts.
"""
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

WORKDIR = Path(os.environ.get(
    "CONVERT_MODEL_WORKDIR",
    str(Path.home() / ".cache" / "pypto-convert-model"),
))
LOG_DIR = WORKDIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def master_logger() -> logging.Logger:
    return get_logger("master", LOG_DIR / "master.log")


def conversion_logger(model_id: str, fmt: str) -> logging.Logger:
    safe = model_id.replace("/", "_")
    return get_logger(f"{safe}__{fmt}", LOG_DIR / f"{safe}__{fmt}.log")


@contextmanager
def timed(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info(f"START {label}")
    try:
        yield
        logger.info(f"DONE  {label} ({time.time() - t0:.1f}s)")
    except Exception as e:
        logger.exception(f"FAIL  {label} ({time.time() - t0:.1f}s): {e}")
        raise
