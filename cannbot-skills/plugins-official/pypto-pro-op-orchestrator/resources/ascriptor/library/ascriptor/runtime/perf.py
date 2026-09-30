# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read msprof's ``op_summary`` CSV into a per-kernel performance record (P0 of the perf track).

``msprof --aic-metrics=PipeUtilization --task-time=on <app>`` writes one row per launched device
task to ``PROF_*/mindstudio_profiler_output/op_summary_*.csv``. The columns we keep:

* ``Task Duration(us)`` — the device-side duration of the task. This is the number to compare
  across backends; host-side JIT dispatch (~400 us on the pypto path, against 8-30 us kernels)
  never enters it, which is why wall-clock timing of these kernels is meaningless.
* ``aic_*_ratio`` / ``aiv_*_ratio`` — the share of the cube / vector core's time each pipe was
  busy. A duration says a kernel is slow; the ratios say which pipe made it slow.
* ``aic_total_cycles`` / ``aiv_total_cycles``, ``Block Num`` / ``Mix Block Num``, ``cube_utilization(%)``.

With ``ASCRIPTOR_REPEAT=N`` the app launches the same kernel N times, so the file carries N rows
for it. :func:`summarise` drops the warm-up rows in launch order and reports the MEDIAN of the
rest plus the spread, because a shared box's neighbours show up as outliers, not as a shift.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

DURATION = "Task Duration(us)"
NAME = "Op Name"
TASK_TYPE = "Task Type"
START = "Task Start Time(us)"

# the pipe columns, in the order a report should print them
AIC_RATIOS = ("aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
              "aic_mte3_ratio", "aic_fixpipe_ratio")
AIV_RATIOS = ("aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio")
EXTRA = ("aic_total_cycles", "aiv_total_cycles", "aicore_time(us)", "aiv_time(us)",
         "cube_utilization(%)", "aic_icache_miss_rate", "aiv_icache_miss_rate")


def _num(v: str) -> float | None:
    v = (v or "").strip()
    if not v or v in ("N/A", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Every task row of one op_summary CSV, in file order (which is launch order)."""
    text = Path(csv_path).read_text(encoding="utf-8", errors="replace")
    return [r for r in csv.DictReader(text.splitlines()) if r.get(NAME)]


def kernel_rows(csv_path: Path) -> list[dict[str, str]]:
    """The rows of the kernel under test: the AI Core tasks, and when several op names appear,
    the one that spent the most time (a run can also carry framework tasks - memset, casts)."""
    rows = [r for r in read_rows(csv_path) if "AI_CPU" not in (r.get(TASK_TYPE) or "")]
    if not rows:
        return []
    by_name: dict[str, float] = {}
    for r in rows:
        by_name[r[NAME]] = by_name.get(r[NAME], 0.0) + (_num(r.get(DURATION, "")) or 0.0)
    top = max(by_name, key=lambda k: by_name[k])
    return [r for r in rows if r[NAME] == top]


def summarise(csv_path: Path, warmup: int = 0) -> dict[str, Any] | None:
    """One perf record from one profiled run, or ``None`` when the CSV holds no kernel task.

    ``warmup`` rows are dropped in launch order before the statistics; the pipe ratios are taken
    from the row whose duration IS the median, so the ratios and the duration describe the same
    launch rather than an average of different ones.
    """
    rows = kernel_rows(Path(csv_path))
    if not rows:
        return None
    rows.sort(key=lambda r: _num(r.get(START, "")) or 0.0)
    kept = rows[warmup:] or rows
    durs = [d for d in (_num(r.get(DURATION, "")) for r in kept) if d is not None]
    if not durs:
        return None
    med = statistics.median(durs)
    mid = min(kept, key=lambda r: abs((_num(r.get(DURATION, "")) or 0.0) - med))
    rec: dict[str, Any] = {
        "op": rows[0][NAME],
        "task_type": rows[0].get(TASK_TYPE, ""),
        "n": len(durs),
        "us": round(med, 3),
        "us_min": round(min(durs), 3),
        "us_max": round(max(durs), 3),
        # the spread that decides whether a difference between two records means anything
        "us_spread": round((max(durs) - min(durs)) / med, 4) if med else None,
        "block_num": mid.get("Block Num", ""),
        "mix_block_num": mid.get("Mix Block Num", ""),
    }
    for col in AIC_RATIOS + AIV_RATIOS + EXTRA:
        v = _num(mid.get(col, ""))
        if v is not None:
            rec[col] = v
    return rec


def busiest_pipe(rec: dict[str, Any]) -> tuple[str, float]:
    """The pipe with the largest ratio, over both cores - the first thing to look at when a
    kernel is slower than its counterpart."""
    best, val = "", 0.0
    for col in AIC_RATIOS + AIV_RATIOS:
        v = float(rec.get(col) or 0.0)
        if v > val:
            best, val = col, v
    return best, val
