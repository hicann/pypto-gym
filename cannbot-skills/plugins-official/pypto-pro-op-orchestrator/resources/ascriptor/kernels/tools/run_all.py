# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run every demo's main.py and report one line per demo.

    python tools/run_all.py                     # every demo, functional simulator
    python tools/run_all.py --launcher pipesim
    python tools/run_all.py --jobs 6            # one pinned CPU per job

This is the repository's acceptance gate: a demo that does not pass here is broken, whatever
its metadata says. Each demo runs in its own process with its own working directory, because
that is how a reader runs it, and pinned to one CPU, because the lane threads of a
multi-core simulation otherwise hand the GIL back and forth instead of working.
"""

import argparse
import concurrent.futures
import json
import queue
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASSED = re.compile(r"^(\d+)/(\d+) cases passed", re.M)


def demos():
    index = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    return [entry["path"] for entry in index["entries"]]


def run(path, launcher, backend, cpus, timeout):
    """`cpus` is a queue, not a number: a worker borrows one CPU for the length of its demo and
    gives it back. Handing each demo a CPU by its position in the index instead would let two
    demos that happen to run at the same time share one core while another sits idle."""
    cpu = cpus.get() if cpus is not None else None
    try:
        return _run(path, launcher, backend, cpu, timeout)
    finally:
        if cpus is not None:
            cpus.put(cpu)


def _run(path, launcher, backend, cpu, timeout):
    folder = ROOT / path
    command = [sys.executable, "main.py", "--launcher", launcher]
    if backend:
        # not every demo declares a second backend; the ones that do take --backend
        if "--backend" in (folder / "main.py").read_text(encoding="utf-8"):
            command += ["--backend", backend]
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]
    start = time.monotonic()
    try:
        done = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return path, "fail", f"TIMEOUT after {timeout}s", time.monotonic() - start
    elapsed = time.monotonic() - start
    match = PASSED.search(done.stdout)
    skipped = [line for line in done.stdout.splitlines() if line.startswith("skipped")]
    if done.returncode == 0 and match and match.group(1) == match.group(2):
        note = f"{match.group(0)}" + (f"  ({len(skipped)} skip note)" if skipped else "")
        return path, "ok", note, elapsed
    # A demo may refuse a whole launcher and say why. That is a third outcome, not a failure:
    # it exits zero, prints its reason and runs nothing, so there is no "cases passed" line.
    if done.returncode == 0 and not match and skipped:
        return path, "skip", skipped[0], elapsed
    tail = (done.stderr or done.stdout).strip().splitlines()
    return path, "fail", (match.group(0) if match else "") + " | " + (tail[-1] if tail else "no output"), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim")
    parser.add_argument("--backend", default="",
                        help="cce or pto_isa; demos that declare only one backend ignore it")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--filter", default="", help="only demos whose path contains this")
    parser.add_argument("--cpus", default="",
                        help="comma-separated CPUs to pin to, borrowed one per running demo; "
                             "unset means no pinning, which is right when the simulator forks")
    args = parser.parse_args()

    selected = [p for p in demos() if args.filter in p]
    pinned = [int(c) for c in args.cpus.split(",") if c]
    pool_of_cpus = None
    if pinned:
        pool_of_cpus = queue.Queue()
        for cpu in (pinned * args.jobs)[:args.jobs]:
            pool_of_cpus.put(cpu)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for path in selected:
            futures[pool.submit(run, path, args.launcher, args.backend,
                                pool_of_cpus, args.timeout)] = path
        for future in concurrent.futures.as_completed(futures):
            path, status, note, elapsed = future.result()
            results.append((path, status, note, elapsed))
            mark = {"ok": "ok  ", "skip": "skip", "fail": "FAIL"}[status]
            print(f"{mark} {elapsed:7.1f}s  {path:62s} {note}", flush=True)

    failed = [r for r in results if r[1] == "fail"]
    skips = [r for r in results if r[1] == "skip"]
    total = sum(r[3] for r in results)
    where = args.launcher + (f"/{args.backend}" if args.backend else "")
    print(f"\n{len(results) - len(failed) - len(skips)}/{len(results)} demos passed under {where}"
          + (f", {len(skips)} refused it" if skips else "")
          + f" ({total / 60:.1f} CPU-minutes)")
    for path, _, note, _ in sorted(skips):
        print(f"  skip {path}: {note}")
    for path, _, note, _ in sorted(failed):
        print(f"  FAIL {path}: {note}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
