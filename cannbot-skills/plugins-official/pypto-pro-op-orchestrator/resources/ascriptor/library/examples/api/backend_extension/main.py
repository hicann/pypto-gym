# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The public Backend protocol: implement it, publish an entry point, be discovered, emit the same.

    python main.py                          # the conformance checks, then every case on sim
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # through the registered backend, on this card
    python main.py --backend cce            # the same kernel through the built-in backend

`backend_plugin.py` holds `TeachingCceBackend`: a transparent adapter that satisfies the accepted
`Backend` protocol, returns `Capabilities`, `ResourceLimits` and `Artifacts`, delegates every
instruction to CCE and adds one key to the artifact metadata. It deliberately implements no new
lowering -- a real instruction would need its own semantic and vendor gates, and nothing here would
stand in for them.

Six things are checked before the cases run, and each is reported on its own line:

    protocol    the class is an instance of `Backend` and returns a `Capabilities`
    resources   its `ResourceLimits` for the device are equal to CCE's
    artifacts   its emitted files and entry symbol are identical to CCE's, byte for byte
    metadata    it adds `teaching_adapter` to the artifact metadata, so the adapter is traceable
    discovery   `discover()` and `get()` find it by name under the `ascriptor.backends` group
    published   `pyproject.toml` declares that same group, name and object path

Discovery is exercised with temporary distribution metadata placed on this process's import path.
Nothing is installed and no dependency is touched: the directory is removed when the check ends, and
`pyproject.toml` is the file that shows what an installed distribution would have published instead.

Then the copy kernel runs through the *registered name*, and the result is compared to the input
bitwise -- a protocol conformance claim is worth little without one numerical result behind it.

This folder carries `backend_plugin.py` and `pyproject.toml` beside the usual four files, because for
a protocol example those two are the subject. The index records them.
"""

import argparse
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from ascriptor.runtime import LAUNCHERS, OpExec, compile_kernel

from backend_plugin import TeachingCceBackend, check_protocol
from kernel import copy_tile
from reference import make_inputs, reference

DEVICE = "a5"
NAME = "teaching_cce"
GROUP = "ascriptor.backends"
OBJECT = "backend_plugin:TeachingCceBackend"

POISON = float("nan")     # an element the copy never wrote is a NaN

OUTPUTS = ("copy",)
CHECKS = ("protocol", "resources", "artifacts", "metadata", "discovery", "published")

CASES = [
    {"id": "copy_row", "seed": 8891, "block_dim": 1, "parameters": {},
     "purpose": "One complete FP32 row through the registered backend. The kernel is a copy on "
                "purpose: the backend is the only variable, so a difference is the backend's"},
    {"id": "second_seed", "seed": 8892, "block_dim": 1, "parameters": {},
     "purpose": "The same program on different data, so the copy is not matching by coincidence"},
]


@contextmanager
def registered_for_process():
    """Standard entry-point discovery, using temporary distribution metadata on this process's path.

    This is what installing `pyproject.toml` would produce. It installs nothing, overwrites no
    dependency, and the directory is gone when the block ends.
    """
    with tempfile.TemporaryDirectory(prefix="ascriptor-backend-example-") as location:
        info = Path(location) / "ascriptor_teaching_backend-0.1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: ascriptor-teaching-backend\nVersion: 0.1.0\n")
        (info / "entry_points.txt").write_text(f"[{GROUP}]\n{NAME} = {OBJECT}\n")
        sys.path.insert(0, location)
        try:
            yield
        finally:
            sys.path.remove(location)


def check_domain(inputs, expected):
    """A copy's reference is its input, and it must be a separate object -- otherwise a kernel that
    handed back the caller's own tensor would compare equal to it for the wrong reason."""
    if not torch.equal(expected["copy"], inputs["x"]):
        raise ValueError("the reference for a copy is the input")
    if expected["copy"].data_ptr() == inputs["x"].data_ptr():
        raise ValueError("the reference must not alias the input")


def conformance(out_dir):
    """The six protocol facts, each printed with its own verdict. Returns the ones that failed."""
    from ascriptor.backends.base import ResourceLimits, discover, get
    from ascriptor.backends.cce import CceBackend
    from ascriptor.devices import load
    from ascriptor.passes import PIPELINE, PassManager

    failed, note = [], {}
    backend, cce, device = TeachingCceBackend(), CceBackend(), load(DEVICE)
    verdict = {"protocol": bool(check_protocol() is not None)}
    limits = backend.resources(device)
    verdict["resources"] = isinstance(limits, ResourceLimits) and limits == cce.resources(device)
    module = PassManager(PIPELINE).run(copy_tile.ir())
    mine, theirs = backend.compile(module), cce.compile(module)
    verdict["artifacts"] = mine.files == theirs.files and mine.entry == theirs.entry
    note["artifacts"] = f"{len(mine.files)} file(s), entry {mine.entry}"
    verdict["metadata"] = mine.metadata.get("teaching_adapter") == NAME
    with registered_for_process():
        found = discover()
        verdict["discovery"] = NAME in found and get(NAME).name == NAME
        note["discovery"] = f"{len(found)} backend(s) discovered"
        artifact = compile_kernel(copy_tile, backend=NAME, block_dim=1)
    for file_name, data in artifact.files.items():
        path = out_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    declared = (Path(__file__).parent / "pyproject.toml").read_text(encoding="utf-8")
    verdict["published"] = f'[project.entry-points."{GROUP}"]' in declared and \
        f'{NAME} = "{OBJECT}"' in declared
    note["published"] = "pyproject.toml declares the same group, name and object"
    for name in CHECKS:
        print(f"    {name:11s} {'ok  ' if verdict[name] else 'FAIL'}  {note.get(name, '')}")
        if not verdict[name]:
            failed.append(name)
    return failed


def execute(case, inputs, launcher, backend):
    """One launch, through the registered backend name rather than a built-in one."""
    with registered_for_process():
        op = OpExec(copy_tile, launcher=launcher, backend=backend, device=DEVICE,
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                    seed_outputs=True)
        return {"copy": op(inputs["x"], torch.full_like(inputs["x"], POISON))}


def compare(name, got, want):
    """Bitwise. A copy has no arithmetic in it, so it has no tolerance either."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:11s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:11s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} FP32 elements")
    if not ok:
        differ = (got != want).flatten()
        index = differ.nonzero().flatten().tolist()
        unwritten = int(torch.isnan(got).sum())
        print(f"      {len(index)}/{got.numel()} differ, first at {index[:6]}"
              + (f"; {unwritten} are still the NaN fill (never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default=NAME, choices=(NAME, "cce"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:14s} {case['purpose']}")
        print("\nconformance checks: " + ", ".join(CHECKS))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    out_dir = Path(f"tmp/{args.launcher}/conformance")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"conformance  ({NAME} against cce, entry-point group {GROUP})")
    failed = conformance(out_dir)
    for case in selected:
        print(f"{case['id']}  (backend={args.backend}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    cases_failed = {f.split("/")[0] for f in failed if "/" in f}
    print(f"\n{len(selected) - len(cases_failed)}/{len(selected)} cases passed, "
          f"{len(CHECKS) - len([f for f in failed if '/' not in f])}/{len(CHECKS)} "
          f"conformance checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
