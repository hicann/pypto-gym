# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device execution on the machine that owns the card. No network transport is provided."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import devices as _devices

REQUIRED = ("workspace",)
OPTIONAL = {"env_script": None, "lock": None, "visible_devices": None,
            "cann_path": None, "python": "python", "cube_cores": None, "local": False}


def config_file() -> Path:
    env = os.environ.get("ASCRIPTOR_BOARDS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "boards.json"


class BoardError(RuntimeError):
    pass


class Board:
    """One box of ``boards.json``. Values are held, never printed."""

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        if cfg.get("local") is not True:
            raise BoardError(f"boards.json[{name!r}] must describe this machine with local=true")
        if set(cfg) & {"ssh", "port", "proxy_jump", "identity", "control_path", "control_persist"}:
            raise BoardError(f"boards.json[{name!r}] contains unsupported remote connection fields")
        missing = [key for key in REQUIRED if not cfg.get(key)]
        if missing:
            raise BoardError(f"boards.json[{name!r}]: missing {missing}")
        self.cfg = {**OPTIONAL, **cfg}

    @classmethod
    def from_config(cls, name: str, path: Path | None = None) -> Board:
        p = path or config_file()
        if not p.is_file():
            raise BoardError(f"no boards.json at {p} — derive one from machine_specs.md (git-ignored; see docs/rfc/0007-cce-backend.md)")
        data = json.loads(p.read_text(encoding="utf-8"))
        aliases = {"950": "a5", "950pr": "a5", "a5pr": "a5", "b1": "a2", "b2": "a2", "b3": "a2", "b4": "a2"}
        key = name if name in data else aliases.get(name, name)
        if key not in data:
            raise BoardError(f"boards.json has no entry {name!r} (keys: {sorted(data)})")
        return cls(key, data[key])

    @classmethod
    def local(cls, path: Path | None = None) -> Board:
        """The board this process is running ON — never one to connect to.

        A launcher that executes on the device runs on the device's own machine (RFC-0012 §board).
        The machine says which entry describes it by marking that entry ``"local": true``; which
        file is read comes from ``ASCRIPTOR_BOARDS``, which the box's environment script exports.
        ``ASCRIPTOR_BOARD`` names one when a box carries several.

        The refusals are the point: a workstation has no local entry, so a launcher that needs the
        device stops here with what to do instead, rather than opening a connection nobody asked for.
        """
        p = path or config_file()
        if not p.is_file():
            raise BoardError(
                f"no board config at {p}: this machine does not describe itself as a board. "
                "A device launcher runs on the card machine; set "
                "ASCRIPTOR_BOARDS to a config whose entry for this machine says \"local\": true.")
        data = json.loads(p.read_text(encoding="utf-8"))
        local = {k: v for k, v in data.items() if isinstance(v, dict) and v.get("local")}
        named = os.environ.get("ASCRIPTOR_BOARD")
        if named:
            if named not in data:
                raise BoardError(f"ASCRIPTOR_BOARD={named!r} is not in {p} (keys: {sorted(data)})")
            if named not in local:
                raise BoardError(f"ASCRIPTOR_BOARD={named!r} does not describe this machine: its entry "
                                 "is missing \"local\": true")
            return cls.from_config(named, path=p)
        if not local:
            raise BoardError(
                f"no entry in {p} says \"local\": true, so none of them describes this machine. "
                "Run the source snapshot on the card machine with ASCRIPTOR_BOARDS set to "
                "a config describing that machine.")
        if len(local) > 1:
            raise BoardError(f"{p} has {len(local)} entries describing this machine "
                             f"({sorted(local)}) — set ASCRIPTOR_BOARD to the one to use")
        return cls.from_config(next(iter(local)), path=p)

    def require_local(self, what: str) -> None:
        """Refuse to drive execution from anywhere but the box itself."""
        if not self.cfg.get("local"):
            raise BoardError(
                f"{what} executes on the device, so it runs on the device's own machine; board "
                f"{self.name!r} does not describe this machine. Run on the assigned card machine.")

    # ---------------------------------------------------------------- local process

    def _redact(self, text: str) -> str:
        for k in ("workspace", "env_script", "lock", "cann_path"):
            value = self.cfg.get(k)
            if value:
                text = text.replace(str(value), f"<{k}>")
        return text

    def _run(self, cmd: list, what: str, *, timeout: float, **kw: Any) -> subprocess.CompletedProcess:
        """Run a local process and redact machine-specific configuration from errors."""
        try:
            return subprocess.run(cmd, timeout=timeout, **kw)
        except subprocess.TimeoutExpired as exc:
            out = exc.output or b""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            raise BoardError(f"{what} timed out after {timeout:.0f}s on {self.name!r}: "
                             f"{self._redact(out[-500:])}") from None
        except OSError as exc:
            raise BoardError(f"{what} failed to launch on {self.name!r}: "
                             f"{self._redact(str(exc))}") from None

    def _run_shell(self, script: str, *, timeout: float = 600.0, check: bool = True,
                   login: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
        command = ["bash", "-lc" if login else "-c", script]
        result = self._run(command, "local command", timeout=timeout, capture_output=True,
                           text=True, input=stdin)
        if check and result.returncode:
            raise BoardError(f"local command failed (exit {result.returncode}):\n"
                             f"{self._redact(result.stderr[-2000:])}\n{self._redact(result.stdout[-2000:])}")
        return result

    # ---------------------------------------------------------------- the box-side script

    def device_env(self) -> str:
        lines = ["set -e"]
        if self.cfg["env_script"]:
            # the box's own env script is not ours: a stale line inside it must not abort the run
            # under `set -e` with no output at all (a CANN reinstall did exactly that once, D-120)
            lines.append(f"set +e; source {shlex.quote(str(self.cfg['env_script']))}; set -e")
        if self.cfg["cann_path"]:
            lines.append(f"export ASCEND_HOME_PATH={shlex.quote(str(self.cfg['cann_path']))}")
        if self.cfg["visible_devices"] is not None:
            lines.append(f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(str(self.cfg['visible_devices']))}")
        lines.append('if [ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]; then source "$ASCEND_HOME_PATH/bin/setenv.bash"; fi')
        return "\n".join(lines)

    # ---------------------------------------------------------------- the profiler (P0)

    @staticmethod
    def perf_env() -> tuple[bool, int]:
        """``(profile, repeat)`` from the environment. ``ASCRIPTOR_PROFILE=1`` wraps the run in
        ``msprof``; ``ASCRIPTOR_REPEAT=N`` makes the driver launch the kernel N times so the
        profiler collects N task records. Both default off, so a correctness run is unchanged."""
        prof = os.environ.get("ASCRIPTOR_PROFILE", "").strip() not in ("", "0", "off", "false")
        reps = max(1, int(os.environ.get("ASCRIPTOR_REPEAT", "1") or 1))
        return prof, reps

    @staticmethod
    def msprof_wrap(cmd: str, out: str) -> str:
        """``cmd`` under msprof. ``PipeUtilization`` is msprof's default aic-metrics group and is
        the one that fills the ``aic_*_ratio`` / ``aiv_*_ratio`` columns of ``op_summary``; the
        durations we read (``Task Duration(us)``) are device-side, so the profiler's own host
        overhead does not enter them."""
        # ASCRIPTOR_AIC_METRICS picks a different counter group for a follow-up question -
        # ResourceConflictRatio (pipe / bank conflicts), MemoryUB, ArithmeticUtilization, L2Cache
        metrics = os.environ.get("ASCRIPTOR_AIC_METRICS", "PipeUtilization").strip() or "PipeUtilization"
        return f"rm -rf {shlex.quote(out)} && mkdir -p {shlex.quote(out)} && " \
               f"msprof --output={shlex.quote(out)} --aic-metrics={shlex.quote(metrics)} --task-time=on {cmd}"

    def read_op_summary(self, prof_dir: str, dest: Path) -> bool:
        """Copy the run's ``op_summary`` CSV off the box. Small (one row per launched task), so it
        travels through ``cat`` rather than a tar, and it is the raw evidence for a perf record."""
        r = self._run_shell(f"cat {shlex.quote(prof_dir)}/PROF_*/mindstudio_profiler_output/op_summary_*.csv 2>/dev/null || true",
                     check=False, login=False)
        text = r.stdout.strip()
        if not text:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self._redact(text) + "\n", encoding="utf-8")
        return True

    def run_script(self, run_dir: str, mode: str, chipset: str) -> str:
        """The script executed on the box for one OpExec run: build the package (once per source hash), build
        the harness, then run it inside the workspace lock."""
        lock = self.cfg["lock"] or f"{self.cfg['workspace']}/.locks/ascriptor_npu.lock"
        cmd = "./test_aclnnop ." if mode == "npu" else f"cannsim record ./test_aclnnop -s {shlex.quote(chipset)}"
        prof, reps = self.perf_env()
        if prof and mode == "npu":
            cmd = self.msprof_wrap(cmd, "./prof")
        return "\n".join([
            self.device_env(),
            f"export ASCRIPTOR_REPEAT={reps}",
            f"cd {shlex.quote(run_dir)}",
            'ARCH=$(uname -m)-linux',
            'INSTALL="$PWD/custom_op"; VENDOR="$INSTALL/vendors/customize"',
            'if [ ! -f .built ] || [ "$(cat .built)" != "$(cat .source_hash)" ]; then',
            '  (cd project && rm -rf build_out && bash build.sh && cd build_out && for f in custom_*.run; do bash ./$f --install-path="$INSTALL"; done) > build.log 2>&1',
            '  (cd aclnn_test && g++ -O2 -std=c++17 -pthread test.cpp -I"$ASCEND_HOME_PATH/$ARCH/include" -I"$ASCEND_HOME_PATH/acllib/include" '
            '-I"$VENDOR/op_api/include" -I. -L"$ASCEND_HOME_PATH/runtime/lib64" -L"$ASCEND_HOME_PATH/$ARCH/lib64" -L"$VENDOR/op_api/lib" '
            '-lruntime -lascendcl -lstdc++ -lnnopbase -lmsprofiler -lcust_opapi -Wl,-rpath="$ASCEND_HOME_PATH/$ARCH/lib64" '
            '-Wl,-rpath="$VENDOR/op_api/lib" -o test_aclnnop) > harness_build.log 2>&1',
            '  cp .source_hash .built',
            'fi',
            'if [ -f "$VENDOR/bin/set_env.bash" ]; then source "$VENDOR/bin/set_env.bash"; fi',
            'export LD_LIBRARY_PATH="$VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"',
            f"mkdir -p {shlex.quote(os.path.dirname(lock))} aclnn_test/output",  # the tar ships files only: recreate the output dir
            f"cd aclnn_test && flock -w 1800 {shlex.quote(lock)} {cmd} > run.log 2>&1",
        ])

    # ---------------------------------------------------------------- OpExec's board launcher

    def run_opexec(self, ex: Any, tensors: dict[str, Any], scalars: dict[str, Any]) -> dict[str, Any]:
        from . import harness as _harness

        self.require_local("the cce board launcher")
        ex.write_sources()
        # Stage sources and inputs beside this run's local output.
        _harness.write_args(ex.spec, ex.test_dir, {k: _to_numpy(v) for k, v in tensors.items()}, scalars, seed_outputs=ex.seed_outputs)
        stage = ex.out_dir / "board_stage"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        shutil.copytree(ex.project_dir, stage / "project", ignore=shutil.ignore_patterns("build_out", "__pycache__"))
        if self.cfg["cann_path"]:  # use this machine's selected CANN path
            preset = stage / "project" / "CMakePresets.json"
            data = json.loads(preset.read_text(encoding="utf-8"))
            for cp in data.get("configurePresets", []):
                cv = cp.get("cacheVariables", {})
                if "ASCEND_CANN_PACKAGE_PATH" in cv:
                    cv["ASCEND_CANN_PACKAGE_PATH"]["value"] = str(self.cfg["cann_path"])
            preset.write_text(json.dumps(data, indent=4), encoding="utf-8")
        shutil.copytree(ex.test_dir, stage / "aclnn_test", ignore=shutil.ignore_patterns("output", "test_aclnnop", "*.log"))
        (stage / "aclnn_test" / "output").mkdir()
        shutil.copy(ex.out_dir / ".source_hash", stage / ".source_hash")
        profile = _devices.load(ex.device)
        mode = "npu"
        t0 = time.time()
        # The stage is recreated for each local run, so removed sources cannot remain importable.
        run_dir = str(stage)
        ex.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._run_shell(self.run_script(run_dir, mode, profile.debug_chipset), timeout=ex.timeout * 3)
        finally:
            # in a ``finally`` because a failing run is exactly when the logs are wanted: the
            # a failed local script can return little output, so fetching after
            # the call left every ``exit 2`` undiagnosable from the run directory
            # (``run_pypto`` below has always done it this way).
            for name in ("run.log", "build.log", "harness_build.log"):
                r = self._run_shell(f"cat {shlex.quote(run_dir)}/aclnn_test/{name} 2>/dev/null || cat {shlex.quote(run_dir)}/{name} 2>/dev/null || true",
                             check=False, login=False)
                ex.out_dir.mkdir(parents=True, exist_ok=True)
                (ex.out_dir / f"board_{name}").write_text(self._redact(r.stdout), encoding="utf-8")
        shutil.copytree(stage / "aclnn_test/output", ex.test_dir / "output", dirs_exist_ok=True)
        (ex.out_dir / "board_timing.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1)}), encoding="utf-8")
        if self.perf_env()[0]:
            self.read_op_summary(f"{run_dir}/aclnn_test/prof", ex.out_dir / "perf" / "op_summary.csv")
        return _harness.read_outputs(ex.test_dir, ex.spec, tensors)


    def pypto_run_script(self, run_dir: str) -> str:
        """The script for one generated-PyPTO run on this machine."""
        lock = self.cfg["lock"] or f"{self.cfg['workspace']}/.locks/ascriptor_npu.lock"
        prof, reps = self.perf_env()
        cmd = "python3 run_case.py"
        if prof:
            cmd = self.msprof_wrap(cmd, "./prof")
        return "\n".join([
            self.device_env(),
            f"cd {shlex.quote(run_dir)}",
            "export TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID:-0}",
            f"export ASCRIPTOR_REPEAT={reps}",
            "mkdir -p output " + shlex.quote(os.path.dirname(lock)),
            f"flock -w 1800 {shlex.quote(lock)} {cmd} > run.log 2>&1",
        ])

    def run_pypto(self, ex: Any, tensors: dict[str, Any], scalars: dict[str, Any]) -> dict[str, Any]:
        """Run generated PyPTO sources and raw-byte inputs on this machine."""
        self.require_local("the pypto launcher")
        # The card's own core count, not the device profile's. `emit` falls back to the profile
        # (a5 says 32) when this is absent, and a vec module doubles that to 64 AIVs - on a 28/56
        # card `sync_all` then waits for cores the launch named and nobody runs, which is a hang,
        # not an error (simt_atomic_add sat there for nine minutes). A missing field must be loud.
        if not self.cfg.get("cube_cores"):
            raise BoardError(
                f"boards.json[{self.name!r}] has no \"cube_cores\": the pypto launcher would emit a "
                "block_dim from the device profile instead of this card's own core count, and a "
                "launch past the card deadlocks in sync_all. Set it to the card's AIC count "
                "(an Ascend950PR is 28, not the a5 profile's 32).")
        spec = ex.spec
        # pl has no tensor-list parameter, so a GMList's arity and member shapes join the
        # specialisation and the printer expands it into one tensor parameter per member (D-119)
        lists = {p["ir_name"]: [list(t.shape) for t in tensors[p["name"]]] for p in spec.lists}
        shapes = {p["ir_name"]: list(tensors[p["name"]].shape) for p in spec.tensors
                  if p["name"] in tensors and hasattr(tensors[p["name"]], "shape")}
        arts = ex.pypto_artifacts({p["ir_name"]: scalars[p["name"]] for p in spec.scalars},
                                  max_block_dim=self.cfg.get("cube_cores"), lists=lists or None,
                                  shapes=shapes)
        stage = ex.out_dir / "pypto_stage"
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "input").mkdir(parents=True)
        for name, data in arts.files.items():
            (stage / name).write_bytes(data)
        import torch

        params = []
        by_ir = {p["ir_name"]: tensors[p["name"]] for p in spec.tensors + spec.lists}
        members_of: dict[str, list[str]] = {}
        for p in arts.metadata["params"]:  # the PYPTO signature: a list is already its members
            if p["kind"] != "tensor":
                continue
            ir = p["ir_name"]
            if "#" in ir:  # one member of a GMList parameter
                base, idx = ir.split("#")
                v = by_ir[base][int(idx)]
                members_of.setdefault(base, []).append(p["name"])
            else:
                v = by_ir[ir]
            t = v.detach().cpu().contiguous()
            if p["output"] and not ex.seed_outputs:
                t = torch.full((t.numel() * t.element_size(),), 255, dtype=torch.uint8).view(t.dtype).reshape(t.shape)
            (stage / "input" / (p["name"] + ".bin")).write_bytes(t.view(torch.uint8).numpy().tobytes())
            # a dense MX scale is the same bytes seen as rank 3: pto's scale load wants a
            # trailing physical-phase axis, so the driver reshapes before the call
            shape = arts.metadata["pypto"].get("reshape", {}).get(p["name"]) or list(t.shape)
            params.append({"name": p["name"], "shape": shape,
                           "torch_dtype": arts.metadata["pypto"]["torch_dtypes"][p["name"]]})
        manifest = {"entry": arts.metadata["kernel"], "params": params,
                    # trailing scalar arguments carrying float constants exactly (D-134)
                    "const_scalars": arts.metadata["pypto"].get("const_scalars", []),
                    # a GMList output is its MEMBERS on this path: the driver writes one file each
                    "outputs": [n for p in spec.outputs for n in members_of.get(p["ir_name"], [p["name"]])],
                    "block_dim": arts.metadata.get("block_dim", ex.block_dim),
                    # the local PyPTO-Pro patches this generated source needs, each with its own
                    # probe: the emitter knows what it printed, only the box knows what is
                    # installed, so the driver checks before it compiles anything
                    "supplements": arts.metadata["pypto"].get("supplements", []),
                    "workspace_bytes": int(arts.metadata.get("workspace_bytes", 0))}
        (stage / "input" / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        run_dir = str(stage)
        t0 = time.time()
        failure: BoardError | None = None
        try:
            self._run_shell(self.pypto_run_script(run_dir), timeout=ex.timeout * 3)
        except BoardError as exc:  # re-raised below, once the log that explains it is in hand
            failure = exc
        finally:
            r = self._run_shell(f"cat {shlex.quote(run_dir)}/run.log 2>/dev/null || true", check=False, login=False)
            ex.out_dir.mkdir(parents=True, exist_ok=True)
            (ex.out_dir / "board_pypto_run.log").write_text(self._redact(r.stdout), encoding="utf-8")
        if failure is not None:
            # The driver's own output explains the failure; the launcher's exit status does not.
            # Both dead-CANN-tree signatures (`libhccl.so: cannot open shared object file`,
            # `F7A008 FILE_ERROR … aarch64 toolchain g++`) reach the caller as an empty message
            # otherwise, and the reader is sent to a file to find out what happened.
            tail = "\n".join(self._redact(r.stdout).splitlines()[-25:]).strip()
            raise BoardError(f"{failure}\n--- the last lines of the run log on the box "
                             f"(full copy: {ex.out_dir / 'board_pypto_run.log'})\n{tail}") from failure
        out_dir = ex.out_dir / "pypto_output"
        shutil.copytree(stage / "output", out_dir, dirs_exist_ok=True)
        (ex.out_dir / "board_timing.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1)}), encoding="utf-8")
        if self.perf_env()[0]:
            self.read_op_summary(f"{run_dir}/prof", ex.out_dir / "perf" / "op_summary.csv")
        outputs: dict[str, Any] = {}
        for p in spec.outputs:
            names = members_of.get(p["ir_name"], [p["name"]])  # a GMList output: its members, in order
            blobs = []
            for n in names:
                f = out_dir / (n + ".bin")
                if not f.is_file():
                    raise BoardError(f"pypto run produced no output {n} - see board_pypto_run.log")
                blobs.append(f.read_bytes())
            outputs[p["name"]] = blobs if p["kind"] == "list" else blobs[0]
        return outputs


def _to_numpy(v: Any) -> Any:
    import torch

    if isinstance(v, (list, tuple)):  # a gmlist argument: its members
        return [_to_numpy(m) for m in v]
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16 or "float8" in str(t.dtype) or t.dtype == getattr(torch, "complex32", None):
            return (tuple(t.shape), t.view(torch.uint8).numpy())  # bytes + the logical shape
        return t.numpy()
    return v


__all__ = ["Board", "BoardError", "config_file"]
