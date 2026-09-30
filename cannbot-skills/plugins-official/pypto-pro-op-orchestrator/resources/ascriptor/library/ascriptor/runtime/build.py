# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Building the custom-op package and the host harness, running the harness (the old b.sh / r.sh).

Every step is a subprocess with an explicit log file; nothing here inspects the NPU (D-013, and the
maintainer's rule for M5: no idle check). Serialisation of runs on a shared card is a plain advisory
``flock`` on ``lock_path``.
"""

from __future__ import annotations

import fcntl
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


class BuildError(RuntimeError):
    pass


def arch_token() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64-linux"
    if m in ("aarch64", "arm64"):
        return "aarch64-linux"
    raise BuildError(f"unknown machine architecture {m!r}")


def gxx_arch_roots(gxx: str) -> list[str]:
    """The architecture-specific half of a C++ standard library, beside the portable half.

    `ASCRIPTOR_GXX_INCLUDE=/usr/include/c++/11` alone gets as far as `bits/c++config.h file not
    found`, because on a distribution toolchain that header lives in a sibling tree named after
    the architecture. A conda toolchain puts it under the root itself. Both are offered; a
    directory that is not there is simply not added."""
    root = Path(gxx)
    candidates = [root / "x86_64-conda-linux-gnu", *(root.parent.parent).glob(f"*-linux-gnu/c++/{root.name}")]
    return [str(path) for path in candidates if (path / "bits/c++config.h").is_file()]


def toolchain_env(cann_path: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of every build / run: CANN's ``setenv.bash`` is sourced by the shell wrapper; here only
    the variables the build scripts read directly. ``ASCRIPTOR_GXX_INCLUDE`` (a C++ standard-library include
    root) is forwarded as ``CPLUS_INCLUDE_PATH`` for toolchains whose clang cannot find one."""
    env = dict(os.environ)
    env.pop("CXX", None)
    env.pop("CC", None)
    env["ASCEND_HOME_PATH"] = cann_path
    env.setdefault("ASCEND_CUSTOM_OPP_PATH", "")
    env["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + env.get("PATH", "")
    gxx = env.get("ASCRIPTOR_GXX_INCLUDE")
    if gxx:
        roots = [gxx, *gxx_arch_roots(gxx)]
        env["CPLUS_INCLUDE_PATH"] = ":".join([*roots, env.get("CPLUS_INCLUDE_PATH", "")])
    if extra:
        env.update(extra)
    return env


def _bash(script: str, cwd: Path, log: Path, env: dict[str, str], timeout: float) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"# cwd {cwd}\n# {script}\n")
        f.flush()
        r = subprocess.run(["bash", "-c", script], cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        text = log.read_text(encoding="utf-8", errors="replace")
        tail = text.splitlines()[-40:]
        raise BuildError(f"{script.splitlines()[-1] if script else 'command'} failed (exit {r.returncode}); log: {log}\n"
                         + "\n".join(tail) + _gxx_hint(text, env))


def _gxx_hint(log_text: str, env: dict[str, str]) -> str:
    """A missing C++ standard library reads as an ordinary compile error 40 lines deep in a TBE log, and the
    knob that fixes it is an environment variable the failure never mentions. Two sessions read that log as
    'the toolchain is broken here' and reported the cannsim gate as unrunnable (D-235), so it is named at the
    point of failure."""
    if "'type_traits' file not found" not in log_text or env.get("ASCRIPTOR_GXX_INCLUDE"):
        return ""
    roots = [p for p in ("/usr/include/c++/11", "/usr/include/c++/12", "/usr/include/c++/13") if os.path.isdir(p)]
    found = f" one is at {roots[0]}" if roots else " none was found under /usr/include/c++"
    return ("\n\nHINT: bisheng's CCE runtime wrapper includes <type_traits> for this architecture and cannot "
            f"find a C++ standard library. ASCRIPTOR_GXX_INCLUDE is unset;{found}. This is a missing setting, "
            "NOT a broken toolchain -- do not report the gate as unrunnable until it has been tried "
            "(docs/rfc/0007-cce-backend.md, AGENTS.md). The architecture-specific half of the library is "
            "added beside it automatically; if the next failure is `bits/c++config.h file not found`, that "
            "root has no such sibling and CPLUS_INCLUDE_PATH has to name one.")


def build_custom_op(project_dir: Path, cann_path: str, install_dir: Path, *, env: dict[str, str] | None = None,
                    timeout: float = 3600.0) -> Path:
    """``bash build.sh`` then install the produced ``custom_opp_*.run`` under ``install_dir``; returns the vendor root
    (``install_dir/vendors/customize``)."""
    env = env or toolchain_env(cann_path)
    install_dir = Path(install_dir).resolve()
    install_dir.mkdir(parents=True, exist_ok=True)
    setenv = shlex.quote(str(Path(cann_path) / "bin/setenv.bash"))
    script = "\n".join([
        "set -e",
        f"if [ -f {setenv} ]; then source {setenv}; fi",
        f"export ASCEND_HOME_PATH={shlex.quote(cann_path)}",
        "bash build.sh",
        "cd build_out",
        f"for f in custom_*.run; do bash ./$f --install-path={shlex.quote(str(install_dir))}; done",
    ])
    _bash(script, Path(project_dir), Path(project_dir) / "build.log", env, timeout)
    vendor = install_dir / "vendors" / "customize"
    if not vendor.is_dir():
        raise BuildError(f"custom op installed but {vendor} is missing")
    return vendor


def build_harness(test_dir: Path, cann_path: str, vendor_dir: Path, *, env: dict[str, str] | None = None,
                  timeout: float = 600.0) -> Path:
    """Compile ``test.cpp`` against ACL and the installed op API library; returns the executable path."""
    env = env or toolchain_env(cann_path)
    vendor_dir = Path(vendor_dir).resolve()  # the compile runs inside test_dir
    arch = arch_token()
    inc = [f"{cann_path}/{arch}/include", f"{cann_path}/acllib/include", f"{vendor_dir}/op_api/include", "."]
    libdirs = [f"{cann_path}/runtime/lib64", f"{cann_path}/{arch}/lib64", f"{vendor_dir}/op_api/lib"]
    libs = ["runtime", "ascendcl", "stdc++", "nnopbase", "msprofiler", "cust_opapi"]
    cmd = ["g++", "-O2", "-std=c++17", "-pthread", "test.cpp", *[f"-I{p}" for p in inc], *[f"-L{p}" for p in libdirs],
           *[f"-l{lib}" for lib in libs], f"-Wl,-rpath={cann_path}/{arch}/lib64", f"-Wl,-rpath={vendor_dir}/op_api/lib",
           "-o", "test_aclnnop"]
    _bash(" ".join(shlex.quote(c) for c in cmd), Path(test_dir), Path(test_dir) / "harness_build.log", env, timeout)
    return Path(test_dir) / "test_aclnnop"


class FileLock:
    """Advisory lock on a file (``flock``); the shared-box rule wants every run inside one."""

    def __init__(self, path: Path | None, timeout: float = 1800.0) -> None:
        self.path = Path(path) if path else None
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"could not acquire {self.path} within {self.timeout} s") from None
                time.sleep(0.5)

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def run_harness(test_dir: Path, cann_path: str, vendor_dir: Path, *, mode: str = "npu", chipset: str | None = None,
                env: dict[str, str] | None = None, lock_path: Path | None = None, timeout: float = 1800.0,
                extra_env: dict[str, str] | None = None) -> Path:
    """Run the harness (``mode`` = ``npu`` on the card, ``cannsim`` under ``cannsim record``); returns the log path."""
    env = dict(env or toolchain_env(cann_path))
    if extra_env:
        env.update(extra_env)
    if mode == "cannsim":
        if not chipset:
            raise BuildError("cannsim needs the chipset name (the device profile's debug_chipset)")
        cmd = f"cannsim record ./test_aclnnop -s {shlex.quote(chipset)}"  # the record dir lands in test_dir
    elif mode == "npu":
        cmd = "./test_aclnnop ."
    else:
        raise BuildError(f"unknown run mode {mode!r}")
    setenv = shlex.quote(str(Path(cann_path) / "bin/setenv.bash"))
    vendor_env = shlex.quote(str(vendor_dir / "bin/set_env.bash"))
    vendor_lib = shlex.quote(str(vendor_dir / "op_api/lib"))
    script = "\n".join([
        "set -e",
        f"if [ -f {setenv} ]; then source {setenv}; fi",
        f"if [ -f {vendor_env} ]; then source {vendor_env}; fi",
        f'export LD_LIBRARY_PATH={vendor_lib}:"${{LD_LIBRARY_PATH:-}}"',
        cmd,
    ])
    log = Path(test_dir) / "run.log"
    with FileLock(lock_path, timeout=timeout):
        _bash(script, Path(test_dir), log, env, timeout)
    return log


__all__ = ["BuildError", "FileLock", "arch_token", "toolchain_env", "build_custom_op", "build_harness", "run_harness"]
