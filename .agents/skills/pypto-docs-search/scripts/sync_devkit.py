#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装配 PyPTO 算子开发资源到本地缓存：docs（主仓）、ops/tests（算子仓）。

工作树已含某类资源则符号链接复用、免重复下载；否则 sparse-checkout 下载。

用法: python3 sync_devkit.py [--pin <git-ref>]
  环境变量覆盖: PYPTO_DEVKIT_DIR / PYPTO_SRC_URL / PYPTO_GYM_URL / PYPTO_SRC / PYPTO_GYM_SRC
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cache_dir():
    d = os.environ.get("PYPTO_DEVKIT_DIR")
    if d:
        return Path(d)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pypto-devkit"


def find_up(marker):
    """从 $PWD 向上查找含标志子路径的仓根（复用已有工作树，避免重复下载）。"""
    d = Path.cwd()
    while True:
        if (d / marker).exists():
            return d
        if d.parent == d:
            return None
        d = d.parent


def relink(dst, src):
    """等价 ln -sfn：替换已存在的目标为指向 src 的符号链接。"""
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def git(args, cwd=None):
    return subprocess.run(
        ["git"] + args, cwd=(str(cwd) if cwd else None),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode


def provision(devkit, tag, marker, url, explicit, pairs, pin, manifest):
    """pairs: [(类名, 仓内子路径), ...]。返回 0 成功，非 0 失败退出码。"""
    base = None
    if explicit and (Path(explicit) / marker).exists():
        base = Path(explicit)
    else:
        base = find_up(marker)

    if base:
        for name, sub in pairs:
            relink(devkit / name, base / sub)
            manifest[name] = {"mode": "symlink", "source": str(base / sub)}
        print("[复用] %s ← %s" % (tag, base))
        return 0

    tmp = devkit / (".repo-" + tag)
    if tmp.exists() or tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)
    if git(["clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(tmp)]) != 0:
        print("[下载失败] %s（设对应 *_URL 覆盖，或检查网络）" % url, file=sys.stderr)
        return 4
    subs = [sub for _name, sub in pairs]
    if git(["sparse-checkout", "set"] + subs, cwd=tmp) != 0:
        print("[sparse 失败] %s" % tag, file=sys.stderr)
        return 4
    if pin:
        git(["fetch", "--depth", "1", "origin", pin], cwd=tmp)
        git(["checkout", pin], cwd=tmp)
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(tmp),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    head = r.stdout.strip() or "unknown"
    for name, sub in pairs:
        relink(devkit / name, tmp / sub)
        manifest[name] = {"mode": "download@" + head, "source": "%s:%s" % (url, sub)}
    print("[下载] %s ← %s@%s" % (tag, url, head))
    return 0


def main():
    devkit = cache_dir()
    src_url = os.environ.get("PYPTO_SRC_URL", "https://gitcode.com/cann/pypto.git")
    gym_url = os.environ.get("PYPTO_GYM_URL", "https://gitcode.com/cann/pypto-gym.git")
    pin = ""
    argv = sys.argv[1:]
    if argv and argv[0] == "--pin":
        pin = argv[1] if len(argv) > 1 else ""

    if shutil.which("git") is None:
        print("[缺 git] 请先安装 git", file=sys.stderr)
        return 3
    try:
        devkit.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("[无法创建缓存] %s" % devkit, file=sys.stderr)
        return 3

    manifest = {}
    rc = provision(devkit, "pypto", "docs/zh/api", src_url, os.environ.get("PYPTO_SRC", ""),
                   [("docs", "docs/zh")], pin, manifest)
    if rc != 0:
        return rc
    rc = provision(devkit, "gym", "src/pypto_gym/ops", gym_url, os.environ.get("PYPTO_GYM_SRC", ""),
                   [("ops", "src/pypto_gym/ops"), ("tests", "tests/ops")], pin, manifest)
    if rc != 0:
        return rc

    (devkit / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[完成] 缓存: %s （docs/ ops/ tests/ + MANIFEST.json）" % devkit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
