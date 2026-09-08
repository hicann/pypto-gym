#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立装配 PyPTO-Pro 文档和官方指定样例；不依赖普通 PyPTO 同步器。

python sync_devkit.py --samples <official_samples.md> [--pin <git-ref>] [--check]
PYPTO_DEVKIT_DIR：缓存目录（默认 cwd/.devkit）
PYPTO_SRC：已有 PyPTO 工作树；PYPTO_SRC_URL：远程仓（默认官方 pypto）
--pin 从远端获取指定版本，不修改已有工作树；--check 仅检查本地缓存。
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

DOC_PATHS = (
    "api/pro_api",
    "guide/programming_guide/pro",
    "guide/quick_start/pro",
    "guide/introduction.md",
    "guide/figures/pro",
)
FRONTEND = "python/tests/st/pypto_pro/frontend"
MARKER = ".pro-docs-v2"


def samples_from(path):
    samples = set(re.findall(r"`(pro_ops/[^`]+\.py)`", path.read_text(encoding="utf-8")))
    if not samples:
        raise ValueError("官方样例清单为空")
    for sample in samples:
        parts = PurePosixPath(sample).parts
        if len(parts) < 2 or ".." in parts or "\\" in sample:
            raise ValueError(f"无效样例路径：{sample}")
    return sorted(samples)


def source_sample(sample):
    return FRONTEND + "/" + sample.removeprefix("pro_ops/")


def git(*args, cwd=None, input_text=None):
    result = subprocess.run(
        ["git", *args], cwd=cwd, input=input_text, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git 命令失败")
    return result.stdout.strip()


def download(url, pin, destination, samples):
    git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(destination))
    if pin:
        git("fetch", "--depth", "1", "origin", pin, cwd=destination)
    paths = ["/docs/zh/" + path + ("" if path.endswith(".md") else "/") for path in DOC_PATHS]
    paths.extend("/" + source_sample(sample) for sample in samples)
    # 自定义 Git 模板可能省略 info；旧版 Git 的 sparse-checkout 不会补建。
    (destination / ".git/info").mkdir(parents=True, exist_ok=True)
    git("config", "core.sparseCheckout", "true", cwd=destination)
    git("config", "core.sparseCheckoutCone", "false", cwd=destination)
    git("sparse-checkout", "set", "--stdin", cwd=destination, input_text="\n".join(paths) + "\n")
    git("checkout", "--detach", "FETCH_HEAD" if pin else "HEAD", cwd=destination)


def local_source():
    explicit = os.environ.get("PYPTO_SRC")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "docs/zh/api/pro_api").is_dir():
            return candidate
    return None


def require_docs(docs):
    if not (docs / "api/pro_api/index.md").is_file():
        raise FileNotFoundError("缺少 Pro API 总索引：api/pro_api/index.md")
    for relative in ("guide/programming_guide/pro", "guide/quick_start/pro"):
        if not (docs / relative / "index.md").is_file():
            raise FileNotFoundError("缺少 Pro 指南总索引：" + relative + "/index.md")
    if not (docs / "guide/introduction.md").is_file():
        raise FileNotFoundError("缺少 PyPTO 简介：guide/introduction.md")


def ready(cache, samples):
    try:
        if not (cache / MARKER).is_file() or not (cache / "MANIFEST.json").is_file():
            return False
        require_docs(cache / "docs")
        if not (cache / "docs/pypto_pro/api/index.md").is_file():
            return False
        forbidden = ("install", "contribute", "invocation", "api/tensor_api",
                     "guide/programming_guide/tensor", "guide/quick_start/tensor", "pypto_pro/tutorials")
        if any((cache / "docs" / path).exists() for path in forbidden):
            return False
        actual = {path.relative_to(cache).as_posix() for path in (cache / "pro_ops").rglob("*.py") if path.is_file()}
        return actual == set(samples)
    except OSError:
        return False


def remove_cached(path):
    # 仅删除明确的缓存目标；符号链接/Windows 目录联接只移除链接自身。
    if path.is_symlink():
        path.unlink()
    elif getattr(path, "is_junction", lambda: False)():
        path.rmdir()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def compatibility_api(docs):
    source = docs / "api/pro_api"
    target = docs / "pypto_pro/api"
    shutil.copytree(source, target)
    (docs / "api/index.md").write_text("# PyPTO Pro API\n\n[API 总索引](pro_api/index.md)\n", encoding="utf-8")


def assemble(source, staged, samples, revision, url):
    original_docs = source / "docs/zh"
    require_docs(original_docs)
    for relative in DOC_PATHS:
        original, target = original_docs / relative, staged / "docs" / relative
        if relative == "guide/figures/pro" and not original.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, target)
        else:
            shutil.copy2(original, target)
    for sample in samples:
        original, target = source / source_sample(sample), staged / sample
        if not original.is_file():
            raise FileNotFoundError("官方样例缺失：" + source_sample(sample))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    compatibility_api(staged / "docs")
    # 仅转换指南中明确指向未缓存资料的跳转；保留代码、图片和 Pro 内部链接。
    external = re.compile(
        r"\]\(((?:\.\./)+(?:install/prepare_environment\.md|"
        r"api/tensor_api/config/pypto-set_(?:host|pass|codegen|verify|debug)_options\.md))\)"
    )
    for page in (staged / "docs/guide").rglob("*.md"):
        def remote(match, page=page):
            original = original_docs / page.relative_to(staged / "docs")
            path = (original.parent / match.group(1)).resolve().relative_to(source)
            ref = revision if revision != "unversioned" else "master"
            return (
                "](" + url.removesuffix(".git") + "/blob/" + quote(ref, safe="")
                + "/" + quote(path.as_posix(), safe="/") + ")"
            )
        content = page.read_text(encoding="utf-8")
        replaced = external.sub(remote, content)
        if content != replaced:
            page.write_text(replaced, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="统一官方样例清单")
    parser.add_argument("--pin", default="", help="下载指定 git ref")
    parser.add_argument("--check", action="store_true", help="仅检查缓存，不联网、不修改文件")
    args = parser.parse_args(argv)
    cache = Path(os.environ.get("PYPTO_DEVKIT_DIR", Path.cwd() / ".devkit")).expanduser().resolve()
    try:
        samples = samples_from(args.samples)
        if args.check:
            usable = ready(cache, samples)
            print("READY" if usable else "NEED_PROVISION")
            return 0 if usable else 4
        cache.mkdir(parents=True, exist_ok=True)
        (cache / MARKER).unlink(missing_ok=True)
        url = os.environ.get("PYPTO_SRC_URL", "https://gitcode.com/cann/pypto.git")
        with tempfile.TemporaryDirectory(prefix=".pro-docs-", dir=cache) as temporary:
            work = Path(temporary)
            source = None if args.pin else local_source()
            mode = "local"
            if source is None:
                source, mode = work / "source", "download"
                download(url, args.pin, source, samples)
            if (cache == source or any(cache.is_relative_to(source / name) for name in ("docs", FRONTEND))
                    or any(source.is_relative_to(cache / name) for name in ("docs", "pro_ops"))):
                raise ValueError("缓存目录不能覆盖源资料目录")
            staged = work / "cache"
            revision = git("rev-parse", "HEAD", cwd=source) if (source / ".git").exists() else "unversioned"
            assemble(source, staged, samples, revision, url)
            manifest = {"kind": "pypto-pro-docs", "layout": MARKER, "mode": mode,
                        "source": str(source) if mode == "local" else url, "revision": revision,
                        "docs": list(DOC_PATHS), "samples": samples}
            (staged / "MANIFEST.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (staged / MARKER).write_text("1\n", encoding="utf-8")
            if not ready(staged, samples):
                raise RuntimeError("Pro 缓存校验失败")
            for name in ("docs", "pro_ops"):
                remove_cached(cache / name)
                (staged / name).rename(cache / name)
            (staged / "MANIFEST.json").replace(cache / "MANIFEST.json")
        (cache / MARKER).write_text("1\n", encoding="utf-8")
        print(f"READY: {cache} ({len(samples)} samples)")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print("Pro 缓存失败：" + str(error), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
