# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build index.json from the metadata.json each demo folder owns.

Every demo folder is the authority on itself: `metadata.json` is written by hand beside the
code it describes, and this tool only collects them. Nothing here is a second place to edit.

    python tools/build_index.py            # write index.json
    python tools/build_index.py --check    # fail if index.json is not what this would write

The case count is read out of `main.py`'s `CASES` literal with the ast module rather than by
importing it, so the index can be built on a machine with no ascriptor install.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GALLERIES = ("ascriptor_kernels", "pypto_pro_kernels")
REQUIRED = ("schema", "id", "title", "formula", "device", "topology", "tags",
            "study_for", "do_not_copy_when")
FILES = ("kernel.py", "reference.py", "main.py", "metadata.json")
DEVICES = ("a2", "a3", "a5", "a5pr")
# A controlled vocabulary, so the index can be filtered on it. Anything a reader needs to know
# beyond the pipe order — an atomic accumulation, an idle cube, a SIMT launch — belongs in
# `tags` or `study_for`, not here.
TOPOLOGIES = ("vec-only", "cube-only", "simt-only",
              "cube->vec", "vec->cube",
              "cube->vec->cube", "vec->cube->vec",
              "cube->vec->cube->vec", "vec->cube->vec->cube")


def case_count(main_py):
    """The number of entries in main.py's CASES.

    A list literal is counted from the syntax tree, so the common case needs no ascriptor
    install. A demo whose case matrix is regular enough to be written as a comprehension
    (mixed_pipeline's 115) is counted by importing it, which does.
    """
    for node in ast.parse(main_py.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets):
            if isinstance(node.value, ast.List):
                return len(node.value.elts)
            return _imported_case_count(main_py)
    raise ValueError(f"{main_py}: no CASES list")


def _imported_case_count(main_py):
    import subprocess

    probe = subprocess.run(
        [sys.executable, "-c", "import main; print(len(main.CASES))"],
        cwd=main_py.parent, capture_output=True, text=True)
    if probe.returncode != 0:
        raise ValueError(f"{main_py}: CASES is not a list literal and importing it failed "
                         f"(needs an ascriptor install):\n{probe.stderr.strip()[-400:]}")
    return int(probe.stdout.strip())


def imported_roots(path):
    """The top-level package of every module a file imports."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        roots.update(n.split(".")[0] for n in names if n)
    return roots


def pypto_note(main_py):
    """True when main.py carries a `# pypto_pro:` comment explaining a backend refusal."""
    return any(line.startswith("# pypto_pro:")
               for line in main_py.read_text(encoding="utf-8").splitlines())


def collect():
    entries, problems, layout = [], [], []
    for gallery in GALLERIES:
        base = ROOT / gallery
        if not base.is_dir():
            continue
        for meta_path in sorted(base.rglob("metadata.json")):
            folder = meta_path.parent
            relative = folder.relative_to(ROOT).as_posix()
            missing = [name for name in FILES if not (folder / name).is_file()]
            extra = sorted(p.name for p in folder.iterdir()
                           if p.name not in FILES and p.name != "tmp" and p.name != "__pycache__")
            if missing:
                layout.append(f"{relative}: missing {', '.join(missing)}")
                continue
            if extra:
                layout.append(f"{relative}: unexpected {', '.join(extra)} — a demo folder is "
                              f"exactly {', '.join(FILES)}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            absent = [key for key in REQUIRED if key not in meta]
            if absent:
                problems.append(f"{relative}: metadata.json missing {', '.join(absent)}")
                continue
            if meta["device"] not in DEVICES:
                problems.append(f"{relative}: device {meta['device']!r} is not one of "
                                f"{', '.join(DEVICES)}")
            if meta["topology"] not in TOPOLOGIES:
                problems.append(f"{relative}: topology {meta['topology']!r} is not one of "
                                f"{', '.join(TOPOLOGIES)} — put the rest in tags or study_for")
            for key in ("study_for", "do_not_copy_when"):
                if not isinstance(meta[key], list) or not meta[key]:
                    problems.append(f"{relative}: {key} must be a non-empty list")
            entry = {"path": relative, "gallery": gallery}
            entry.update({key: meta[key] for key in REQUIRED if key != "schema"})
            # The two invariants that make a comparison mean something, checked rather than
            # promised: a reference that called the compiler would be checking itself, and a
            # kernel that reached for torch would not be the thing the reference is checking.
            if "ascriptor" in imported_roots(folder / "reference.py"):
                problems.append(f"{relative}: reference.py imports ascriptor; the reference has "
                                f"to be independent of what it checks")
            if "torch" in imported_roots(folder / "kernel.py"):
                problems.append(f"{relative}: kernel.py imports torch; host arithmetic belongs "
                                f"in reference.py or main.py")
            entry["cases"] = case_count(folder / "main.py")
            entry["pypto_pro_note"] = pypto_note(folder / "main.py")
            entries.append(entry)
    ids = [e["id"] for e in entries]
    for value in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate id {value!r}")
    return entries, problems, layout


def build(strict=False):
    entries, problems, layout = collect()
    # A metadata defect is always fatal. A folder that still has its old protocol files beside
    # the new ones is only fatal at the --check gate: during a migration it is work in progress.
    if problems or (strict and layout):
        raise SystemExit("index refused:\n  " + "\n  ".join(problems + (layout if strict else [])))
    for line in layout:
        print(f"warning: {line}", file=sys.stderr)
    devices = sorted({e["device"] for e in entries})
    return {"schema": "ascriptor.kernel-demo-index/1",
            "generated_by": "tools/build_index.py",
            "scope": "Navigation only. Each demo folder's metadata.json is the source; this "
                     "file is generated and is never edited by hand. It records no validation "
                     "or development state — run a folder's main.py to find out what works.",
            "counts": {"demos": len(entries), "cases": sum(e["cases"] for e in entries),
                       "devices": devices},
            "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if index.json is stale, and write nothing")
    args = parser.parse_args()
    index = build(strict=args.check)
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    target = ROOT / "index.json"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current != text:
            print("index.json is stale; run python tools/build_index.py", file=sys.stderr)
            return 1
    else:
        target.write_text(text, encoding="utf-8")
    print(json.dumps({**index["counts"], "checked": args.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
