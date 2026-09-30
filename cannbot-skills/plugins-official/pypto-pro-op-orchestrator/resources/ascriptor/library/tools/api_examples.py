# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build examples/api/index.json from the metadata.json each example folder owns.

Every folder is the authority on itself: `metadata.json` is written by hand beside the code it
describes, and this tool only collects them. Nothing here is a second place to edit, and the
index records no validation or development state -- run a folder's `main.py` to find out what
works on a backend.

    python tools/api_examples.py            # write examples/api/index.json
    python tools/api_examples.py --check    # fail if the index is stale, or a folder is malformed

The `--check` gate is what keeps the collection honest, so run it after adding or removing a
case. Beyond staleness it refuses:

  * a folder that is not `kernel(.py|/)`, `reference(.py|/)`, `main.py`, `metadata.json` and
    nothing else -- no runner, no contract, no per-folder README, no recorded evidence;
  * metadata with a missing field, an unknown device or a topology outside the vocabulary;
  * a `reference` that imports `ascriptor`, which would have it check itself, or a `kernel` that
    imports `torch`, which would put host arithmetic where the device code belongs;
  * a `main.py` with no `CASES` list, or a case with no `id`, `seed`, `block_dim`, `parameters`
    or `purpose` -- a case whose reason for existing is not written down is a case nobody can
    tell is redundant.

The case count and every per-case field are read out of `main.py` with the ast module rather
than by importing it, so the index builds on a machine with no ascriptor install. An example
whose CASES is not a literal -- a comprehension, or entries that compute a value -- is the
exception: that one is imported, which does need the install.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/api"
INDEX = BASE / "index.json"

REQUIRED = ("schema", "id", "kind", "title", "surface", "devices", "topology", "tags",
            "study_for", "do_not_copy_when")
PROSE = ("study_for", "do_not_copy_when")
KINDS = ("kernel", "protocol")
DEVICES = ("a2", "a3", "a5", "a5pr")
# A controlled vocabulary, so the index can be filtered on it. Anything a reader needs to know
# beyond the pipe order -- a scalar-only body, an atomic accumulation, a SIMT launch -- belongs
# in `tags` or `study_for`, not here.
TOPOLOGIES = ("vec-only", "cube-only", "simt-only",
              "cube->vec", "vec->cube",
              "cube->vec->cube", "vec->cube->vec",
              "cube->vec->cube->vec", "vec->cube->vec->cube")
CASE_FIELDS = ("id", "seed", "block_dim", "parameters", "purpose")


def members(folder: Path, stem: str) -> list[Path]:
    """The files `stem` stands for: `stem.py`, or every module of a `stem/` package."""
    if (folder / f"{stem}.py").is_file():
        return [folder / f"{stem}.py"]
    if (folder / stem / "__init__.py").is_file():
        return sorted((folder / stem).rglob("*.py"))
    return []


def layout(folder: Path, kind: str) -> tuple[list[str], list[str]]:
    """What the folder is missing, and what it holds that a folder of this kind must not."""
    expected = {"main.py", "metadata.json"} | ({"kernel", "reference"} if kind == "kernel" else set())
    missing = [name for name in sorted(expected)
               if not (folder / name).is_file() and not members(folder, name)]
    allowed = {"main.py", "metadata.json", "kernel.py", "reference.py", "kernel", "reference",
               "tmp", "__pycache__"}
    extra = sorted(p.name for p in folder.iterdir() if p.name not in allowed)
    return missing, extra


def imported_roots(paths: list[Path]) -> set[str]:
    """The top-level package of every module these files import."""
    roots: set[str] = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            roots.update(name.split(".")[0] for name in names if name)
    return roots


def literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def assignment(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return node.value
    return None


def imported_cases(main_py: Path) -> list | None:
    """CASES read by importing main.py, for an example whose case matrix is a comprehension.

    Needs an ascriptor install, which the literal path deliberately does not.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import json, main; print(json.dumps(main.CASES))"],
        cwd=main_py.parent, capture_output=True, text=True)
    if probe.returncode:
        return None
    return json.loads(probe.stdout)


def cases_of(main_py: Path) -> tuple[list[dict], list[str]]:
    """`main.py`'s CASES as data, plus what is wrong with them."""
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    node = assignment(tree, "CASES")
    if node is None:
        return [], ["main.py declares no CASES list"]
    value = literal(node)
    if value is None:
        # Not a literal: a comprehension, or a list whose entries compute something (`2**63 + 1`,
        # `list(range(64))`). Either way the values come from importing it.
        value = imported_cases(main_py)
        if value is None:
            return [], ["main.py's CASES is neither a list literal nor importable "
                        "(an import needs an ascriptor install)"]
    if not isinstance(value, list) or not value:
        return [], ["main.py's CASES is not a non-empty list"]
    problems, ids = [], set()
    for index, case in enumerate(value):
        where = f"CASES[{index}]"
        if not isinstance(case, dict):
            problems.append(f"{where} is not an object")
            continue
        absent = [field for field in CASE_FIELDS if field not in case]
        if absent:
            problems.append(f"{where} ({case.get('id', '?')}) has no {', '.join(absent)}")
            continue
        where = f"case {case['id']}"
        if not isinstance(case["id"], str) or not case["id"]:
            problems.append(f"{where}: id must be a non-empty string")
        elif case["id"] in ids:
            problems.append(f"{where}: duplicate case id")
        ids.add(case["id"])
        if not isinstance(case["purpose"], str) or len(case["purpose"].split()) < 4:
            problems.append(f"{where}: purpose must say what the case is for")
        if isinstance(case["seed"], bool) or not isinstance(case["seed"], int):
            problems.append(f"{where}: seed must be an integer")
        if isinstance(case["block_dim"], bool) or not isinstance(case["block_dim"], int) or case["block_dim"] < 1:
            problems.append(f"{where}: block_dim must be a positive integer")
        if not isinstance(case["parameters"], dict):
            problems.append(f"{where}: parameters must be an object")
    return value, problems


def refusals(main_py: Path) -> list[str]:
    """What `main.py` records a refusal for, as `# <name>:` comment lines.

    A backend that cannot emit the kernel, or a launcher that cannot run a case, is written down
    beside the code it is about rather than in a support matrix somewhere else.
    """
    names = ("pypto_pro", "pto_isa", "cce", "sim", "pipesim", "cannsim", "aclnn", "board", "pypto")
    text = main_py.read_text(encoding="utf-8")
    return [name for name in names
            if any(line.startswith(f"# {name}:") for line in text.splitlines())]


def orphans() -> list[str]:
    """Directories under examples/api that describe no example.

    A folder with no `metadata.json` is invisible to `collect`, so without this an example
    dropped from the collection -- or one never added to it -- would leave no trace in the index
    and no complaint here.
    """
    lines = []
    for folder in sorted(p for p in BASE.rglob("*") if p.is_dir()):
        name, relative = folder.name, folder.relative_to(ROOT).as_posix()
        if name in ("tmp", "__pycache__", "kernel", "reference") or "tmp" in folder.parts:
            continue
        if (folder / "metadata.json").is_file():
            continue
        if any((child / "metadata.json").is_file() for child in folder.iterdir() if child.is_dir()):
            continue  # a parent that only groups examples, as fixpipe_scaled_formats does
        lines.append(f"{relative}: no metadata.json, so nothing here is indexed")
    return lines


def collect() -> tuple[list[dict], list[str], list[str]]:
    entries, problems, structure = [], [], []
    for meta_path in sorted(BASE.rglob("metadata.json")):
        folder = meta_path.parent
        relative = folder.relative_to(ROOT).as_posix()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        absent = [key for key in REQUIRED if key not in meta]
        if absent:
            problems.append(f"{relative}: metadata.json has no {', '.join(absent)}")
            continue
        if meta["kind"] not in KINDS:
            problems.append(f"{relative}: kind {meta['kind']!r} is not one of {', '.join(KINDS)}")
            continue
        missing, extra = layout(folder, meta["kind"])
        if missing:
            problems.append(f"{relative}: missing {', '.join(missing)}")
            continue
        if extra and meta["kind"] != "protocol":
            structure.append(f"{relative}: unexpected {', '.join(extra)} -- an example folder is "
                             f"kernel, reference, main.py and metadata.json, and nothing else")
        if not isinstance(meta["devices"], list) or not meta["devices"]:
            problems.append(f"{relative}: devices must be a non-empty list")
        for device in meta["devices"] if isinstance(meta["devices"], list) else ():
            if device not in DEVICES:
                problems.append(f"{relative}: device {device!r} is not one of {', '.join(DEVICES)}")
        if meta["topology"] not in TOPOLOGIES:
            problems.append(f"{relative}: topology {meta['topology']!r} is not one of "
                            f"{', '.join(TOPOLOGIES)} -- put the rest in tags or study_for")
        for key in PROSE:
            if not isinstance(meta[key], list) or not meta[key]:
                problems.append(f"{relative}: {key} must be a non-empty list")
        if not isinstance(meta["tags"], list) or not meta["tags"]:
            problems.append(f"{relative}: tags must be a non-empty list")
        # The two invariants that make a comparison mean something, checked rather than promised:
        # a reference that called the compiler would be checking itself, and a kernel that
        # reached for torch would not be the thing the reference is checking.
        if meta["kind"] == "kernel":
            if "ascriptor" in imported_roots(members(folder, "reference")):
                problems.append(f"{relative}: reference imports ascriptor; it has to be "
                                f"independent of what it checks")
            if "torch" in imported_roots(members(folder, "kernel")):
                problems.append(f"{relative}: kernel imports torch; host arithmetic belongs in "
                                f"reference or main.py")
        cases, case_problems = cases_of(folder / "main.py")
        problems += [f"{relative}: {line}" for line in case_problems]
        entry = {"path": relative}
        entry.update({key: meta[key] for key in REQUIRED if key != "schema"})
        entry["cases"] = len(cases)
        entry["case_ids"] = [case["id"] for case in cases if isinstance(case, dict) and "id" in case]
        entry["refusals"] = refusals(folder / "main.py")
        # A protocol example's subject can be a file the four-file shape has no slot for -- a plugin
        # module, the packaging metadata that publishes its entry point. Those are recorded here
        # rather than warned about, so the index says what the folder carries.
        if extra and meta["kind"] == "protocol":
            entry["carries"] = extra
        entries.append(entry)
    ids = [entry["id"] for entry in entries]
    for value in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate id {value!r}")
    return entries, problems, structure


def build(strict: bool = False) -> dict:
    entries, problems, structure = collect()
    # A metadata or case defect is always fatal. A folder that still holds files the protocol
    # owned, or a directory that describes no example, is only fatal at the --check gate: during a
    # migration both are work still in progress.
    pending = structure + orphans()
    if problems or (strict and pending):
        raise SystemExit("index refused:\n  " + "\n  ".join(problems + (pending if strict else [])))
    for line in pending:
        print(f"warning: {line}")
    return {"schema": "ascriptor.api-example-index/1",
            "generated_by": "tools/api_examples.py",
            "scope": "Navigation only. Each example folder's metadata.json is the source; this "
                     "file is generated and is never edited by hand. It records no validation, "
                     "backend support or development state -- run a folder's main.py on the "
                     "launcher you care about to find out what that source does today.",
            "counts": {"examples": len(entries), "cases": sum(e["cases"] for e in entries),
                       "devices": sorted({d for e in entries for d in e["devices"]})},
            "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the index is stale, and write nothing")
    args = parser.parse_args()
    index = build(strict=args.check)
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.is_file() else ""
        if current != text:
            print("examples/api/index.json is stale; run python tools/api_examples.py")
            return 1
    else:
        INDEX.write_text(text, encoding="utf-8")
    print(json.dumps({**index["counts"], "checked": args.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
