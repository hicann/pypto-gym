# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join the checked-in API examples and kernel gallery into a navigation index."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Navigation-only fields an owner's index brings with it. `cases` is a count, not a list: which
# cases exist is `python main.py --list` in the folder, and printing them here would be a second
# copy to keep in step. The two owners name their searchable API text differently -- a demo has a
# `formula`, an API example a `surface` -- and each keeps its own word.
DEMO_FIELDS = ("formula", "topology", "tags", "cases", "pypto_pro_note")
EXAMPLE_FIELDS = ("surface", "topology", "tags", "cases", "refusals")
SCOPE = ('Navigation only. Run each demo on the selected device to establish a current result. '
         'The index does not certify hardware or performance.')


def build_index(library_root, kernels_root):
    gallery = json.loads((kernels_root / "index.json").read_text())
    api = json.loads((library_root / "examples/api/index.json").read_text())
    pyproject = (library_root / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$', pyproject, re.M)
    if match is None:
        raise ValueError("library/pyproject.toml has no source version")
    declared = sorted({device for entry in api["entries"] for device in entry["devices"]}
                      | {entry["device"] for entry in gallery["entries"]})
    data = {"schema": "ascriptor.agent-kernel-index/1",
            "release": {"version": match.group(1), "declared_hardware": declared,
                        "qualification": "workload-specific"},
            "source_roots": {"kernels": "../kernels", "library": "../library"},
            "scope": SCOPE,
            "counts": {"kernel_demos": gallery["counts"]["demos"],
                       "api_examples": api["counts"]["examples"]},
            # The gallery's declared families limit navigation, not qualification.
            "deferred": {"handoff": "../library/docs/api/README.md"}}

    candidates = []
    for entry in api["entries"]:
        folder = library_root / entry["path"]
        if not (folder / "main.py").is_file():
            raise ValueError(f"Missing navigation source: library/{entry['path']}/main.py")
        candidates.append({"id": entry["id"], "owner": "library",
                           "source": f"library/{entry['path']}",
                           "description": entry["title"],
                           "guide": "references/patterns.md",
                           "devices": entry["devices"],
                           **{name: entry[name] for name in EXAMPLE_FIELDS if entry.get(name)},
                           "study_for": entry["study_for"],
                           "do_not_copy_when": entry["do_not_copy_when"]})

    for entry in gallery["entries"]:
        folder = kernels_root / entry["path"]
        if not (folder / "main.py").is_file():
            raise ValueError(f"Missing navigation source: kernels/{entry['path']}/main.py")
        candidates.append({"id": entry["id"], "owner": "kernels",
                           "source": f"kernels/{entry['path']}",
                           "description": entry["title"],
                           "guide": "references/patterns.md",
                           "devices": [entry["device"]],
                           **{name: entry[name] for name in DEMO_FIELDS if name in entry},
                           "study_for": entry["study_for"],
                           "do_not_copy_when": entry["do_not_copy_when"]})

    for key in ("id", "source"):
        if len({row[key] for row in candidates}) != len(candidates):
            raise ValueError(f"Duplicate navigation {key}")
    # Retained for readers of this file: no owner declares material outside the release scope today,
    # and `select_example` still distinguishes the case.
    data.update(candidates=candidates, deferred_candidates=[])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--kernels-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build_index(args.library_root, args.kernels_root)
    path = ROOT / "index/kernels.json"
    content = json.dumps(data, indent=2) + "\n"
    if args.check:
        if not path.is_file() or path.read_text() != content:
            raise ValueError("Agent navigation index is stale; rerun tools/build_index.py")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps({**data["counts"], "candidates": len(data["candidates"]), "checked": args.check}))


if __name__ == "__main__":
    main()
