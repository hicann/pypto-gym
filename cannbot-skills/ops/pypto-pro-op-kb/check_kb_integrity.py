#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Knowledge-base integrity checks.

Run from anywhere::

    python3 cannbot-skills/ops/pypto-pro-op-kb/check_kb_integrity.py

Exits non-zero and prints every violation.  Deliberately dependency-free (no
torch, no pytest) so it runs in CI and on a laptop.

What it enforces, and why each rule exists:

* **Reachability** -- every retained file must be reachable from ROUTER.md or a
  maintained index.  An unreferenced file is one no stage will ever select, and
  is how raw experiment output accumulates in a knowledge base.
* **Link resolution** -- a broken link is a reference that silently does nothing.
* **Filename hygiene** -- no run ids, dates, scores, or status words.  A
  knowledge base names *what a thing is*, not which experiment produced it.
* **Portable paths** -- no machine-specific absolute paths, so the KB works on
  any checkout.
* **Topology map** -- every path it routes to must exist, and patterns and
  constraints must stay in their declared namespaces.
* **Sample provenance headers** -- a retained sample must not assert a
  validation result that a generated copy would inherit unearned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = KB_ROOT.parents[2]

#: Entry points a reader can actually reach. Everything else must be linked
#: from one of these, directly or transitively.
ROOTS = ("ROUTER.md", "README.md")

#: Indexes are roots for the directories they curate.
INDEXES = ("patterns/pattern-index.md", "examples/kernel-index.md")

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)#]+?)(?:#[^)]*)?\)")

#: Transient labels that must never appear in a retained filename.
_BAD_NAME_TOKENS = (
    "run_2",
    "wip",
    "tmp",
    "temp",
    "draft",
    "final",
    "old",
    "new",
    "backup",
    "todo",
    "fixme",
    "status",
    "report_2",
    "result_2",
    "score",
    "v1",
    "v2",
    "copy",
    "20250",
    "20260",
)

#: Absolute paths that point at somebody's machine rather than a repository.
#: ``/tmp`` and the other shared roots belong here too: a sample that writes to a
#: global directory is modelling behaviour the agent rules forbid, and twelve of them
#: did exactly that while this pattern only looked for home directories.
#: ``/usr`` and ``/opt`` are deliberately absent: ``#!/usr/bin/env python3`` is portable,
#: and flagging it would bury the real findings in shebangs.
_ABS_PATH = re.compile(
    r"(?<![\w.])/(?:home|data|root|Users|mnt|tmp|var/tmp|dev/shm|workspace)"
    r"(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)?"
)

#: A retained sample must not delete a directory tree it did not create.
_DESTRUCTIVE = re.compile(r"shutil\.rmtree|rm\s+-rf")

#: A line may opt out of the path check, but only in the open and with a stated reason.
#: Two cases genuinely need it: prose that quotes the paths it is banning, and a
#: cross-process lock, which stops being a lock if each run puts it somewhere else.
_ALLOW_PATH = re.compile(r"kb-integrity:\s*allow-path\s*\(([^)]+)\)")


def _path_exempt(line: str) -> bool:
    return bool(_ALLOW_PATH.search(line))


#: Claims a sample must not make about itself: a generated kernel that copies
#: the header would inherit an unearned validation record.
_PROVENANCE_CLAIM = re.compile(r"STATUS:\s*VALIDATED|max_abs_diff\s*=|\bPASS\b\.\s|verified on .* NPU", re.IGNORECASE)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _kb_files() -> list[Path]:
    return sorted(
        p for p in KB_ROOT.rglob("*") if p.is_file() and p.name != Path(__file__).name and "__pycache__" not in p.parts
    )


def _absolute_path_errors(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        for lineno, line in enumerate(_text(path).splitlines(), 1):
            if _path_exempt(line):
                continue
            match = _ABS_PATH.search(line)
            if match:
                errors.append(
                    f"machine-specific absolute path {match.group(0)!r} at {path.relative_to(REPO_ROOT)}:{lineno}"
                )
    return errors


def _links_from(path: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for target in _MD_LINK.findall(_text(path)):
        target = target.strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append((target, (path.parent / target)))
    return out


def check_links() -> list[str]:
    """Resolve relative links, KB-internal only.

    Cross-package links are checked too, by `check_cross_package_links` -- not here.

    This docstring used to say they must **not** be checked, on the reasoning that a skill
    reaches the KB with ``../../pypto-pro-op-kb/`` once installed but ``../pypto-pro-op-kb/``
    in a checkout, so a repo-relative scan would report ~35 correct links as broken. That
    was wrong on the facts: skills install as **symlinks** into ``cannbot-skills/ops/``, so
    a ``..`` traversal out of an installed skill resolves physically back to the KB's
    siblings -- the same place a checkout puts it. The sibling form is correct under both
    resolvers, and every link the widened scan reported was genuinely broken. See
    `CONTRACT.md`.

    KB-internal links are safe to check for the separate reason that the directory moves as
    a unit, so relative paths between its own pages hold in either layout.
    """
    errors: list[str] = []
    for path in _kb_files():
        if path.suffix != ".md":
            continue
        for raw, resolved in _links_from(path):
            if not resolved.exists():
                rel = path.relative_to(REPO_ROOT)
                errors.append(f"broken link in {rel}: {raw}")
    return errors


def check_reachability() -> list[str]:
    """Every retained file must be reachable from a maintained entry point."""
    seen: set[Path] = set()
    queue = [KB_ROOT / name for name in (*ROOTS, *INDEXES) if (KB_ROOT / name).is_file()]
    seen.update(queue)

    while queue:
        current = queue.pop()
        if current.suffix != ".md":
            continue
        for _, resolved in _links_from(current):
            try:
                resolved = resolved.resolve()
            except OSError:
                continue
            if resolved.is_file() and resolved not in seen and KB_ROOT in resolved.parents:
                seen.add(resolved)
                queue.append(resolved)

    # topology-map.json is machine-read, and everything it routes to is reachable.
    topo = KB_ROOT / "topology-map.json"
    if topo.is_file():
        seen.add(topo)
        for rel in _topology_paths(topo):
            candidate = KB_ROOT / rel
            if candidate.is_file():
                seen.add(candidate.resolve())

    errors: list[str] = []
    for path in _kb_files():
        if path.resolve() in seen or path.name in (*ROOTS,):
            continue
        # Directory READMEs are reachable by convention; samples are reached
        # through the kernel index entry that names their directory.
        if path.name == "README.md":
            continue
        if "samples" in path.parts and _sample_is_indexed(path):
            continue
        errors.append(f"unreachable KB file (no router, index or reference links to it): {path.relative_to(REPO_ROOT)}")
    return errors


def _sample_is_indexed(path: Path) -> bool:
    index = KB_ROOT / "examples" / "kernel-index.md"
    if not index.is_file():
        return False
    body = _text(index)
    # A sample counts as reached when the index names its directory or the file.
    return path.parent.name in body or path.name in body


def _topology_paths(topo: Path) -> list[str]:
    try:
        data = json.loads(_text(topo))
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for entry in (data.get("topologies") or {}).values():
        out.extend(entry.get("patterns") or [])
        out.extend(entry.get("constraints") or [])
    for entry in (data.get("property_modifiers") or {}).values():
        if isinstance(entry, dict):
            out.extend(entry.get("patterns") or [])
            out.extend(entry.get("constraints") or [])
    out.extend((data.get("target_gated") or {}).keys())
    out.extend((data.get("mandatory_constraints") or {}).keys())
    return out


def _validate_topology_entry(name: str, entry: dict, errors: list[str]) -> None:
    if not str(entry.get("note") or "").strip():
        errors.append(f"topology {name!r} has no note explaining what the computation is")
    for rel in entry.get("patterns") or []:
        if not rel.startswith("patterns/"):
            errors.append(f"topology {name!r} classifies non-pattern as a pattern: {rel}")
    for rel in entry.get("constraints") or []:
        if not rel.startswith("constraints/"):
            errors.append(f"topology {name!r} classifies non-constraint as a constraint: {rel}")


def _validate_property_modifier(prop: str, entry, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"property modifier {prop!r} must separate patterns and constraints")
        return
    for rel in entry.get("patterns") or []:
        if not rel.startswith("patterns/"):
            errors.append(f"property modifier {prop!r} classifies non-pattern as a pattern: {rel}")
    for rel in entry.get("constraints") or []:
        if not rel.startswith("constraints/"):
            errors.append(f"property modifier {prop!r} classifies non-constraint as a constraint: {rel}")


def _validate_mandatory_constraints(data: dict, errors: list[str]) -> None:
    for rel, policy in (data.get("mandatory_constraints") or {}).items():
        if not rel.startswith("constraints/"):
            errors.append(f"mandatory constraint is outside constraints/: {rel}")
        roles = policy.get("applies_to") if isinstance(policy, dict) else None
        if not roles:
            errors.append(f"mandatory constraint {rel!r} has no applies_to roles")


def check_topology_map() -> list[str]:
    topo = KB_ROOT / "topology-map.json"
    if not topo.is_file():
        return ["topology-map.json is missing; the router has no machine-readable map"]
    try:
        data = json.loads(_text(topo))
    except json.JSONDecodeError as exc:
        return [f"topology-map.json is not valid JSON: {exc}"]

    errors: list[str] = []
    for rel in _topology_paths(topo):
        if rel.startswith("/"):
            errors.append(f"topology-map.json routes to an absolute path: {rel}")
        elif not (KB_ROOT / rel).is_file():
            errors.append(f"topology-map.json routes to a missing file: {rel}")

    for name, entry in (data.get("topologies") or {}).items():
        _validate_topology_entry(name, entry, errors)

    for prop, entry in (data.get("property_modifiers") or {}).items():
        _validate_property_modifier(prop, entry, errors)

    _validate_mandatory_constraints(data, errors)
    return errors


def check_filenames() -> list[str]:
    """Reject transient labels, matching whole name segments only.

    Substring matching is wrong here: ``golden`` contains ``old`` and
    ``reduce_sum`` contains ``sum``. Names are split on their separators and
    each segment compared as a unit.
    """
    errors: list[str] = []
    for path in _kb_files():
        segments = {seg for seg in re.split(r"[-_.\s]+", path.stem.lower()) if seg}
        hit = segments & set(_BAD_NAME_TOKENS)
        # Date- and run-shaped segments are rejected by pattern, not by list.
        hit |= {seg for seg in segments if re.fullmatch(r"(19|20)\d{6}|run\d*|v\d+", seg)}
        if hit:
            errors.append(f"filename carries a transient label {sorted(hit)}: {path.relative_to(REPO_ROOT)}")
    return errors


def check_absolute_paths() -> list[str]:
    return _absolute_path_errors(_kb_files())


# --- shared by both index gates -------------------------------------------------------
# The same word `validated` is claimed in two indexes, and every bypass found in one had to
# be re-found by hand in the other. These two helpers are the part that genuinely is one
# rule; the checks themselves stay separate because they verify different things (one that
# the artifact self-documents its validation, one that the citation resolves).

_VALIDATED_CELL = re.compile(r"\bvalidated\b")
_DISCLAIMED = re.compile(r"\b(?:not\s+validated|unvalidated|never\s+validated)\b")


def _claims_validated(cell: str) -> bool:
    """True when a Validation cell claims validation for the row as a whole.

    A raw substring scan was wrong twice over: `"validated" not in v` skipped nothing, and
    `"unvalidated" in v` exempted the WHOLE row on one word -- while "validated skeleton for
    A; conceptual for B" is the norm here, and one page's own prose says "treat anything
    marked there as unvalidated as unvalidated". A row that claims validation anywhere is
    checked; a row that only disclaims it is not.
    """
    text = cell.lower()
    without_disclaimers = _DISCLAIMED.sub("", text)
    return bool(_VALIDATED_CELL.search(without_disclaimers))


def _table_rows(text: str):
    """Yield ``(lineno, cells)`` for GFM table rows.

    Accept variable column counts and an omitted leading pipe so validation remains
    independent of presentation-only table formatting changes.
    """
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if "|" not in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2 or all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        yield lineno, cells


def check_sample_provenance_claims() -> list[str]:
    """A sample's validation record must be scoped to the sample.

    A validation record documents what was verified, on which hardware and to what
    accuracy. Its wording must scope the claim to the validated sample so consumers
    cannot mistake it for evidence about generated code.
    """
    errors: list[str] = []
    for path in _kb_files():
        if path.suffix != ".py":
            continue
        head = _text(path).splitlines()[:40]
        claim_line = next((i for i, line in enumerate(head, 1) if _PROVENANCE_CLAIM.search(line)), None)
        if claim_line is None:
            continue
        if not any("SAMPLE PROVENANCE" in line for line in head):
            errors.append(
                f"validation record is not scoped to the sample (missing the "
                f"'SAMPLE PROVENANCE' marker a copy would carry with it), at "
                f"{path.relative_to(REPO_ROOT)}:{claim_line}"
            )
    return errors


def _ref_exists(ref: str) -> bool:
    """True when `ref` is present locally. A shallow or --single-branch clone has none."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                cwd=REPO_ROOT,
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _git_has(ref: str, path: str) -> bool:
    """True when `path` exists in `ref`.

    Callers must check `_ref_exists(ref)` first. Collapsing "the branch is not fetched"
    into "the path is not there" made the gate assert a fact it never established: in a
    `--single-branch` clone -- actions/checkout's default -- every branch-qualified citation
    reported "resolves nowhere", naming a path that is in fact present on that branch.
    """
    try:
        return (
            subprocess.run(
                ["git", "cat-file", "-e", f"{ref}:{path}"], cwd=REPO_ROOT, capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _expand(citation: str) -> list[str]:
    """`custom/gather/{A.md,B.py}` -> each named file. NOT the parent directory.

    Returning the directory as well made the brace form unfalsifiable: `git cat-file -e
    <ref>:custom/gather` exits 0 for a tree, and the resolvers take `any(...)` over the
    parts, so a page could cite `custom/gather/{NEVER_EXISTED.md}` and still pass. Four of
    the six live citations are directories in their own right, which is fine -- a page may
    cite a directory. What is not fine is a citation that NAMES files and is satisfied by
    their parent.
    """
    head, _, rest = citation.partition("{")
    if not rest:
        return [citation]
    inner = rest.rstrip("}")
    named = [head + part.strip() for part in inner.split(",") if part.strip()]
    return named or [head.rstrip("/")]


_PATTERN_CELL_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PATTERN_ARTIFACT_CITE = re.compile(r"`((?:custom|tools)/[\w./{},-]+)`")
_PATTERN_EXAMPLE_LINK = re.compile(r"\((\.\./examples/[\w./-]+)\)")
_PATTERN_BRANCH = re.compile(r"on\s+branch\s+`([^`]+)`")


def _citation_resolution_result(
    lineno: int,
    name: str,
    page: Path,
    citation: str,
    refs: tuple[list[str], list[str]],
) -> str | None:
    present, absent = refs
    parts = _expand(citation)
    # `all`, not `any`. A brace citation names several files and is evidence only for
    # what it names; under `any` one present member vouched for every absent one. This
    # was fixed in `check_reference_artifact_citations` and missed here, so the hole
    # stayed open on the path that guards `validated` rows -- the stronger claim.
    on_disk = all(((page.parent / part) if part.startswith("..") else (REPO_ROOT / part)).exists() for part in parts)
    if on_disk or (present and all(any(_git_has(ref, part) for ref in present) for part in parts)):
        return None
    if absent and not present:
        # Cannot be settled locally: a named ref may be unfetched (shallow clone) or may
        # never have existed. Returning None passed both, so a fabricated branch name
        # plus a path that exists nowhere produced a green run. Report it as SKIP --
        # inconclusive, not verified -- which `main` prints as SKIP rather than ok.
        return (
            f"SKIP: pattern-index.md:{lineno} ({name}) cites {citation!r} against ref(s) {absent} that "
            f"are not present locally; fetch them to verify, or the citation is unchecked"
        )
    hint = (
        f"; it names branch(es) {present} but the path is not in them"
        if present
        else "; name the branch that retains it with `on branch <name>`"
    )
    return (
        f"pattern-index.md:{lineno} claims validation for {name} but its citation {citation!r} resolves nowhere{hint}"
    )


def check_pattern_validated_claims_cite_evidence() -> list[str]:
    """Check locally verifiable implementation citations for `validated` rows.

    `pattern-index.md` requires a runnable implementation and a scope-matching passing
    result. This check verifies implementation citations on disk or in fetched refs.
    If a citation is absent on disk, names one or more refs and none is fetched, it
    reports `SKIP`. Reviewers match the retained result to the claimed scope.

    This repo splits the knowledge base from the operator trees that validate it, so an
    artifact may legitimately live on another branch. That is resolved by **looking**:
    `git cat-file -e <branch>:<path>`. An earlier version of this function accepted the
    prose "on branch `X`" instead, without checking X -- a made-up branch name passed, and
    the escape hatch removed the only thing the rule did. Three further bypasses came with
    it: an empty citation set satisfied `for ... else`, a `break` on the first resolving
    citation hid every dangling one behind it, and the keyword was searched body-wide so an
    unrelated sentence laundered a broken path.

    Every locally verifiable citation must resolve. Each failure is reported separately;
    citations against only unfetched refs are inconclusive, not passing.
    """
    index = KB_ROOT / "patterns" / "pattern-index.md"
    if not index.is_file():
        return []
    # Match any table row, then look for a link inside the first cell. Keying the whole
    # check on `| [name](href) |` meant a row written `| bogus.md | ... | validated |` was
    # skipped in silence -- the gate answered the row's FORMATTING, not its claim.
    errors: list[str] = []
    for lineno, cells in _table_rows(_text(index)):
        cell = cells[0]
        if not any(_claims_validated(c) for c in cells[1:]):
            continue
        cell_link = _PATTERN_CELL_LINK.search(cell)
        if not cell_link:
            errors.append(
                f"pattern-index.md:{lineno} claims validation but its Pattern cell "
                f"{cell.strip()!r} is not a link, so the page it refers to cannot be checked"
            )
            continue
        name, href = cell_link.groups()
        page = (KB_ROOT / "patterns" / href).resolve()
        if not page.is_file():
            errors.append(f"pattern-index.md:{lineno} points at a missing page: {href}")
            continue
        body = _text(page)
        citations = sorted(set(_PATTERN_ARTIFACT_CITE.findall(body)) | set(_PATTERN_EXAMPLE_LINK.findall(body)))
        if not citations:
            errors.append(
                f"pattern-index.md:{lineno} claims validation for {name} but the page cites "
                f"no artifact at all; cite one or mark it conceptual only"
            )
            continue
        refs = sorted(set(_PATTERN_BRANCH.findall(body)))
        present = [ref for ref in refs if _ref_exists(ref)]
        absent = [ref for ref in refs if ref not in present]
        for citation in citations:
            result = _citation_resolution_result(lineno, name, page, citation, (present, absent))
            if result:
                errors.append(result)
    return errors


def check_every_page_is_routable() -> list[str]:
    """Every pattern/constraint page must be reachable by ROUTING, not only by prose.

    ``check_topology_map`` validates routed references, but the reverse relationship
    must also hold. A page that is linked from an index but absent from routing cannot
    be selected for a matching class; discoverability alone does not make it actionable.
    """
    path = KB_ROOT / "topology-map.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(_text(path))
    except json.JSONDecodeError as exc:
        return [f"topology-map.json is not valid JSON: {exc}"]

    routed: set[str] = set()

    def collect(node):
        if isinstance(node, list):
            routed.update(x for x in node if isinstance(x, str))
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in ("patterns", "constraints", "references"):
                    collect(value)
                elif isinstance(value, (dict, list)):
                    collect(value)

    for section in ("topologies", "property_modifiers"):
        collect(data.get(section) or {})
    routed.update(data.get("target_gated") or {})
    routed.update(data.get("mandatory_constraints") or {})

    errors: list[str] = []
    for sub in ("patterns", "constraints"):
        folder = KB_ROOT / sub
        if not folder.is_dir():
            continue
        for page in sorted(folder.glob("*.md")):
            if page.name in ("README.md", "pattern-index.md"):
                continue  # indexes, not routable content
            rel = f"{sub}/{page.name}"
            if rel not in routed:
                errors.append(
                    f"{rel} is reachable from an index but routed by no topology, property "
                    f"modifier, target gate or mandatory reference -- a class matching its "
                    f"shape cannot select it; add it to topology-map.json or delete it"
                )
    return errors


def check_validated_rows_have_records() -> list[str]:
    """A row may claim validation only if its file records one.

    A validated label requires a validation record in the referenced file. Without that
    evidence, the row is a ``study`` entry and must not imply validated status.
    """
    index = KB_ROOT / "examples" / "kernel-index.md"
    if not index.is_file():
        return []
    errors: list[str] = []
    for lineno, cells in _table_rows(_text(index)):
        if not any(_claims_validated(c) for c in cells[1:]):
            continue
        joined = " | ".join(cells)
        # A link, or a bare backticked path -- both are how rows cite a sample here, and
        # keying on the link form alone let `| bogus | ... | validated |` through in silence.
        match = re.search(r"\]\((samples/[^)]+\.py)\)", joined) or re.search(r"`(samples/[^`]+\.py)`", joined)
        if not match:
            errors.append(
                f"kernel-index.md:{lineno} claims 'validated' but cites no sample path; cite one or mark it 'study'"
            )
            continue
        target = KB_ROOT / "examples" / match.group(1)
        if not target.is_file():
            errors.append(f"kernel-index.md:{lineno} points at a missing file: {match.group(1)}")
            continue
        head = "\n".join(_text(target).splitlines()[:40])
        if not _PROVENANCE_CLAIM.search(head):
            errors.append(
                f"kernel-index.md:{lineno} claims 'validated' but {match.group(1)} records no "
                f"validation result; mark it 'study' or retain the record"
            )
    return errors


_VALIDATION_HASH_FIELD = "VALIDATED-CODE-SHA256"
_VALIDATION_HASH_RE = re.compile(
    rf"^#\s*{_VALIDATION_HASH_FIELD}:\s*([0-9a-f]{{64}})\s*$", re.MULTILINE
)


def _code_fingerprint(path: Path) -> str | None:
    """SHA-256 over a sample's code lines, ignoring whole-line comments and blank lines.

    Deliberately **not** tokenize-based. An earlier version hashed the token stream and
    claimed version stability; it had neither. ``token.type`` is a numeric id that shifts
    between releases (``OP`` is 54 on 3.9-3.11 and 55 on 3.13), and PEP 701 re-tokenized
    f-strings in 3.12 into FSTRING_START/MIDDLE/END where earlier versions emit one STRING.
    Six of the stamped samples contain f-strings, so hashes stamped on one interpreter
    failed on another and told the reader to re-validate kernels nobody had touched --
    on a repo whose declared floor is Python 3.9.

    Line normalisation has no such dependency. Whole-line comments and blank lines are
    dropped so header and provenance edits (including the stamp line itself, which starts
    with ``#``) do not invalidate a record, and trailing whitespace is stripped. A trailing
    comment on a code line *is* part of that line and will invalidate: distinguishing it
    would require tokenizing, which is what this avoids. Over-invalidating is the safe
    direction -- it asks a human to re-confirm, where under-invalidating ships a stale
    record silently.

    Returns None when the file cannot be read; callers report that rather than treating an
    unreadable sample as matching.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = []
    for raw in source.splitlines():
        stripped = raw.rstrip()
        if stripped.strip() and not stripped.lstrip().startswith("#"):
            lines.append(stripped)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _validated_sample_paths() -> list[Path]:
    """Every sample that kernel-index.md marks `validated`, resolved to a real file."""
    index = KB_ROOT / "examples" / "kernel-index.md"
    if not index.is_file():
        return []
    found: list[Path] = []
    for _lineno, cells in _table_rows(_text(index)):
        if not any(_claims_validated(c) for c in cells[1:]):
            continue
        joined = " | ".join(cells)
        # findall, not search: rows that cite an impl *and* its golden (kernel-index rows
        # 24 and 37 both do) left the second file unpinned, so a golden could be rewritten
        # under an unchanged `validated` row and the check still reported ok.
        cited = re.findall(r"\]\((samples/[^)]+\.py)\)", joined) + re.findall(
            r"`(samples/[^`]+\.py)`", joined
        )
        for rel in cited:
            target = KB_ROOT / "examples" / rel
            if target.is_file() and target not in found:
                found.append(target)
    return found


def check_validation_records_match_code() -> list[str]:
    """A validation record must be pinned to the code it was measured against.

    Without this, a sample could be rewritten under an unchanged ``STATUS: VALIDATED``
    header and every check still passed -- which is how an ABI rewrite ended up shipping
    under a validation record measured weeks earlier. The stamped hash makes any later
    code edit fail this check until the sample is re-validated and re-stamped with
    ``--stamp-validation-hashes``.

    The stamp pins content to the *recorded* result; it is not itself evidence that the
    result is current. If the record's toolchain, target or device no longer match the
    environment in use, the sample still needs re-validating.
    """
    samples = _validated_sample_paths()
    if not samples:
        # No validated sample resolved -- kernel-index.md moved, every row downgraded, or
        # the citation spelling drifted past the regexes above. Reporting `ok` here would
        # make a gate that examined nothing look green, which is the failure this whole
        # mechanism exists to prevent. main() renders SKIP as inconclusive.
        return ["SKIP: no sample in kernel-index.md resolved to a validated file to check"]
    errors: list[str] = []
    for sample in samples:
        rel = sample.relative_to(REPO_ROOT)
        text = _text(sample)
        recorded = _VALIDATION_HASH_RE.search(text)
        if not recorded:
            errors.append(
                f"{rel} claims validation but records no {_VALIDATION_HASH_FIELD}; "
                f"re-run with --stamp-validation-hashes after confirming the result"
            )
            continue
        actual = _code_fingerprint(sample)
        if actual is None:
            errors.append(f"{rel} could not be tokenized, so its validation record cannot be checked")
            continue
        if actual != recorded.group(1):
            errors.append(
                f"{rel} code changed since it was validated "
                f"(recorded {recorded.group(1)[:12]}..., actual {actual[:12]}...); "
                f"re-validate and re-stamp, or mark the row 'study'"
            )
    return errors


def stamp_validation_hashes() -> int:
    """Write the current code fingerprint into every validated sample's header.

    Run this only after a sample has actually been re-validated -- stamping is what
    records "this exact code is what produced the result above", so stamping unverified
    code launders it into looking verified.

    One narrower case is permitted and must be **written into the sample's header**: a
    text-only edit (comment or docstring) that provably leaves the executable code
    identical -- compare the parsed AST with docstrings stripped, before and after. The
    record still describes the same computation, but the stamp then asserts "semantically
    the same code", not "just re-validated", and a reader deciding whether to trust the
    number needs to know which of the two it is. Anything touching executable code
    requires a real re-run; there is no AST argument for it.
    """
    changed = 0
    for sample in _validated_sample_paths():
        rel = sample.relative_to(REPO_ROOT)
        fingerprint = _code_fingerprint(sample)
        if fingerprint is None:
            LOGGER.info(f"skip  {rel} (cannot tokenize)")
            continue
        text = _text(sample)
        line = f"# {_VALIDATION_HASH_FIELD}: {fingerprint}"
        existing = _VALIDATION_HASH_RE.search(text)
        if existing:
            if existing.group(1) == fingerprint:
                continue
            updated = text[: existing.start()] + line + text[existing.end():]
        else:
            lines = text.splitlines(keepends=True)
            # Prefer immediately after the STATUS line, so the record reads as one block.
            insert_at = next(
                (i + 1 for i, l in enumerate(lines[:40]) if "STATUS:" in l and l.lstrip().startswith("#")),
                None,
            )
            if insert_at is not None:
                # Keep the stamp below any continuation lines of the STATUS comment block.
                while insert_at < len(lines) and re.match(r"^#\s{2,}\S", lines[insert_at]):
                    insert_at += 1
            else:
                # Reference goldens carry no STATUS line -- the validated claim lives on the
                # impl -- but the row covers the impl/golden pair, so the golden still has to
                # be pinned: editing it silently changes what the impl was validated against.
                # Anchor after the leading comment header instead.
                insert_at = 0
                while insert_at < len(lines) and lines[insert_at].lstrip().startswith("#"):
                    insert_at += 1
            lines.insert(insert_at, line + "\n")
            updated = "".join(lines)
        sample.write_text(updated, encoding="utf-8")
        LOGGER.info(f"stamp {rel} -> {fingerprint[:12]}...")
        changed += 1
    LOGGER.info(f"{changed} sample(s) stamped")
    return 0


def check_destructive_samples() -> list[str]:
    """A retained sample must not delete a directory it did not create.

    Samples are what a coder models an implementation on, so a sample that does
    ``shutil.rmtree("/tmp/...")`` teaches the pattern the agent rules forbid outright:
    every file an agent writes belongs under the working directory. Twelve samples did
    this and passed, because the path pattern only looked for home directories.
    """
    errors: list[str] = []
    for path in _kb_files():
        if path.suffix != ".py":
            continue
        for lineno, line in enumerate(_text(path).splitlines(), 1):
            if _DESTRUCTIVE.search(line):
                errors.append(
                    f"sample deletes a directory tree; write under the run's own output "
                    f"directory instead, at {path.relative_to(REPO_ROOT)}:{lineno}"
                )
    return errors


def check_skill_absolute_paths() -> list[str]:
    """The PyPTO skill surface must not carry machine-specific paths either.

    Scoped to ``ops/pypto-*``: these are the skills the operator-development
    flow loads. Other domains document their own installation defaults and are
    not in this flow's contract.
    """
    skills_root = KB_ROOT.parent
    if not skills_root.is_dir():
        return []
    self_path = Path(__file__).resolve()
    suffixes = {".md", ".py", ".sh", ".json"}

    def _is_skill_file(path: Path) -> bool:
        return path.is_file() and path.suffix in suffixes and path.resolve() != self_path

    paths = (path for path in sorted(skills_root.glob("pypto-*/**/*")) if _is_skill_file(path))
    return _absolute_path_errors(paths)


_CROSS_PACKAGE_LINK = re.compile(r"\[[^\]]*\]\(((?:\.\./)+pypto-[^)]+)\)")


_REPORT_ENTRY = re.compile(r"^### (\d+(?:\.\d+)?\..*)$", re.M)


def check_report_copies_agree() -> list[str]:
    """The KB's A5 limitations page and Part I of the consolidated report must agree.

    They are two copies of the same entries for two audiences: the KB page is routed to
    agents, the `docs/` report goes upstream. Nothing kept them in step, and they drifted
    exactly as predicted -- one copy called a defect `unexplained` while the other had
    already located it, and an entry title said "evaluation server" on one side and
    "deployed runtime" on the other. Both were found by review, not by this checker.

    Entry titles are compared, not prose: the titles carry the entry number and the claim,
    so a renumber, a dropped entry or a reworded verdict all show up, while ordinary
    editorial differences between the two framings do not.
    """
    kb_report = KB_ROOT / "references" / "pypto-pro-dsl-limitations-a5.md"
    docs_report = REPO_ROOT / "docs" / "pypto-pro-dsl-limitations.md"
    if not (kb_report.is_file() and docs_report.is_file()):
        return ["SKIP: one of the two report copies is absent"]
    docs = _text(docs_report)
    if "# Part I" not in docs or "# Part II" not in docs:
        return ["SKIP: the consolidated report has no Part I span to compare"]
    part_one_start = docs.index("# Part I")
    part_one_end = docs.index("# Part II")
    part_one = docs[part_one_start:part_one_end]
    in_docs = {m.group(1).strip() for m in _REPORT_ENTRY.finditer(part_one)}
    in_kb = {m.group(1).strip() for m in _REPORT_ENTRY.finditer(_text(kb_report))}
    errors = [
        f"{docs_report.relative_to(REPO_ROOT)} Part I has an entry the KB copy lacks: {title!r}"
        for title in sorted(in_docs - in_kb)
    ]
    errors += [
        f"{kb_report.relative_to(REPO_ROOT)} has an entry Part I lacks: {title!r}"
        for title in sorted(in_kb - in_docs)
    ]
    return errors


def check_cross_package_links() -> list[str]:
    """Every link that leaves its own package must resolve.

    These are the links between a skill and this KB, in both directions. They were
    excluded from checking on the theory that they are "authored for the installed
    layout" and only look broken in a checkout. They are not: skills install as
    **symlinks** into `cannbot-skills/ops/`, so a `..` traversal out of an installed
    skill is resolved physically and lands back among its siblings -- the same place a
    checkout puts it. The sibling form is correct under both resolvers, the deeper
    "installed" form under neither, and 17 links written to the latter resolved nowhere
    while this check did not exist to say so.

    Scoped to `ops/pypto-*`, matching `check_skill_absolute_paths`.
    """
    ops_root = KB_ROOT.parent
    if not ops_root.is_dir():
        return ["SKIP: no ops/ root to scan"]
    errors: list[str] = []
    seen = 0
    for md in sorted(ops_root.glob("pypto-*/**/*.md")):
        text = _text(md)
        for m in _CROSS_PACKAGE_LINK.finditer(text):
            rel = m.group(1)
            seen += 1
            if (md.parent / rel).exists():
                continue
            lineno = text[: m.start()].count("\n") + 1
            errors.append(
                f"{md.relative_to(REPO_ROOT)}:{lineno} links out of its package to "
                f"{rel!r}, which resolves nowhere; use the sibling form "
                f"(see CONTRACT.md)"
            )
    if not seen:
        return ["SKIP: no cross-package links found"]
    return errors


def check_tool_absolute_paths() -> list[str]:
    """``tools/`` must not carry machine-specific paths either.

    These scripts sit outside both surfaces checked above, and three of them
    shipped a contributor's own home-directory path as an argparse default: a
    leaked username, and a tool nobody else could run. Companion checkouts are
    located through ``tools/_repo_paths.py`` instead — an environment variable,
    then a sibling search.
    """
    tools_root = REPO_ROOT / "tools"
    if not tools_root.is_dir():
        # Not an ok. This branch carries the knowledge base only, so there is nothing to
        # scan -- and printing "ok" over zero files is how a green line comes to mean
        # "checked and clean" when it means "did not look". The first review's absolute-path
        # finding is still open on the branch that HAS tools/, so this line was masking.
        return ["SKIP: tools/ is absent here; run this on the branch that carries it"]
    paths = (
        path
        for path in sorted(tools_root.glob("**/*"))
        if path.is_file() and path.suffix in {".md", ".py", ".sh", ".json"}
    )
    return _absolute_path_errors(paths)


def check_contract_block() -> list[str]:
    """The shared vocabulary must be declared here, and internally consistent.

    ``topology-map.json`` is the single source for the contract a consumer validates
    generated knowledge artifacts against. Before this block existed the consumer kept its
    own copy of the topologies and the reference bound, and nothing noticed when the two
    drifted.
    """
    errors: list[str] = []
    topo = KB_ROOT / "topology-map.json"
    if not topo.is_file():
        return ["topology-map.json is missing; the contract has no source"]
    try:
        data = json.loads(topo.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"topology-map.json is not valid JSON: {exc}"]

    contract = data.get("contract")
    if not isinstance(contract, dict):
        return ["topology-map.json declares no `contract` block; see CONTRACT.md"]

    if not isinstance(contract.get("contract_version"), int):
        errors.append("contract.contract_version must be an integer")
    for key in ("selection_filename", "usage_filename", "consumer"):
        if not contract.get(key):
            errors.append(f"contract.{key} is required")
    for key in ("implementation_statuses", "property_keys"):
        if not isinstance(contract.get(key), list) or not contract[key]:
            errors.append(f"contract.{key} must be a non-empty list")

    if data.get("schema_version") != contract.get("contract_version"):
        errors.append(
            "topology-map schema_version must equal contract.contract_version"
        )

    # Every property a modifier routes on must be a declared property key, or a
    # selection could name a property the consumer will not recognize.
    declared = set(contract.get("property_keys") or [])
    modifiers = set((data.get("property_modifiers") or {}).keys())
    unknown = sorted(modifiers - declared)
    if unknown:
        errors.append("property_modifiers not listed in contract.property_keys: " + ", ".join(unknown))

    doc = KB_ROOT / "CONTRACT.md"
    if not doc.is_file():
        errors.append("CONTRACT.md is missing; the contract block has no prose")
    else:
        text = doc.read_text(encoding="utf-8")
        version = contract.get("contract_version")
        if version is not None and str(version) not in text:
            errors.append(f"CONTRACT.md does not state contract version {version}")
    return errors


def _module_constants(path: Path) -> dict:
    """Read module-level assignments without importing the module."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or node.value is None:
                continue
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}


_DOCS_ROOT = REPO_ROOT / "docs"
_ARTIFACT_CITE = re.compile(r"`((?:custom|tools)/[\w./{},-]+)`")
# A markdown link to another page. The `§N` it carries may sit **inside** the link text
# (`[ROUTER §99](../ROUTER.md)`) or after it (`[ROUTER](../ROUTER.md) §99`), and there may
# be more than one (`§1, §99`). The first version of this matched only the trailing shape
# and only the first number, so two of the three ways a citation is actually written went
# unchecked -- a gap an audit found, not a control.
_SECTION_LINK = re.compile(r"\[([^\]]*)\]\(([^)#]+?\.md)(?:#[^)]*)?\)")
_SECTION_NUM = re.compile(r"§(\d+(?:\.\d+)?)")
_PART_HEADING = re.compile(r"^# Part ([IVX]+)\b", re.M)
# A bare `entry N` is only ambiguous when the reader has to guess which Part it
# belongs to. `entry 16 of `some-page.md`` already names its target, and that
# target keeps its own numbering, so the lookahead exempts it -- without it this
# check demands a Part prefix for a cross-document citation, which is how a first
# run of it turned a correct reference into a wrong one.
_BARE_ENTRY = re.compile(
    r"(?<![-\w])(?:entry|Entry) (\d+)\b(?!\s+of\s+`[^`]*\.md`)")
# A qualified citation: `entry II-14`, or the bare `II-14` token the consolidated-ask
# lists use. Roman-dash-digit is distinctive enough to match unqualified -- a survey of
# every KB and docs page returned 58 hits, all of them genuine entry references.
_QUALIFIED_ENTRY = re.compile(r"\b([IVX]+)-(\d+)\b")
# Entries are numbered paragraphs in one Part of the merged report and `###` headings in
# another, because the two source reports wrote them differently. Match both, or the Part
# that uses bold paragraphs reads as defining no entries at all and every citation into it
# is reported missing.
_ENTRY_HEADING = re.compile(r"^(?:\*\*|#{2,4} )(\d+)\. ", re.M)


def _paragraphs(text: str):
    """(start_lineno, block) for each blank-line-delimited block, 1-indexed."""
    lineno, block, start = 1, [], 1
    for line in text.split("\n"):
        if line.strip():
            if not block:
                start = lineno
            block.append(line)
        elif block:
            yield start, "\n".join(block)
            block = []
        lineno += 1
    if block:
        yield start, "\n".join(block)


def _heading_numbers(text: str) -> set[str]:
    """Section numbers a page actually defines, as strings ('14', '14.2')."""
    return {
        m.group(1)
        for m in re.finditer(r"^#{1,4} +(?:§)?(\d+(?:\.\d+)?)[.\s]", text, re.M)
    }


def check_reference_artifact_citations() -> list[str]:
    """Any KB page citing `custom/...` or `tools/...` must cite something that RESOLVES.

    This rule already existed for `validated` rows in `patterns/pattern-index.md`, and
    nowhere else -- so `references/` and `playbooks/` pages could cite a retained artifact
    in exactly the form the KB reserves for evidence, on the one page class where nothing
    checked retention. Three such citations shipped: two naming files that exist on no ref
    at all, carrying a page's entire measured sweep, and the page named no branch.

    The failure is not cosmetic. A citation a reader cannot open invites re-running a
    measurement that was already paid for -- which is the reason this KB gives elsewhere
    for deleting dangling pointers.

    The `on branch <name>` escape hatch is honoured, but scoped to the **paragraph** that
    carries the citation. Searching the whole page for the keyword is how an unrelated
    sentence once laundered a broken path (see
    `check_pattern_validated_claims_cite_evidence`), and a page-wide scope would reopen it.
    """
    errors: list[str] = []
    # `docs/` is scanned too. Leaving it out is how a dangling `custom/...` path in the
    # merged DSL report survived a full green run: the citation rule existed, but not for
    # the directory that carried the citation.
    pages = list(_kb_files()) + (
        sorted(_DOCS_ROOT.glob("*.md")) if _DOCS_ROOT.is_dir() else []
    )
    for page in pages:
        if page.suffix == ".md" and page.name != "pattern-index.md":
            errors.extend(_artifact_citation_errors(page))
    return errors


def _artifact_citation_errors(page: Path) -> list[str]:
    """Unresolved `custom/`/`tools/` citations on one page.

    Split out so neither function nests more than four deep.
    """
    errors: list[str] = []
    text = _text(page)
    for start, block in _paragraphs(text):
        cites = _ARTIFACT_CITE.findall(block)
        if cites:
            named = _PATTERN_BRANCH.findall(block)
            present = [r for r in named if _ref_exists(r)]
            absent = [r for r in named if not _ref_exists(r)]
            errors.extend(
                _artifact_citation_error(page, start, citation, present, absent)
                for citation in cites
            )
    return [e for e in errors if e]


def _artifact_citation_error(
    page: Path,
    start: int,
    citation: str,
    present: list[str],
    absent: list[str],
) -> str | None:
    """One citation's verdict: an error string, a `SKIP:` note, or None when it resolves."""
    rel = page.relative_to(REPO_ROOT)
    parts = _expand(citation)
    # `all`, not `any`: `{present.py,absent.py}` names two files, and a citation is
    # only evidence for what it names. Under `any` the present member vouched for the
    # absent one -- the same "one member covers the rest" hole `_expand` was written to
    # close, reintroduced one line later.
    if all((REPO_ROOT / part).exists() for part in parts):
        return None
    if present and all(any(_git_has(r, part) for r in present) for part in parts):
        return None
    if absent and not present:
        # Same reasoning as `_citation_resolution_result`: unfetched and never-existed
        # are indistinguishable here, so say "unchecked" rather than passing a citation
        # nothing verified.
        return (
            f"SKIP: {rel}:{start} cites {citation!r} "
            f"against ref(s) {absent} not present locally; fetch to verify"
        )
    hint = (
        f"; it names branch(es) {present} but the path is not in them"
        if present
        else "; retain it, name the branch with `on branch <name>`, or drop the path"
    )
    return f"{rel}:{start} cites {citation!r} which resolves nowhere{hint}"


def _cited_sections(text: str, match: "re.Match[str]") -> list[str]:
    """Section numbers a single markdown link carries.

    Both spellings occur: the number can sit inside the link text
    (``[ROUTER §99](../ROUTER.md)``) or follow the link (``[ROUTER](../ROUTER.md) §99``),
    and a run may name several (``§1, §99``). The trailing window stops at the next ``[``
    so a second link's sections are never attributed to this one.
    """
    tail = text[match.end():match.end() + 40].split("[")[0]
    cited = _SECTION_NUM.findall(match.group(1)) + _SECTION_NUM.findall(tail)
    return list(dict.fromkeys(cited))


def _section_citation_errors(page: Path) -> list[str]:
    """Unresolved `§N` citations on one page.

    Split out of `check_section_citations` so neither function nests more than four
    deep; the loop-inside-loop-inside-loop form tripped the depth limit.
    """
    text = _text(page)
    rel = page.relative_to(REPO_ROOT)
    errors: list[str] = []
    for match in _SECTION_LINK.finditer(text):
        target_rel = match.group(2)
        target = (page.parent / target_rel).resolve()
        if not target.is_file():
            continue  # link resolution is check_links's job, not this one
        cited = _cited_sections(text, match)
        if not cited:
            continue
        defined = _heading_numbers(_text(target))
        missing = [section for section in cited if section not in defined]
        lineno = text[: match.start()].count("\n") + 1
        errors.extend(
            f"{rel}:{lineno} cites §{section} of {target_rel}, which defines no "
            f"such section"
            for section in missing
        )
    return errors


def check_section_citations() -> list[str]:
    """A `§N` citation must point at a section the target page actually defines.

    A number is a silent reference: when the target renumbers, the citation keeps
    resolving -- to the wrong section -- and nothing announces it. That happened here: a
    page cited `§15` while the target stopped at `§14.x`, so the link was visibly broken;
    a later commit added a `§15` on an unrelated subject and the citation became quietly
    wrong, which is worse than the dangling version it replaced.

    Only citations of the form `[...](target.md) ... §N` are checked -- the ones that name
    both the page and the number, so both halves can be compared.
    """
    errors: list[str] = []
    pages = list(_kb_files()) + (
        sorted(_DOCS_ROOT.glob("*.md")) if _DOCS_ROOT.is_dir() else []
    )
    for page in pages:
        if page.suffix == ".md":
            errors.extend(_section_citation_errors(page))
    return errors


def _entries_by_part(text: str) -> dict[str, set[str]]:
    """Which entry numbers each `# Part <ROMAN>` actually defines."""
    marks = [(m.group(1), m.start()) for m in _PART_HEADING.finditer(text)]
    parts: dict[str, set[str]] = {}
    for i, (roman, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        parts[roman] = {m.group(1) for m in _ENTRY_HEADING.finditer(text[start:end])}
    return parts


def check_multi_part_entry_citations() -> list[str]:
    """In a report with numbered Parts, `entry N` must say which Part.

    Merging two separately-numbered reports into one document makes every bare `entry N`
    ambiguous, and the reader lands on whichever Part comes first -- a wrong-number defect
    that never announces itself. One such document here carries two entries numbered 20,
    two numbered 24, and so on; thirteen bare citations pointed at the wrong one.

    Applies only to documents that declare `# Part <ROMAN>` headings more than once, since
    that is what makes a bare number ambiguous. A citation that names an external page --
    ``entry 16 of `pypto-pro-framework-findings.md` `` -- is exempt: the number indexes that
    page's numbering, not this document's Parts, and demanding a Part prefix there produces
    a citation that points at the wrong entry.

    A citation that *is* qualified is then resolved: `II-14` must be an entry Part II
    actually defines. Checking only that a prefix is present would accept any number.
    The limit is worth stating: resolution rejects an out-of-range prefix, not one
    that lands on a real but unrelated entry. No static rule reaches that.
    """
    errors: list[str] = []
    pages = list(_kb_files()) + (
        sorted(_DOCS_ROOT.glob("*.md")) if _DOCS_ROOT.is_dir() else []
    )
    for page in pages:
        if page.suffix != ".md":
            continue
        text = _text(page)
        if len(_PART_HEADING.findall(text)) < 2:
            continue
        rel = page.relative_to(REPO_ROOT)
        for m in _BARE_ENTRY.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            errors.append(
                f"{rel}:{lineno} cites bare 'entry {m.group(1)}' in a multi-part report; "
                f"say which part (e.g. 'entry II-{m.group(1)}')"
            )
        # Qualifying a citation is not the same as it resolving: the rule above demands a
        # Part prefix, and any number satisfies it. This catches a prefix that names an
        # entry the Part does not have. It does NOT catch a prefix that is in range and
        # still wrong -- `entry I-16` resolves, because Part I does define an entry 16;
        # that one is caught upstream by exempting citations which name a target file.
        parts = _entries_by_part(text)
        for m in _QUALIFIED_ENTRY.finditer(text):
            roman, number = m.groups()
            if number in parts.get(roman, ()):
                continue
            if roman not in parts:
                lineno = text[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} cites {roman}-{number}, but this document defines no "
                    f"Part {roman} (it defines {', '.join(sorted(parts))})"
                )
                continue
            lineno = text[: m.start()].count("\n") + 1
            known = sorted(parts[roman], key=int)
            span = f"{known[0]}-{known[-1]}" if known else "none"
            errors.append(
                f"{rel}:{lineno} cites {roman}-{number}, but Part {roman} defines no "
                f"entry {number} (it defines {span})"
            )
    return errors


CHECKS = (
    ("links resolve", check_links),
    ("files are reachable", check_reachability),
    ("topology map is valid", check_topology_map),
    ("contract block is declared", check_contract_block),
    ("filenames are durable", check_filenames),
    ("paths are portable", check_absolute_paths),
    ("samples do not delete shared directories", check_destructive_samples),
    ("validation records are scoped", check_sample_provenance_claims),
    ("validated rows have a record", check_validated_rows_have_records),
    ("validation records match the code", check_validation_records_match_code),
    ("validated patterns cite evidence", check_pattern_validated_claims_cite_evidence),
    ("every page is routable", check_every_page_is_routable),
    ("pypto skill paths are portable", check_skill_absolute_paths),
    ("repo tool paths are portable", check_tool_absolute_paths),
    ("cross-package links resolve", check_cross_package_links),
    ("report copies agree", check_report_copies_agree),
    ("reference citations resolve", check_reference_artifact_citations),
    ("section citations resolve", check_section_citations),
    ("multi-part entries are qualified", check_multi_part_entry_citations),
)


def main() -> int:
    if "--stamp-validation-hashes" in sys.argv[1:]:
        # Deliberately not part of the check run: stamping is an assertion that the code
        # below the record is what produced the recorded result. Only a human who has just
        # re-validated the sample is in a position to make it.
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        return stamp_validation_hashes()
    failed = 0
    skipped = 0
    for label, check in CHECKS:
        errors = check()
        skips = [e for e in errors if e.startswith("SKIP:")]
        errors = [e for e in errors if not e.startswith("SKIP:")]
        if errors:
            failed += len(errors)
            LOGGER.error(f"FAIL  {label} ({len(errors)})")
            for message in errors:
                LOGGER.info(f"        {message}")
        elif skips:
            # A check that examined no applicable artifact is inconclusive, not successful.
            skipped += len(skips)
            LOGGER.info(f"SKIP  {label}")
            for message in skips:
                LOGGER.info(f"        {message[len('SKIP:') :].strip()}")
        else:
            LOGGER.info(f"ok    {label}")
    if skipped:
        LOGGER.info(f"\n{skipped} check(s) SKIPPED -- they examined nothing, so they confirm nothing.")
    if failed:
        LOGGER.error(f"\n{failed} KB integrity violation(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
