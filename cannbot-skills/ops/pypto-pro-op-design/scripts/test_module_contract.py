#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Contract test: what gen_module_interfaces.py emits must pass validate_module_yaml.py.

This existed as a gap, not a test. The generator and SKILL.md documented
`module_<j>` while the validator's regex accepted only the pre-rename `phase_<j>`,
so every artifact produced by following the skill failed the mandatory Stage-3
self-check with rule2/rule3 violations -- and the validator's own self-test passed,
because its fixtures had been rewritten to the spelling only it accepted.

The test deliberately derives the module-reference spelling from the *generator's
own output* rather than hardcoding it. If someone changes the generator's spelling,
this test follows it and the validator must keep up -- which is the contract.

Run::

    python3 test_module_contract.py
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_LOGGER = logging.getLogger("test_module_contract")

_HERE = Path(__file__).resolve().parent
_GEN = _HERE / "gen_module_interfaces.py"
_VALIDATE = _HERE / "validate_module_yaml.py"

_GOLDEN = '''\
import torch


def demo_golden_cpu(x, w):
    h = x + w
    return h * 2
'''

_SPEC = "# SPEC for demo\n\natol: 1e-3\nrtol: 1e-3\n"


def _generate_skeleton(tmp: Path) -> str:
    golden = tmp / "demo_golden_cpu.py"
    golden.write_text(_GOLDEN, encoding="utf-8")
    spec = tmp / "SPEC.md"
    spec.write_text(_SPEC, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(_GEN), str(golden), "--spec", str(spec), "--op", "demo"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


def _module_ref_prefix(skeleton: str) -> str:
    """Read the module-reference spelling out of the generator's own skeleton.

    The generator emits `source: <prefix>_TODO` under final_outputs; that token is
    the spelling the architect is told to use, so it is the spelling the validator
    has to accept.
    """
    match = re.search(r"source:\s*([A-Za-z]+)_TODO", skeleton)
    if not match:
        raise AssertionError(
            "generator no longer emits a `source: <prefix>_TODO` token; "
            "update this test to track the new skeleton shape"
        )
    return match.group(1)


def _fill_skeleton(prefix: str) -> str:
    """Hand-fill the TODO parts the way SKILL.md instructs an architect to.

    Two modules so that rule2 (cross-module reference) is actually exercised;
    a single-module fill would never emit a `<prefix>_<j>` input at all.
    """
    return f"""\
schema_version: 1
op: demo
module_count: 2
has_cross_core: false
is_fusion: true
primary_inputs:
  - {{name: x, shape: "[M, N]", dtype: float16}}
  - {{name: w, shape: "[M, N]", dtype: float16}}
modules:
  - id: 1
    name: add_inputs
    description: elementwise add
    section: vector
    golden_steps:
      - h = x + w
    inputs:
      - {{name: x, source: primary}}
      - {{name: w, source: primary}}
    outputs:
      - {{name: h, shape: "[M, N]", dtype: float16}}
    golden_stage_fn: demo_golden_stage1
  - id: 2
    name: scale
    description: multiply by two
    section: vector
    golden_steps:
      - out = h * 2
    inputs:
      - {{name: h, source: {prefix}_1}}
    outputs:
      - {{name: out, shape: "[M, N]", dtype: float16}}
    golden_stage_fn: demo_golden_stage12
final_outputs:
  - {{name: out, source: {prefix}_2}}
composition_verification:
  atol: 0.001
  rtol: 0.001
  seeds: [42, 123, 456]
  shapes:
    - {{M: 8, N: 16}}
"""


def _validate(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_VALIDATE), str(path), "--json"],
        capture_output=True, text=True,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)

        skeleton = _generate_skeleton(tmp)
        prefix = _module_ref_prefix(skeleton)
        _LOGGER.info("generator emits module references as `%s_<j>`", prefix)

        # 1. The generator's documented spelling must validate.
        filled = tmp / "module_interfaces.yaml"
        filled.write_text(_fill_skeleton(prefix), encoding="utf-8")
        proc = _validate(filled)
        if proc.returncode != 0:
            failures.append(
                "a module_interfaces.yaml filled in per the generator skeleton "
                f"was rejected by the validator:\n{proc.stdout}{proc.stderr}"
            )
            _LOGGER.warning("FAIL generator_output_validates")
        else:
            _LOGGER.info("PASS generator_output_validates")

        # 2. The pre-rename spelling must keep validating, so artifacts written
        #    before the rename are not invalidated by the canonicalisation.
        legacy = tmp / "module_interfaces_legacy.yaml"
        legacy.write_text(_fill_skeleton("phase"), encoding="utf-8")
        proc = _validate(legacy)
        if proc.returncode != 0:
            failures.append(
                f"the legacy `phase_<j>` spelling no longer validates:\n{proc.stdout}"
            )
            _LOGGER.warning("FAIL legacy_spelling_still_validates")
        else:
            _LOGGER.info("PASS legacy_spelling_still_validates")

        # 3. An unrelated prefix must still be refused, so the relaxation above
        #    did not turn the source rule into a no-op.
        bogus = tmp / "module_interfaces_bogus.yaml"
        bogus.write_text(_fill_skeleton("stage"), encoding="utf-8")
        proc = _validate(bogus)
        if proc.returncode == 0:
            failures.append("an unknown `stage_<j>` source prefix was accepted")
            _LOGGER.warning("FAIL unknown_prefix_refused")
        else:
            _LOGGER.info("PASS unknown_prefix_refused")

    for problem in failures:
        _LOGGER.warning("\n%s", problem)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
