#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Generate module_interfaces.yaml skeleton from golden_cpu.py + SPEC.md.

Auto-fills the deterministic parts (schema_version, op, primary_inputs,
composition_verification). The architect fills the TODO-marked judgement
parts (modules[] boundaries + final_outputs wiring + is_fusion +
has_cross_core + golden_steps).

Usage::

    python gen_module_interfaces.py custom/<op>/<op>_golden_cpu.py \\
        --spec custom/<op>/SPEC.md --op <op> \\
        --design custom/<op>/DESIGN.md \\
        > custom/<op>/module_interfaces.yaml

The skeleton is emitted on stdout; the architect redirects to the target
path and then hand-fills the TODO sections.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

_LOGGER = logging.getLogger("gen_module_interfaces")


def _emit_skeleton(skeleton: str) -> None:
    """Write generated YAML to stdout through a non-propagating logger."""
    output_logger = logging.Logger("gen_module_interfaces.output", level=logging.INFO)
    output_logger.propagate = False
    output_handler = logging.StreamHandler(sys.stdout)
    output_handler.setFormatter(logging.Formatter("%(message)s"))
    output_logger.addHandler(output_handler)
    output_logger.info("%s", skeleton)


def extract_golden_signature(golden_path: Path) -> dict:
    """Parse the golden function signature to extract primary input names.

    Looks for `def {op}_golden_cpu(...)` and extracts parameter names
    that are tensors (skip device/dtype kwargs).
    """
    source = golden_path.read_text(encoding="utf-8")
    # Find the golden_cpu function definition
    m = re.search(r"^def\s+\w+_golden_cpu\s*\(([^)]*)\)", source, re.MULTILINE)
    if not m:
        # Fallback: try _golden
        m = re.search(r"^def\s+\w+_golden\s*\(([^)]*)\)", source, re.MULTILINE)
    if not m:
        return {"params": [], "source": "could not find golden function"}

    params_str = m.group(1).strip()
    params = []
    for raw_param in params_str.split(","):
        param = raw_param.strip()
        if not param or param.startswith("*"):
            continue
        name = re.split(r"[:=]", param)[0].strip()
        # Skip non-tensor params (device, dtype, etc.)
        if name in ("device", "dtype", "self", "cls"):
            continue
        params.append(name)

    return {"params": params}


def extract_op_name(golden_path: Path) -> str:
    """Extract op name from golden filename: <op>_golden_cpu.py → <op>."""
    stem = golden_path.stem  # e.g. "softmax_golden_cpu"
    if stem.endswith("_golden_cpu"):
        return stem[: -len("_golden_cpu")]
    if stem.endswith("_golden"):
        return stem[: -len("_golden")]
    return stem


def generate_skeleton(golden_path: Path, spec_path: Path, op_name: str) -> str:
    """Generate the YAML skeleton string."""
    sig = extract_golden_signature(golden_path)
    params = sig["params"]

    # Build primary_inputs section (shape/dtype are TODO — architect fills from SPEC)
    primary_inputs_lines = []
    for p in params:
        primary_inputs_lines.append(
            f'  - {{name: {p}, shape: TODO, dtype: TODO}}'
        )
    if primary_inputs_lines:
        primary_inputs_block = "\n".join(primary_inputs_lines)
    else:
        primary_inputs_block = "  - {name: TODO, shape: TODO, dtype: TODO}"

    # Build modules section (skeleton with 1 TODO module; architect adds more)
    modules_block = """  - id: 1
    name: TODO                           # 语义名，如 "compute_qk_softmax"
    description: TODO                    # 该 Module 做什么
    section: TODO                        # cube 或 vector（按 R0：Cube 和 Vector 必须划分到不同 Module）
    golden_steps:                        # 该 Module 对应的数学步骤（供 mathematician 切分 golden 用，来源: DESIGN.md §0 R0 + §1 R1）
      - TODO
    inputs:                              # source 只能是 primary 或 module_<j>（j < 当前 id）
      - {name: TODO, source: primary}
    outputs:                             # Module 输出 = 下游 Module 输入 或 final_outputs
      - {name: TODO, shape: TODO, dtype: TODO}
    golden_stage_fn: %s_golden_stage1   # per-Module 累积 golden 函数名（mathematician 从 golden_cpu.py 切分产出）
  # 复制上方块添加更多 Module...""" % op_name

    skeleton = f"""\
schema_version: 1
op: {op_name}
module_count: TODO                        # ≥1，等于 modules 数组长度
has_cross_core: TODO                     # true/false（来自 DESIGN.md §6，信息记录用，不影响分流判据）
is_fusion: TODO                          # true/false（同时含 cube 和 vec section → true；来自 DESIGN.md §0 R0）
                                         # L0: is_fusion=false（纯vec/纯cube 一口气开发）
                                         # L1: is_fusion=true（融合算子逐 Module 开发，按 R0 定义 module_count≥2）
primary_inputs:                           # 镜像 golden 签名（shape/dtype 从 SPEC.md 填写）
{primary_inputs_block}
modules:                                   # 按 Module 顺序，id 从 1 起
{modules_block}
final_outputs:                            # 每个 golden 返回值对应到产出 Module
  - {{name: TODO, source: module_TODO}}
composition_verification:                 # 组合验证参数（atol/rtol 从 SPEC.md 填写）
  atol: TODO
  rtol: TODO
  seeds: [42, 123, 456]
  shapes:
    - {{TODO}}
"""

    return skeleton


def main() -> int:
    # The skeleton goes to stdout with print(); diagnostics stay on stderr.
    #
    # Both halves are load-bearing, and getting either wrong is silent. The documented
    # usage redirects stdout into custom/<op>/module_interfaces.yaml, so logging the
    # skeleton (logging defaults to stderr) produced a 0-byte file. But routing the whole
    # logger to stdout to fix that was worse: an error then landed *inside* the YAML, and
    # validate_module_yaml.py reports PASS on it -- a one-key mapping parses fine, so
    # `modules` is [] and module_count defaults to 0 == len([]). The architect advanced to
    # Stage 4 with an error string as the module contract. Split the two streams instead.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    ap = argparse.ArgumentParser(
        description="Generate module_interfaces.yaml skeleton"
    )
    ap.add_argument("golden", type=Path, help="Path to {op}_golden_cpu.py")
    ap.add_argument("--spec", type=Path, required=True, help="Path to SPEC.md")
    ap.add_argument("--op", type=str, required=True, help="Operator name")
    ap.add_argument("--design", type=Path, required=False, help="Path to DESIGN.md (unused in skeleton, reserved)")
    args = ap.parse_args()

    if not args.golden.exists():
        _LOGGER.error("golden file not found: %s", args.golden)
        return 1

    # Use --op if provided, otherwise extract from filename
    op_name = args.op or extract_op_name(args.golden)

    skeleton = generate_skeleton(args.golden, args.spec, op_name)
    _emit_skeleton(skeleton)
    return 0


if __name__ == "__main__":
    sys.exit(main())
