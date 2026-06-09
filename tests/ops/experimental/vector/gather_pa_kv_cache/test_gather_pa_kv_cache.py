# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

_repo = THIS_DIR
while _repo != os.path.dirname(_repo) and not os.path.isdir(os.path.join(_repo, "src")):
    _repo = os.path.dirname(_repo)
if os.path.isdir(os.path.join(_repo, "src")):
    sys.path.insert(0, os.path.join(_repo, "src"))
    sys.path.insert(0, os.path.join(_repo, "src", "pypto_gym", "ops", "pypto_tile"))

from gather_pa_kv_cache_golden import gather_pa_kv_cache_golden, make_case  # noqa: E402
from experimental.vector.gather_pa_kv_cache.gather_pa_kv_cache_impl import (  # noqa: E402
    gather_pa_kv_cache_wrapper,
)


REQUIRED_LEVELS = tuple(f"level{idx}" for idx in range(10))


def _device() -> str:
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _load_cases(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(THIS_DIR, "test_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cases = cfg["test_cases"]
    ids = {case["id"] for case in cases}
    for level in REQUIRED_LEVELS:
        if level not in ids:
            raise RuntimeError(f"missing required test level '{level}' in {path}")
    return cases


def _select_cases(cases: list[dict], selected: list[str]) -> list[dict]:
    if not selected:
        return cases
    wanted = set(selected)
    out = [case for case in cases if case["id"] in wanted]
    missing = wanted - {case["id"] for case in out}
    if missing:
        raise RuntimeError(f"unknown test case(s): {sorted(missing)}")
    return out


def _make_inputs(case: dict, device: str) -> tuple[dict, dict]:
    inp = case["input"]
    case_cpu = make_case(
        total_tokens=int(inp["total_tokens"]),
        q_count=int(inp["q_count"]),
        num_blocks=int(inp["key_cache_shape"][0]),
        block_table_cols=int(inp["block_tables_shape"][1]),
        block_size=int(inp["key_cache_shape"][1]),
        key_num_heads=int(inp["key_cache_shape"][2]),
        key_dim=int(inp["key_cache_shape"][3]),
        value_num_heads=int(inp["value_cache_shape"][2]),
        value_dim=int(inp["value_cache_shape"][3]),
        is_seq_lens_cumsum=bool(inp.get("is_seq_lens_cumsum", False)),
        compute_golden=False,
        seed=int(case.get("seed", 42)),
    )
    case_dev = dict(case_cpu)
    for name in (
        "key_cache",
        "value_cache",
        "block_tables",
        "seq_lens",
        "key_ref",
        "value_ref",
        "seq_offset",
    ):
        case_dev[name] = case_cpu[name].to(device).contiguous()
    return case_cpu, case_dev


def _assert_shapes(case: dict, key_out: torch.Tensor, value_out: torch.Tensor) -> None:
    expected = case["output"]
    key_shape = tuple(expected["key_shape"])
    value_shape = tuple(expected["value_shape"])
    if tuple(key_out.shape) != key_shape:
        raise AssertionError(f"key output shape mismatch: got {tuple(key_out.shape)}, expected {key_shape}")
    if tuple(value_out.shape) != value_shape:
        raise AssertionError(f"value output shape mismatch: got {tuple(value_out.shape)}, expected {value_shape}")
    if key_out.dtype != torch.bfloat16 or value_out.dtype != torch.bfloat16:
        raise AssertionError(f"output dtype mismatch: key={key_out.dtype}, value={value_out.dtype}")


def _run_case(case: dict, device: str) -> bool:
    case_id = case["id"]
    inp = case["input"]
    print("=" * 60)
    print(f"Test: {case_id} - {case.get('description', '')}")
    print("=" * 60)

    case_cpu, case_dev = _make_inputs(case, device)
    key_out, value_out = gather_pa_kv_cache_wrapper(
        case_dev["key_cache"],
        case_dev["value_cache"],
        case_dev["block_tables"],
        case_dev["seq_lens"],
        case_dev["key_ref"],
        case_dev["value_ref"],
        case_dev["seq_offset"],
        cache_mode=inp.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=bool(inp.get("is_seq_lens_cumsum", False)),
        run_mode="npu",
    )
    torch.npu.synchronize()
    _assert_shapes(case, key_out, value_out)

    key_ref, value_ref = gather_pa_kv_cache_golden(
        case_cpu["key_cache"],
        case_cpu["value_cache"],
        case_cpu["block_tables"],
        case_cpu["seq_lens"],
        case_cpu["key_ref"],
        case_cpu["value_ref"],
        case_cpu["seq_offset"],
        cache_mode=inp.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=bool(inp.get("is_seq_lens_cumsum", False)),
    )
    key_equal = torch.equal(key_out.detach().cpu(), key_ref)
    value_equal = torch.equal(value_out.detach().cpu(), value_ref)
    print(f"  [key] shape={tuple(key_out.shape)} equal={key_equal}")
    print(f"  [value] shape={tuple(value_out.shape)} equal={value_equal}")
    return bool(key_equal and value_equal)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precision test for gather_pa_kv_cache ND network sweep")
    parser.add_argument("cases", nargs="*", help="case ids from test_cases.json, e.g. level0 level9")
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    parser.add_argument("--run-mode", default="npu", choices=["npu"], help="execution backend")
    args = parser.parse_args()

    try:
        cases = _load_cases()
        if args.list:
            for case in cases:
                print(f"{case['id']}: {case.get('description', '')}")
            return 0
        selected_cases = _select_cases(cases, args.cases)
        device = _device()
        print(f"Using device: {device}")
        all_ok = True
        for case in selected_cases:
            try:
                ok = _run_case(case, device)
            except Exception:
                traceback.print_exc()
                ok = False
            all_ok = all_ok and ok
        if all_ok:
            print("[PRECISION_PASS]")
            return 0
        print("[PRECISION_FAIL]")
        return 1
    except Exception:
        traceback.print_exc()
        print("[PRECISION_FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
