# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# test_template.py
# Starter for verifier-owned test files:
#   - Per-module scaffolding step C:
#       custom/<op>/modules/test_<op>_module<suffix_k>.py
#   - Cleanup E2E:
#       custom/<op>/test_<op>.py
#
# Every test file imports the impl and the golden SEPARATELY and uses
# `detailed_tensor_compare` on every leaf output. L0 (small shapes) and
# L1 (P0 shapes from SPEC.md) levels are mandatory. Larger levels
# (L2..L5) are exercised by `eval/adversarial_runner.py`, not by these
# per-module / E2E tests.
# =============================================================================
#
# CRITICAL invariants enforced by lint (rules.json D4 dimension):
#
#   OL19 — every assertion uses `detailed_tensor_compare` (no hand-rolled
#          `assert max_diff < tol`).
#   OL20 — `torch.npu.set_device(int(os.environ["TILE_FWK_DEVICE_ID"]))`
#          MUST appear before any tensor moves to NPU.
#   OL21 — both `test_*_l0` (small shapes) and `test_*_l1` (P0 shapes
#          from SPEC.md) functions exist.
#   OL22 — `torch.manual_seed(42)` is set before generating random inputs.
#   OL42 — `run_mode` is NOT hard-coded to 'sim' when the NPU env is
#          available.
#
# Convention (no lint rule, enforced by `test_inputs.py::make_inputs`):
#   - Every tensor returned by `make_inputs(case)` MUST be created on the
#     same NPU device via the explicit `device=` keyword at construction
#     time (e.g. `torch.randn(shape, dtype=..., device=DEVICE)`).
#     Mixed-device inputs (some on CPU, some on NPU) cause PyPTO kernel
#     dispatch failures. Do NOT create tensors on CPU first and then
#     `.npu()`; do NOT rely on `torch.npu.set_device(...)` alone. See
#     `pypto-op-verifier.md` Step B.1 for the canonical pattern.
# =============================================================================

import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# detailed_tensor_compare path bootstrap (self-contained — no PYTHONPATH needed)
# ═══════════════════════════════════════════════════════════════════════════════
# Walk up from this test file to locate
# `.agents/skills/pypto-op-verify/scripts/` and prepend it to sys.path. This
# lets `python custom/<op>/test_<op>.py` work from any cwd without the user
# having to set `PYTHONPATH=.agents/skills/pypto-op-verify/scripts`. Keep the
# helper symbol names underscore-prefixed and `del` them at the end so the
# test file's public surface stays clean.
_test_dir = os.path.dirname(os.path.abspath(__file__))
_current = _test_dir
_candidate = None
for _ in range(8):  # bounded ascent — repo root is typically 2-5 levels up
    _candidate = os.path.join(_current, ".agents", "skills", "pypto-op-verify", "scripts")
    if os.path.isdir(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
    _parent = os.path.dirname(_current)
    if _parent == _current:
        _candidate = None
        break
    _current = _parent
if _candidate is None or not os.path.isdir(_candidate):
    raise ImportError(
        "Could not locate detailed_tensor_compare. Expected "
        "`.agents/skills/pypto-op-verify/scripts/detailed_tensor_compare.py` "
        f"reachable from {_test_dir} by walking up the directory tree."
    )
del _test_dir, _current, _candidate
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch_npu  # noqa: F401  required for NPU device init

# ── Imports — exact filenames, exact symbol names. Do NOT alias. ──
#
# For modules/test_<op>_module<suffix_k>.py:
#   from <op>_module<suffix_k>_impl   import <op>_module<suffix_k>_wrapper
#   from <op>_module<suffix_k>_golden import <op>_module<suffix_k>_golden
#
# For test_<op>.py (cleanup E2E):
#   from <op>_impl   import <op>_wrapper
#   from <op>_golden import <op>_golden

# from <module>_impl   import <op>_wrapper          # impl side
# from <module>_golden import <op>_golden_fn        # golden side

from detailed_tensor_compare import detailed_tensor_compare

# `make_inputs` lives in custom/<op>/eval/test_inputs.py and is produced
# in scaffolding step B by the verifier. It reads the JSON
# entries in `custom/<op>/eval/test_cases.json` (copied from
# pypto-op-verify/templates/test_cases_template.json and filled in with
# the SPEC.md shapes / dtypes) and returns the input tensors required
# by the impl wrapper for one given test case.
# from test_inputs import make_inputs


# =============================================================================
# Helpers
# =============================================================================

def _set_device() -> None:
    """Initialise the NPU device from the standard env var."""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))


def _to_tuple(x):
    """Normalise single tensor or tuple of tensors into a tuple."""
    return x if isinstance(x, tuple) else (x,)


def _compare_all_leaves(impl_out, gold_out, *, atol: float, rtol: float, label: str) -> None:
    """Run detailed_tensor_compare on every leaf output. Multi-output
    kernels MUST iterate (NPU lesson 5: aggregating into a single
    boolean masks per-output drift)."""
    impl_t = _to_tuple(impl_out)
    gold_t = _to_tuple(gold_out)
    assert len(impl_t) == len(gold_t), (
        f"output count mismatch: impl={len(impl_t)} golden={len(gold_t)}"
    )
    for i, (a, b) in enumerate(zip(impl_t, gold_t)):
        detailed_tensor_compare(
            a, b,
            atol=atol, rtol=rtol,
            tensor_name=f"{label}_out{i}",
        )


# =============================================================================
# Tests
#
# Function naming convention (used by the orchestrator to dispatch and
# by lint OL21 to validate the L0 / L1 pair exists):
#
#   For modules/test_<op>_module<suffix_k>.py:
#     def test_module<suffix_k>_l0(): ...     # small shapes, fast smoke
#     def test_module<suffix_k>_l1(): ...     # P0 shapes from SPEC.md
#
#   For test_<op>.py (cleanup E2E):
#     def test_<op>_l0(): ...
#     def test_<op>_l1(): ...
# =============================================================================

# Per-test-case fields (id / level / seed / shape / dtype / atol / rtol)
# live in `custom/<op>/eval/test_cases.json` (copied from
# pypto-op-verify/templates/test_cases_template.json). The helper
# `make_inputs` looks up an entry by `id` and returns the input tensors.
# atol / rtol are read from the same JSON entry, so do NOT re-inline
# them as Python constants here.


def test_<label>_l0() -> None:
    """L0: smallest legal shapes — fast smoke test. Must run on NPU."""
    _set_device()
    torch.manual_seed(42)

    # case = load_case("<label>_l0")              # reads test_cases.json
    # inputs = make_inputs(case)                  # tensors per case["shape"]/case["dtype"]/case["seed"]
    # impl_out = <op>_wrapper(*inputs.values())
    # gold_out = <op>_golden_fn(*inputs.values())
    # _compare_all_leaves(
    #     impl_out, gold_out,
    #     atol=case["atol"], rtol=case["rtol"],
    #     label=case["id"],
    # )
    raise NotImplementedError


def test_<label>_l1() -> None:
    """L1: P0 shapes from SPEC.md — full-fidelity precision test."""
    _set_device()
    torch.manual_seed(42)

    # case = load_case("<label>_l1")
    # inputs = make_inputs(case)
    # impl_out = <op>_wrapper(*inputs.values())
    # gold_out = <op>_golden_fn(*inputs.values())
    # _compare_all_leaves(
    #     impl_out, gold_out,
    #     atol=case["atol"], rtol=case["rtol"],
    #     label=case["id"],
    # )
    raise NotImplementedError


# =============================================================================
# Optional: standalone runner (cleanup E2E only)
#
# Uncomment for `test_<op>.py` so the file can be invoked directly
# (`python test_<op>.py`). Per-module tests under modules/ stay pytest-only.
# =============================================================================

# if __name__ == "__main__":
#     test_<label>_l0()
#     test_<label>_l1()
#     print("[PRECISION_PASS]")
