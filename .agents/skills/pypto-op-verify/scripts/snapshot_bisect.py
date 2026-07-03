#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# snapshot_bisect.py
#
# Per-iteration bisection runner for the snapshot automation pipeline.
#
# Given a snapshot manifest and the generated artifacts under custom/<op>/_debug/,
# it:
#   (a) loads test_inputs.make_inputs(case) to build concrete inputs,
#   (b) runs the generated snapshot kernel under each requested mode
#       (torch/sim/npu) and captures the inspection_<name> buffers,
#   (c) runs the generated golden's module_<k>_inspect() for reference,
#   (d) compares per-iteration (axis=2 is the NT axis for inside_nt_loop
#       probes) and prints a drift-onset report: the first iteration where
#       each intermediate diverges beyond atol+rtol*|truth|.
#
# Output is human-readable plus a JSON report under custom/<op>/_debug/snapshot_report.json.
#
# This is a DEBUG TOOL — no information-barrier sanitization is applied;
# @pypto-op-debugger sees full per-iteration metrics (but NOT raw golden tensor values,
# since the comparison reduces to scalar max diffs per iteration).
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_manifest_schema import load_manifest   # noqa: E402


# -----------------------------------------------------------------------------
# Module loading helpers
# -----------------------------------------------------------------------------

def _load_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def _set_run_mode(impl_module, mode: str) -> None:
    """Reconfigure the JIT wrapper in impl_module to run under the given mode."""
    import pypto
    mode_map = {"sim": pypto.RunMode.SIM, "npu": pypto.RunMode.NPU}
    try:
        rm = mode_map[mode]
    except KeyError as e:
        raise KeyError(
            f"{mode!r} is not a valid mode; available: {sorted(mode_map)}"
        ) from e
    # The JIT wrapper stores runtime_options; poke it directly. We look for the
    # wrapper by scanning the module for callables whose `_runtime_options`
    # attr exists (the contract from python/pypto/frontend/parser/entry.py).
    for name in dir(impl_module):
        obj = getattr(impl_module, name)
        if hasattr(obj, "_runtime_options") and isinstance(obj._runtime_options, dict):
            obj._runtime_options["run_mode"] = rm.value
            return
    raise RuntimeError(f"could not locate JIT wrapper in {impl_module.__name__} to set run_mode={mode}")


# -----------------------------------------------------------------------------
# Inspection buffer allocation
# -----------------------------------------------------------------------------

def _alloc_inspection_buffers(
    intermediates: List[Dict],
    B: int, H: int, NT: int,
    resolve_shape,                # callable: (shape_list, env) -> concrete list[int]
    env: Dict[str, int],
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Allocate zero buffers for every inspection_<name> the manifest requires.
    Shape is [B, H, NT, *shape] for inside_nt_loop probes, [B, H, *shape] else.
    """
    buffers: Dict[str, torch.Tensor] = {}
    for entry in intermediates:
        concrete = [int(s) if isinstance(s, int) else int(resolve_shape(s, env)) for s in entry["shape"]]
        if entry["probe_point"] == "inside_nt_loop":
            shape = [B, H, NT, *concrete]
        else:
            shape = [B, H, *concrete]
        buffers[f"inspection_{entry['name']}"] = torch.zeros(shape, dtype=dtype, device=device)
    return buffers


# -----------------------------------------------------------------------------
# Per-mode execution
# -----------------------------------------------------------------------------

def _run_mode(
    impl_module,
    mode: str,
    inputs: Dict[str, Any],
    inspection_buffers: Dict[str, torch.Tensor],
) -> Tuple[str, Dict[str, torch.Tensor], str]:
    """
    Invoke the snapshot kernel under one mode. Returns (status, buffers, log).
    Buffers are moved to CPU on return so the caller can free device memory
    between modes.
    """
    try:
        _set_run_mode(impl_module, mode)
    except Exception as e:
        return ("SKIPPED_MODE_SETUP", {}, f"{e}")

    # Check mode availability at runtime.
    if mode == "npu":
        if not os.environ.get("ASCEND_HOME_PATH"):
            return ("SKIPPED_NO_NPU_ENV", {}, "ASCEND_HOME_PATH not set")
        try:
            if not torch.npu.is_available():
                return ("SKIPPED_NO_NPU_DEVICE", {}, "torch.npu.is_available() == False")
        except Exception as e:
            return ("SKIPPED_NO_NPU_DEVICE", {}, f"{e}")
    t0 = time.time()
    try:
        impl_module.host_wrapper(**inputs, **inspection_buffers)
    except Exception as e:
        return ("ERROR", {}, f"{e}\n{traceback.format_exc()}")
    runtime = time.time() - t0
    cpu_buffers = {k: v.detach().cpu() for k, v in inspection_buffers.items()}
    return ("OK", cpu_buffers, f"runtime={runtime:.3f}s")


# -----------------------------------------------------------------------------
# Per-iteration compare
# -----------------------------------------------------------------------------

def _compare_per_iter(
    impl_tensor: torch.Tensor,
    gold_tensor: torch.Tensor,
    atol: float,
    rtol: float,
    has_iter_axis: bool,
) -> List[Dict[str, Any]]:
    """
    Return a list of per-iteration records:
      [{iter, max_abs_diff, max_rel_diff, all_close}, ...]
    If has_iter_axis=False (before/after probe), returns a single-element list.
    """
    impl = impl_tensor.float()
    gold = gold_tensor.float()
    if impl.shape != gold.shape:
        return [{"iter": -1, "max_abs_diff": float("inf"), "max_rel_diff": float("inf"),
                 "all_close": False, "error": f"shape mismatch {impl.shape} vs {gold.shape}"}]

    if not has_iter_axis:
        diff = (impl - gold).abs()
        denom = gold.abs().clamp_min(1e-30)
        rel = diff / denom
        tol_ok = bool((diff <= atol + rtol * gold.abs()).all().item())
        return [{"iter": -1, "max_abs_diff": float(diff.max().item()),
                 "max_rel_diff": float(rel.max().item()), "all_close": tol_ok}]

    # axis=2 is the NT axis; iterate over it.
    NT = impl.shape[2]
    out: List[Dict[str, Any]] = []
    for k in range(NT):
        diff = (impl[:, :, k] - gold[:, :, k]).abs()
        denom = gold[:, :, k].abs().clamp_min(1e-30)
        rel = diff / denom
        tol_ok = bool((diff <= atol + rtol * gold[:, :, k].abs()).all().item())
        out.append({
            "iter": k,
            "max_abs_diff": float(diff.max().item()),
            "max_rel_diff": float(rel.max().item()),
            "all_close": tol_ok,
        })
    return out


def _drift_onset(per_iter: List[Dict[str, Any]]) -> Optional[int]:
    """First iter index where all_close=False, or None if all pass."""
    for rec in per_iter:
        if not rec["all_close"]:
            return int(rec["iter"])
    return None


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def run(manifest_path: str | Path, *, modes_override: Optional[List[str]] = None,
        device: str = "9", verbose: bool = True) -> Dict[str, Any]:
    m = load_manifest(manifest_path)
    op, module, case_id = m["op"], m["module"], m["case"]
    modes = modes_override or m["modes"]

    repo_root = Path(manifest_path).resolve().parents[3]
    # parents: [0]=_debug, [1]=<op>, [2]=custom, [3]=repo root
    op_dir = repo_root / "custom" / op
    debug_dir = op_dir / "_debug"
    eval_dir = op_dir / "eval"

    # 1. Load the impl (snapshot version), golden-snapshot, and suite.
    impl_path = debug_dir / f"{op}_{module}_snapshot.py"
    golden_path = debug_dir / f"{op}_golden_modular_snapshot.py"
    suite_path = eval_dir / "adversarial_suite.json"
    test_inputs_path = eval_dir / "test_inputs.py"

    for p in (impl_path, golden_path, suite_path, test_inputs_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    sys.path.insert(0, str(eval_dir))
    sys.path.insert(0, str(op_dir))
    test_inputs_mod = _load_from_path(f"{op}_test_inputs", test_inputs_path)
    impl_mod = _load_from_path(f"{op}_{module}_snapshot_impl", impl_path)
    golden_mod = _load_from_path(f"{op}_golden_snapshot", golden_path)

    # Resolve the target case.
    with suite_path.open("r") as f:
        suite = json.load(f)
    cases = suite if isinstance(suite, list) else suite.get("cases", [])
    case = next((c for c in cases if c.get("id") == case_id), None)
    if case is None:
        raise ValueError(f"case {case_id!r} not found in {suite_path}")

    # 2. Build inputs via make_inputs(case). Extract B/H/NT from shape env.
    inputs = test_inputs_mod.make_inputs(case)
    shape_env = case.get("shape", {})
    B = int(shape_env.get("B", 1))
    H = int(shape_env.get("H", 1))
    T = int(shape_env.get("T", shape_env.get("S", 0)))
    BT = int(shape_env.get("BT", 1))
    NT = max(1, T // BT) if T else int(shape_env.get("NT", 1))

    # Resolve intermediate shapes via a simple env lookup. Anything more
    # elaborate should land in test_inputs.py's _resolve_shape helper.
    def _resolve(s, env):
        if isinstance(s, int):
            return s
        if isinstance(s, str) and s in env:
            return int(env[s])
        try:
            return int(eval(str(s), {"__builtins__": {}}, env))
        except Exception as e:
            raise ValueError(f"cannot resolve shape token {s!r} with env keys {list(env.keys())}") from e

    env = {"B": B, "H": H, "T": T, "NT": NT, "BT": BT, **shape_env}

    # 3. Run impl under each mode, collect inspection buffers.
    per_mode_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
    per_mode_log: Dict[str, str] = {}
    per_mode_status: Dict[str, str] = {}
    device_str = f"npu:{device}" if os.environ.get("ASCEND_HOME_PATH") else "cpu"

    for mode in modes:
        # Allocate fresh buffers per mode (device memory or cpu).
        alloc_device = device_str if mode != "torch" else "cpu"
        buffers = _alloc_inspection_buffers(
            m["intermediates"], B, H, NT, _resolve, env,
            alloc_device, torch.float32,
        )
        status, cpu_buffers, log = _run_mode(impl_mod, mode, inputs, buffers)
        per_mode_status[mode] = status
        per_mode_log[mode] = log
        per_mode_buffers[mode] = cpu_buffers
        if verbose:
            logging.info("[%-5s] status=%s  %s", mode, status, log[:180])

    # 4. Run golden inspect for reference.
    inspect_fn_name = f"module_{int(module[1:])}_inspect"
    if not hasattr(golden_mod, inspect_fn_name):
        raise AttributeError(
            f"{golden_path.name} does not define {inspect_fn_name}. "
            "Either the generator failed, or @pypto-op-debugger removed the stub — "
            "restore it and fill in per-iteration capture per the manifest."
        )
    gold_intermediates = getattr(golden_mod, inspect_fn_name)(**inputs)

    # 5. Per-iteration compare for each mode × each intermediate.
    report: Dict[str, Any] = {
        "op": op, "module": module, "case": case_id,
        "modes": modes, "per_mode_status": per_mode_status,
        "per_mode_log": per_mode_log,
        "intermediates": {},
    }
    for entry in m["intermediates"]:
        name = entry["name"]
        has_iter = entry["probe_point"] == "inside_nt_loop"
        gold_tensor = gold_intermediates.get(name)
        if gold_tensor is None:
            report["intermediates"][name] = {
                "status": "GOLDEN_STUB_UNFILLED",
                "note": "module_<k>_inspect returned None for this intermediate. "
                        "@pypto-op-debugger must fill in the per-iteration capture before bisecting.",
            }
            if verbose:
                logging.info("  [%s] golden stub not filled — skipping compare", name)
            continue

        entry_report: Dict[str, Any] = {"per_mode": {}}
        for mode in modes:
            if per_mode_status[mode] not in ("OK",):
                entry_report["per_mode"][mode] = {"status": per_mode_status[mode],
                                                   "per_iter": [], "drift_onset": None}
                continue
            impl_tensor = per_mode_buffers[mode].get(f"inspection_{name}")
            if impl_tensor is None:
                entry_report["per_mode"][mode] = {"status": "NO_BUFFER",
                                                   "per_iter": [], "drift_onset": None}
                continue
            per_iter = _compare_per_iter(impl_tensor, gold_tensor, m["atol"], m["rtol"], has_iter)
            onset = _drift_onset(per_iter)
            entry_report["per_mode"][mode] = {
                "status": "OK",
                "per_iter": per_iter,
                "drift_onset": onset,
                "max_abs_diff": max(r["max_abs_diff"] for r in per_iter),
                "max_rel_diff": max(r["max_rel_diff"] for r in per_iter),
            }
        report["intermediates"][name] = entry_report

    # 6. Pretty-print drift-onset summary.
    if verbose:
        logging.info("\n%s", "=" * 72)
        logging.info("Drift-onset summary — %s/%s, case %s", op, module, case_id)
        logging.info("=" * 72)
        header = "{:<28s} " + " ".join("{:>16s}".format(m) for m in modes)
        logging.info(header.format("intermediate", *modes))
        for name, entry_report in report["intermediates"].items():
            if "per_mode" not in entry_report:
                logging.info("%-28s [%s]", name, entry_report.get("status"))
                continue
            cells = []
            for mode in modes:
                pm = entry_report["per_mode"].get(mode, {})
                if pm.get("status") != "OK":
                    cells.append(f"[{pm.get('status', '?')[:14]}]")
                else:
                    onset = pm.get("drift_onset")
                    if onset is None:
                        cells.append("PASS all")
                    else:
                        cells.append(f"FAIL@iter={onset}")
            row = "{:<28s} ".format(name) + " ".join("{:>16s}".format(c) for c in cells)
            logging.info(row)
        logging.info("=" * 72)
        logging.info("Drift-onset interpretation:")
        logging.info("  * If intermediate X drifts first at iter K under `npu` but passes under `sim`,")
        logging.info("    the bug is NPU-specific in the expression that produces X (tile/pipe/memory).")
        logging.info("  * If all intermediates drift under both modes at the same iter, the bug is in")
        logging.info("    the shared expression upstream — narrow by eyeballing which expression they all depend on.")

    # 7. Write JSON report.
    report_path = debug_dir / "snapshot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        logging.info("\nreport written: %s", report_path)
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Per-iteration snapshot bisection runner.")
    ap.add_argument("--manifest", required=True, help="path to custom/<op>/_debug/snapshot_manifest.yaml")
    ap.add_argument("--modes", default=None, help="comma-separated subset of {sim,npu} (default: from manifest)")
    ap.add_argument("--device", default="9", help="NPU device id (default: 9)")
    ap.add_argument("--quiet", action="store_true", help="suppress stdout prints; still writes JSON report")
    args = ap.parse_args()
    modes = args.modes.split(",") if args.modes else None
    try:
        run(args.manifest, modes_override=modes, device=args.device, verbose=not args.quiet)
    except Exception as e:
        logging.error("[snapshot_bisect FAILED] %s", e)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
