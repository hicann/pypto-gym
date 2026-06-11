"""Main runner: model -> all target formats -> tested.

Usage:
    python run.py [model_id ...]   # default: all models in registry

Paths:
    Frozen test evidence (committed to repo):
        ROOT/references/matrix.md
        ROOT/references/results/<id>.json
    Runtime artifacts (gitignored; default ~/.cache/pypto-convert-model,
    override with $CONVERT_MODEL_WORKDIR):
        WORKDIR/models/   HF/timm download cache
        WORKDIR/outputs/  converted artifacts
        WORKDIR/logs/     master + per-conversion logs
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from registry import REGISTRY, TARGET_FORMATS  # noqa: E402
from log_util import master_logger, conversion_logger, timed  # noqa: E402
from loaders import load  # noqa: E402
from converters import CONVERTERS  # noqa: E402
import compare as cmp  # noqa: E402
from device_util import pick_device, onnxruntime_providers  # noqa: E402

WORKDIR = Path(os.environ.get(
    "CONVERT_MODEL_WORKDIR",
    str(Path.home() / ".cache" / "pypto-convert-model"),
))
CACHE = WORKDIR / "models"
OUT = WORKDIR / "outputs"
RESULTS = ROOT / "references" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MATRIX = ROOT / "references" / "matrix.md"


def run_target(target: Path, fmt: str, sample, model_template, meta):
    if fmt == "onnx":
        return cmp.run_onnx(target, sample, meta)
    if fmt == "pt":
        return cmp.run_pt(target, sample, meta)
    if fmt == "safetensors":
        return cmp.run_safetensors(target, model_template, sample, meta)
    raise ValueError(fmt)


def _convert_and_test_single_format(model_id: str, fmt: str, model, sample, ref, meta, out_root: Path) -> dict:
    clog = conversion_logger(model_id, fmt)
    out_dir = out_root / fmt
    out_dir.mkdir(exist_ok=True)
    result = {"status": "PENDING"}
    t0 = time.time()
    try:
        with timed(clog, f"convert {model_id} -> {fmt}"):
            path = CONVERTERS[fmt](model, sample, out_dir, meta)
        size_mb = path.stat().st_size / 1e6 if path.is_file() else sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file()
        ) / 1e6
        result["convert_seconds"] = round(time.time() - t0, 2)
        try:
            result["artifact"] = str(path.relative_to(WORKDIR))
        except ValueError:
            result["artifact"] = str(path)
        result["size_mb"] = round(size_mb, 2)

        # Test stage
        try:
            with timed(clog, f"test {model_id} <- {fmt}"):
                # For safetensors, need a fresh template instance
                if fmt == "safetensors":
                    template, _, _ = load(model_id, CACHE)
                    out = run_target(path, fmt, sample, template, meta)
                    del template
                else:
                    out = run_target(path, fmt, sample, None, meta)
                d = cmp.diff(ref, out)
                result["diff"] = d
                result["status"] = "PASS" if d["allclose"] else "DIFF"
        except NotImplementedError as e:
            result["status"] = "TEST_SKIP"
            result["error"] = str(e)
            clog.warning(f"test skipped: {e}")
        except Exception as e:
            result["status"] = "TEST_FAIL"
            result["error"] = repr(e)
            clog.exception(f"test failed: {e}")
    except NotImplementedError as e:
        result["status"] = "SKIP"
        result["error"] = str(e)
        clog.warning(f"skipped: {e}")
    except Exception as e:
        result["status"] = "CONVERT_FAIL"
        result["error"] = repr(e)
        clog.exception(f"convert failed: {e}")
    return result


def process_model(model_id: str) -> dict:
    mlog = master_logger()
    mlog.info(f"=== {model_id} ===")
    cfg = REGISTRY[model_id]
    out_root = OUT / model_id
    out_root.mkdir(parents=True, exist_ok=True)
    results = {fmt: {"status": "PENDING"} for fmt in TARGET_FORMATS}

    # Load source once
    try:
        with timed(mlog, f"load {model_id}"):
            model, sample, meta = load(model_id, CACHE)
        ref = cmp.reference_output(model, sample)
    except Exception as e:
        mlog.exception(f"LOAD FAILED for {model_id}: {e}")
        for fmt in TARGET_FORMATS:
            results[fmt] = {"status": "LOAD_FAIL", "error": str(e)}
        return results

    for fmt in TARGET_FORMATS:
        results[fmt] = _convert_and_test_single_format(
            model_id, fmt, model, sample, ref, meta, out_root,
        )
        gc.collect()
    # Write per-model JSON
    (RESULTS / f"{model_id}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Free the loaded model
    del model
    gc.collect()
    return results


STATUS_EMOJI = {
    "PASS": "OK",
    "DIFF": "DIFF",
    "SKIP": "SKIP",
    "CONVERT_FAIL": "C-FAIL",
    "TEST_FAIL": "T-FAIL",
    "TEST_SKIP": "T-SKIP",
    "LOAD_FAIL": "L-FAIL",
    "PENDING": "?",
}


MATRIX_FORMATS = TARGET_FORMATS


def write_matrix(all_results: dict):
    _, dev_label = pick_device()
    lines = ["# Conversion Matrix",
             "",
             f"Test device (auto-detected): **{dev_label}**",
             "",
             "| model | " + " | ".join(MATRIX_FORMATS) + " |",
             "|---" + "|---" * len(MATRIX_FORMATS) + "|"]
    for mid in REGISTRY:
        if mid not in all_results:
            continue
        row = [mid]
        for fmt in MATRIX_FORMATS:
            r = all_results[mid].get(fmt, {})
            tag = STATUS_EMOJI.get(r.get("status", "PENDING"), "?")
            extra = ""
            d = r.get("diff")
            if d:
                extra = f"<br/>max_abs={d['max_abs']:.2e}"
            row.append(f"{tag}{extra}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Legend: OK=pass, DIFF=converted but output differs, "
                 "SKIP=not applicable, C-FAIL=convert failed, T-FAIL=test failed, "
                 "T-SKIP=test skipped, L-FAIL=load failed.")
    MATRIX.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help="model ids (default: all)")
    args = p.parse_args()
    targets = args.models or list(REGISTRY.keys())

    dev, label = pick_device()
    master_logger().info(f"test device: {label} ({dev}); ORT providers: {onnxruntime_providers()}")

    all_results = {}
    # Load any existing per-model results so partial progress accumulates
    for jp in RESULTS.glob("*.json"):
        try:
            all_results[jp.stem] = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            pass

    for mid in targets:
        try:
            all_results[mid] = process_model(mid)
        except KeyboardInterrupt:
            master_logger().warning("Interrupted by user")
            break
        except Exception as e:
            master_logger().exception(f"unexpected error for {mid}: {e}")
            all_results[mid] = {fmt: {"status": "LOAD_FAIL", "error": traceback.format_exc()}
                                for fmt in TARGET_FORMATS}
        write_matrix(all_results)

    write_matrix(all_results)
    master_logger().info(f"Matrix written to {MATRIX}")


if __name__ == "__main__":
    main()
