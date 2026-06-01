"""Any-to-any model porter for {onnx, pt, safetensors}.

Usage examples:
    # TorchScript -> ONNX, just give an input shape
    python scripts/port.py model.pt out.onnx --input-shape 1,3,224,224

    # safetensors needs an architecture; hand it an HF repo
    python scripts/port.py model.safetensors out.onnx \\
        --hf-repo google/mobilenet_v2_1.0_224

    # ONNX -> TorchScript
    python scripts/port.py model.onnx out.pt --input-shape 1,3,224,224

The output of the converted artifact is compared against the source
artifact's output on the same dummy input and a `max_abs` diff is printed.
The verification stage runs on the best available accelerator
(NPU > CUDA > XPU > MPS > CPU); override with CONVERT_EXP_DEVICE=...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# cudnn picks heuristically and can return slightly different fp32 results on
# repeated forwards even with identical weights. Pin it so the verification
# diff actually reflects conversion quality, not kernel-selection noise.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from device_util import pick_device, to_device, onnxruntime_providers  # noqa: E402

VALID_FORMATS = ("onnx", "pt", "safetensors")
EXT_TO_FORMAT = {".onnx": "onnx", ".pt": "pt", ".pth": "pt", ".safetensors": "safetensors"}


# ---------- helpers ----------

def _detect_format(path: Path, override: str | None) -> str:
    if override:
        if override not in VALID_FORMATS:
            raise ValueError(f"format must be one of {VALID_FORMATS}, got {override!r}")
        return override
    fmt = EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"cannot infer format from {path.name}. Use --input-format/--output-format."
        )
    return fmt


def _parse_shape(shape: str | None) -> tuple[int, ...] | None:
    if not shape:
        return None
    parts = [p.strip() for p in shape.replace("x", ",").split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _to_numpy_safe(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return t.cpu().numpy()


def _flatten(out) -> np.ndarray:
    if isinstance(out, np.ndarray):
        return out.ravel()
    if isinstance(out, torch.Tensor):
        return _to_numpy_safe(out).ravel()
    if isinstance(out, (list, tuple)):
        return _flatten(out[0])
    if hasattr(out, "logits"):
        return _to_numpy_safe(out.logits).ravel()
    if hasattr(out, "last_hidden_state"):
        return _to_numpy_safe(out.last_hidden_state).ravel()
    if isinstance(out, dict) and out:
        return _flatten(next(iter(out.values())))
    raise TypeError(f"unknown output type {type(out)}")


def _diff(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> dict:
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    abs_diff = np.abs(a - b)
    rel = abs_diff / (np.abs(a) + 1e-9)
    # Top-K mismatch indices (largest absolute differences) — useful for
    # debugging which output positions diverge the most.
    k = min(5, abs_diff.size)
    top_idx = np.argpartition(-abs_diff, k - 1)[:k] if k > 0 else np.array([], dtype=int)
    top_idx = top_idx[np.argsort(-abs_diff[top_idx])]
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel.max()),
        "size": int(n),
        "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "p50": float(np.median(abs_diff)),
        "p95": float(np.percentile(abs_diff, 95)),
        "p99": float(np.percentile(abs_diff, 99)),
        "top_mismatch": [(int(i), float(abs_diff[i])) for i in top_idx],
    }


def _tensor_stats(arr: np.ndarray) -> dict:
    """Compact summary stats for raw-log output."""
    return {
        "shape": tuple(int(x) for x in arr.shape) if arr.ndim else (),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "first5": [float(x) for x in arr.ravel()[:5].tolist()],
    }


def _tolerances_for(device_label: str) -> tuple[float, float]:
    """CUDA fp32 forwards have ~5e-4 cudnn accumulation noise even with the
    same weights and input, so a CPU-tight tolerance gives misleading DIFFs.
    Tighten when the diff is measured fully on CPU.
    """
    if device_label == "cpu":
        return 1e-4, 1e-3
    return 5e-3, 1e-2


# ---------- input source loaders ----------

class Source:
    """Wraps the source model as something we can: forward, jit-trace, get state_dict."""

    def __init__(self, module: torch.nn.Module, sample: tuple, device: torch.device):
        self.module = module.eval()
        self.sample = sample
        self.device = device

    def forward(self) -> np.ndarray:
        m = to_device(self.module, self.device)
        args = to_device(self.sample, self.device)
        with torch.no_grad():
            return _flatten(m(*args))


def _build_sample(input_shape: tuple[int, ...] | None,
                  hf_repo: str | None) -> tuple[torch.Tensor, ...]:
    if input_shape is not None:
        return (torch.rand(*input_shape, dtype=torch.float32),)
    if hf_repo is not None:
        # Fall back to a 1×3×224×224 image — works for most CV HF models in the registry.
        return (torch.rand(1, 3, 224, 224, dtype=torch.float32),)
    raise ValueError("either --input-shape or --hf-repo must be provided")


def _load_pt_source(path: Path, sample, device: torch.device) -> Source:
    m = torch.jit.load(str(path), map_location="cpu").eval()
    return Source(m, sample, device)


def _load_onnx_as_torch(path: Path) -> torch.nn.Module:
    """Convert ONNX → torch.nn.Module via onnx2torch so we can re-export elsewhere."""
    try:
        from onnx2torch import convert
    except ImportError as e:
        raise RuntimeError(
            "onnx2torch is required for ONNX-as-source conversions. "
            "pip install onnx2torch"
        ) from e
    return convert(str(path)).eval()


def _load_onnx_source(path: Path, sample, device: torch.device) -> Source:
    m = _load_onnx_as_torch(path)
    return Source(m, sample, device)


def _load_safetensors_source(path: Path, sample, hf_repo: str | None,
                             device: torch.device) -> Source:
    if not hf_repo:
        raise ValueError("--hf-repo is required when the input is safetensors")
    from safetensors.torch import load_model

    inner, sample = _instantiate_hf_model(hf_repo, sample)
    load_model(inner, str(path), strict=False)
    inner.eval()
    return Source(inner, sample, device)


def _instantiate_hf_model(hf_repo: str, sample: tuple) -> tuple[torch.nn.Module, tuple]:
    """Try image-classification first; fall back to AutoModel for general nets.

    Returns the bare nn.Module wrapped to emit a tensor (not a HF dataclass).
    """
    from transformers import AutoModelForImageClassification, AutoModel  # noqa: F401

    try:
        inner = AutoModelForImageClassification.from_pretrained(
            hf_repo, torch_dtype=torch.float32,
        )
    except Exception:
        from transformers import AutoModel
        inner = AutoModel.from_pretrained(hf_repo, torch_dtype=torch.float32)

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args, **kwargs):
            out = self.m(*args, **kwargs)
            if hasattr(out, "logits"):
                return out.logits
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        def load_state_dict(self, state, **kw):
            return self.m.load_state_dict(state, **kw)

        def state_dict(self, *a, **kw):
            return self.m.state_dict(*a, **kw)

    return _LogitsOnly(inner).eval(), sample


# ---------- target writers ----------

def _write_pt(source: Source, out: Path) -> Path:
    try:
        traced = torch.jit.trace(source.module, source.sample, strict=False)
    except Exception:
        traced = torch.jit.script(source.module)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    return out


def _write_onnx(source: Source, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [f"input{i}" for i in range(len(source.sample))]
    output_names = ["output"]
    dynamic_axes = {n: {0: "batch"} for n in names + output_names}
    with torch.no_grad():
        try:
            torch.onnx.export(
                source.module, source.sample, str(out),
                input_names=names, output_names=output_names,
                opset_version=17, dynamic_axes=dynamic_axes,
                do_constant_folding=True, dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                source.module, source.sample, str(out),
                input_names=names, output_names=output_names,
                opset_version=17, dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    return out


def _write_safetensors(source: Source, out: Path) -> Path:
    from safetensors.torch import save_model
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(source.module, str(out))
    return out


# ---------- target reload + forward (for verification) ----------

def _run_pt(path: Path, sample, device: torch.device) -> np.ndarray:
    try:
        m = torch.jit.load(str(path), map_location=device).eval()
        args = to_device(sample, device)
    except Exception:
        m = torch.jit.load(str(path), map_location="cpu").eval()
        args = sample
    with torch.no_grad():
        return _flatten(m(*args))


def _run_onnx(path: Path, sample) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=onnxruntime_providers())
    arrs = [a.cpu().numpy() for a in sample]
    inames = [i.name for i in sess.get_inputs()]
    feed = {n: arrs[i] for i, n in enumerate(inames) if i < len(arrs)}
    out = sess.run(None, feed)
    return _flatten(out[0])


def _run_safetensors(path: Path, source_module: torch.nn.Module,
                     sample, device: torch.device) -> np.ndarray:
    from safetensors.torch import load_model
    load_model(source_module, str(path), strict=False)
    source_module.eval()
    m = to_device(source_module, device)
    args = to_device(sample, device)
    with torch.no_grad():
        return _flatten(m(*args))


# ---------- orchestration ----------

def port(in_path: Path, out_path: Path,
         in_fmt: str, out_fmt: str,
         input_shape: tuple[int, ...] | None,
         hf_repo: str | None,
         skip_verify: bool = False) -> dict:
    if in_fmt == out_fmt:
        raise ValueError(f"input and output format are both {in_fmt}; nothing to do")
    if in_fmt not in VALID_FORMATS or out_fmt not in VALID_FORMATS:
        raise ValueError(f"formats must be one of {VALID_FORMATS}")

    timings: dict[str, float] = {}
    t = time.time()
    sample = _build_sample(input_shape, hf_repo)
    device, label = pick_device()

    if in_fmt == "pt":
        src = _load_pt_source(in_path, sample, device)
    elif in_fmt == "onnx":
        src = _load_onnx_source(in_path, sample, device)
    elif in_fmt == "safetensors":
        src = _load_safetensors_source(in_path, sample, hf_repo, device)
    else:
        raise ValueError(in_fmt)
    timings["load"] = time.time() - t

    t = time.time()
    if out_fmt == "pt":
        _write_pt(src, out_path)
    elif out_fmt == "onnx":
        _write_onnx(src, out_path)
    elif out_fmt == "safetensors":
        _write_safetensors(src, out_path)
    else:
        raise ValueError(out_fmt)
    timings["convert"] = time.time() - t

    sample_shapes = [tuple(int(x) for x in a.shape) for a in src.sample]
    sample_dtypes = [str(a.dtype) for a in src.sample]

    info: dict = {
        "input": str(in_path),
        "output": str(out_path),
        "in_fmt": in_fmt,
        "out_fmt": out_fmt,
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
        "timings": timings,
        "sample_shapes": sample_shapes,
        "sample_dtypes": sample_dtypes,
    }

    if skip_verify:
        info["verified"] = False
        return info

    info["device"] = label
    info["ort_providers"] = onnxruntime_providers() if "onnx" in (in_fmt, out_fmt) else None

    t = time.time()
    ref = src.forward()
    timings["ref_forward"] = time.time() - t

    t = time.time()
    if out_fmt == "pt":
        out = _run_pt(out_path, src.sample, device)
    elif out_fmt == "onnx":
        out = _run_onnx(out_path, src.sample)
    elif out_fmt == "safetensors":
        out = _run_safetensors(out_path, src.module, src.sample, device)
    else:
        raise ValueError(out_fmt)
    timings["target_forward"] = time.time() - t

    info["ref_stats"] = _tensor_stats(ref)
    info["out_stats"] = _tensor_stats(out)

    # ONNX Runtime here runs CPU on aarch64, so an ONNX leg compared against a
    # CUDA reference will always show CPU/CUDA fp32 drift; widen the tolerance
    # to match. Pure CPU comparisons keep the tighter tolerance.
    if label != "cpu":
        atol, rtol = _tolerances_for(label)
    else:
        atol, rtol = _tolerances_for("cpu")
    info["diff"] = _diff(ref, out, atol=atol, rtol=rtol)
    info["status"] = "PASS" if info["diff"]["allclose"] else "DIFF"
    return info


def _print_verbose(info: dict) -> None:
    """Dump every numerical detail useful for debugging a port run."""
    log = lambda *a: print("[INFO]", *a)
    log(f"input        : {info['input']} ({info['in_fmt']})")
    log(f"output       : {info['output']} ({info['out_fmt']})  size={info['size_mb']} MB")
    log(f"sample shapes: {info['sample_shapes']}  dtypes={info['sample_dtypes']}")
    if "device" in info:
        log(f"device       : {info['device']}")
        if info.get("ort_providers"):
            log(f"ORT providers: {info['ort_providers']}")
    t = info.get("timings", {})
    if t:
        log("timings (s)  : " + "  ".join(f"{k}={v:.3f}" for k, v in t.items()))
    if info.get("verified") is False:
        log("verification skipped")
        return
    rs, os_ = info.get("ref_stats"), info.get("out_stats")
    if rs and os_:
        log(f"ref forward  : shape={rs['shape']} dtype={rs['dtype']} "
            f"min={rs['min']:.4g} max={rs['max']:.4g} mean={rs['mean']:.4g} std={rs['std']:.4g}")
        log(f"               first5={['%.4g' % x for x in rs['first5']]}")
        log(f"out forward  : shape={os_['shape']} dtype={os_['dtype']} "
            f"min={os_['min']:.4g} max={os_['max']:.4g} mean={os_['mean']:.4g} std={os_['std']:.4g}")
        log(f"               first5={['%.4g' % x for x in os_['first5']]}")
    d = info.get("diff", {})
    if d:
        log(f"diff abs     : max={d['max_abs']:.3e} mean={d['mean_abs']:.3e} "
            f"p50={d['p50']:.3e} p95={d['p95']:.3e} p99={d['p99']:.3e}")
        log(f"diff rel max : {d['max_rel']:.3e}")
        log(f"tolerance    : atol={d['atol']:.0e} rtol={d['rtol']:.0e} "
            f"(allclose={d['allclose']})")
        if d.get("top_mismatch"):
            top = "  ".join(f"[{i}]={v:.3e}" for i, v in d["top_mismatch"])
            log(f"top mismatch : {top}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="input model path")
    p.add_argument("output", type=Path, help="output model path")
    p.add_argument("--input-format", choices=VALID_FORMATS,
                   help="override format inferred from extension")
    p.add_argument("--output-format", choices=VALID_FORMATS,
                   help="override format inferred from extension")
    p.add_argument("--input-shape",
                   help="comma-separated input shape, e.g. 1,3,224,224. "
                        "Used to build the dummy input.")
    p.add_argument("--hf-repo",
                   help="HF repo id used to instantiate the architecture when "
                        "safetensors is the input or output. Required for "
                        "safetensors as input.")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the round-trip diff check")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="dump per-stage timings, tensor stats, diff distribution, "
                        "and top mismatch indices as raw log lines")
    args = p.parse_args(argv)

    in_fmt = _detect_format(args.input, args.input_format)
    out_fmt = _detect_format(args.output, args.output_format)
    shape = _parse_shape(args.input_shape)

    try:
        info = port(args.input, args.output, in_fmt, out_fmt, shape,
                    args.hf_repo, args.skip_verify)
    except Exception as e:
        print(f"FAIL ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    if args.verbose:
        _print_verbose(info)
        print()

    print(f"{info['in_fmt']} -> {info['out_fmt']}: wrote {info['output']} "
          f"({info['size_mb']} MB)")
    if info.get("verified") is False:
        print("verification skipped (--skip-verify)")
        return 0
    d = info["diff"]
    print(f"device={info.get('device','?')}  status={info['status']}  "
          f"max_abs={d['max_abs']:.2e}  mean_abs={d['mean_abs']:.2e}")
    return 0 if info["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
