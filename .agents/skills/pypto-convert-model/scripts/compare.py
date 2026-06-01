"""Run a converted artifact and compare its output against the original."""
from pathlib import Path
import numpy as np
import torch

from device_util import pick_device, to_device, onnxruntime_providers


def _as_args(sample):
    if isinstance(sample, tuple):
        return sample
    if isinstance(sample, dict):
        return tuple(sample.values())
    return (sample,)


def _names(sample, meta):
    if isinstance(sample, tuple):
        return meta.get("input_names") or [f"arg{i}" for i in range(len(sample))]
    if isinstance(sample, dict):
        return list(sample.keys())
    return meta.get("input_names") or ["input"]


def _to_numpy_args(sample) -> list:
    return [a.cpu().numpy() for a in _as_args(sample)]


def _to_numpy_safe(t: torch.Tensor) -> np.ndarray:
    """Numpy doesn't grok bfloat16 / float8; cast to float32 before exporting."""
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return t.cpu().numpy()


def _flatten_output(out) -> np.ndarray:
    """Reduce model output to a single 1-D numpy array for diff."""
    if isinstance(out, np.ndarray):
        x = out
    elif isinstance(out, torch.Tensor):
        x = _to_numpy_safe(out)
    elif isinstance(out, (list, tuple)):
        x = _flatten_output(out[0])
    elif hasattr(out, "logits"):
        x = _to_numpy_safe(out.logits)
    elif hasattr(out, "last_hidden_state"):
        x = _to_numpy_safe(out.last_hidden_state)
    elif isinstance(out, dict) and out:
        x = _flatten_output(next(iter(out.values())))
    else:
        raise TypeError(f"unknown output type {type(out)}")
    return np.asarray(x).ravel()


def _model_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "out of memory", "cuda out of memory", "no_memory", "nv_err_no_memory",
        "cudnn_status_alloc", "cublas_status_alloc",
    ))


def reference_output(model, sample) -> np.ndarray:
    """Run the model on the best available accelerator and restore its device.

    Restoring matters because the same Module instance is later handed to the
    converters (ONNX / TorchScript), which assume CPU placement.
    On accelerator OOM (common for large models on unified-memory hosts) fall
    back to CPU so the conversion run can continue.
    """
    device, _ = pick_device()
    orig = _model_device(model)
    if device == orig:
        with torch.no_grad():
            return _flatten_output(model(*_as_args(sample)))

    try:
        model_on_device = to_device(model, device)
        args = to_device(_as_args(sample), device)
        with torch.no_grad():
            out = model_on_device(*args)
        result = _flatten_output(out)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
        if not _is_oom(e):
            raise
        # Restore to CPU and run there.
        try:
            model.to(orig)
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            return _flatten_output(model(*_as_args(sample)))
    finally:
        try:
            model.to(orig)
        except Exception:
            pass
    return result


def run_onnx(path: Path, sample, meta=None) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=onnxruntime_providers())
    arrs = _to_numpy_args(sample)
    names_meta = _names(sample, meta or {})
    inames = [i.name for i in sess.get_inputs()]
    feed = {}
    for i, name in enumerate(inames):
        if i < len(arrs):
            feed[name] = arrs[i]
    out = sess.run(None, feed)
    return _flatten_output(out[0])


def run_pt(path: Path, sample, meta=None) -> np.ndarray:
    device, _ = pick_device()
    try:
        m = torch.jit.load(str(path), map_location=device).eval()
        args = to_device(_as_args(sample), device)
    except Exception:
        # If the TorchScript artifact can't be moved to the device, fall back to CPU
        m = torch.jit.load(str(path), map_location="cpu").eval()
        args = _as_args(sample)
    with torch.no_grad():
        out = m(*args)
    return _flatten_output(out)


def run_safetensors(path: Path, model_template, sample, meta=None) -> np.ndarray:
    from safetensors.torch import load_model
    load_model(model_template, str(path), strict=False)
    model_template.eval()
    return reference_output(model_template, sample)


def diff(a: np.ndarray, b: np.ndarray, atol=1e-4, rtol=1e-3) -> dict:
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    abs_diff = np.abs(a - b)
    rel = abs_diff / (np.abs(a) + 1e-9)
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel.max()),
        "size": int(n),
        "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
    }
