# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Format converters. Each function takes (model, sample, out_dir, meta)
and writes the artifact, returning the file/dir path of the result.

Convention: raise on failure with a clear exception message; the runner
catches and logs.
"""
from importlib import import_module
from pathlib import Path

import torch


def _normalize(sample, meta):
    """Return (args_tuple, names_list)."""
    if isinstance(sample, tuple):
        names = meta.get("input_names") or [f"arg{i}" for i in range(len(sample))]
        return sample, list(names)
    if isinstance(sample, dict):
        return tuple(sample.values()), list(sample.keys())
    return (sample,), (meta.get("input_names") or ["input"])


def to_safetensors(model, sample, out_dir: Path, meta: dict) -> Path:
    """Use save_model so weight-tied tensors don't trip safetensors."""
    save_model = import_module("safetensors.torch").save_model
    out = out_dir / "model.safetensors"
    save_model(model, str(out))
    return out


def to_pt(model, sample, out_dir: Path, meta: dict) -> Path:
    out = out_dir / "model.pt"
    args, _ = _normalize(sample, meta)
    try:
        traced = torch.jit.trace(model, args, strict=False)
    except Exception:
        traced = torch.jit.script(model)
    traced.save(str(out))
    return out


def to_onnx(model, sample, out_dir: Path, meta: dict) -> Path:
    out = out_dir / "model.onnx"
    args, names = _normalize(sample, meta)
    output_names = ["output"]
    dynamic_axes = {n: {0: "batch"} for n in names + output_names}
    # torch>=2.10 defaults to the dynamo exporter which fails on data-dependent
    # control flow (e.g. MoE routing). Pin to the legacy tracing exporter.
    with torch.no_grad():
        try:
            torch.onnx.export(
                model, args, str(out),
                input_names=names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model, args, str(out),
                input_names=names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    return out


CONVERTERS = {
    "safetensors": to_safetensors,
    "pt": to_pt,
    "onnx": to_onnx,
}
