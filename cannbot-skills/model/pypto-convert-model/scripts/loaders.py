# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load source models (HF / timm / toy) into a uniform interface.

Returns (model, sample_input, meta) where:
  - model: torch.nn.Module in eval mode on CPU
  - sample_input: a torch tensor or dict suitable for model(**) or model()
  - meta: dict with kind, hf_dir (if HF), and category
"""
import os
import sys
from importlib import import_module
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
REGISTRY = import_module("registry").REGISTRY
build_toy = import_module("toy_moe").build_toy


class TensorOutputAdapter(torch.nn.Module):
    """Strip HF dict/dataclass outputs to bare logits tensor for cleaner export."""
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


def _hf_image_classifier(repo: str, cache_dir: Path):
    transformers = import_module("transformers")
    proc = transformers.AutoImageProcessor.from_pretrained(
        repo, cache_dir=str(cache_dir),
    )
    inner = transformers.AutoModelForImageClassification.from_pretrained(
        repo, cache_dir=str(cache_dir), torch_dtype=torch.float32,
    )
    inner.eval()
    model = TensorOutputAdapter(inner).eval()
    img = torch.rand(1, 3, 224, 224)
    return model, img, {"hf_dir": str(cache_dir), "processor": proc}


class _Seq2SeqLogits(torch.nn.Module):
    """Fixed positional signature: forward(input_ids, decoder_input_ids) -> logits."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, decoder_input_ids):
        out = self.m(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return out.logits

    def load_state_dict(self, state, **kw):
        return self.m.load_state_dict(state, **kw)

    def state_dict(self, *a, **kw):
        return self.m.state_dict(*a, **kw)


class _CausalLogits(torch.nn.Module):
    """Fixed positional signature: forward(input_ids) -> logits."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids=input_ids)
        return out.logits

    def load_state_dict(self, state, **kw):
        return self.m.load_state_dict(state, **kw)

    def state_dict(self, *a, **kw):
        return self.m.state_dict(*a, **kw)


def _hf_seq2seq(repo: str, cache_dir: Path, prompt: str):
    transformers = import_module("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(repo, cache_dir=str(cache_dir))
    inner = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        repo, cache_dir=str(cache_dir), torch_dtype=torch.float32,
    )
    inner.eval()
    model = _Seq2SeqLogits(inner).eval()
    enc = tok(prompt, return_tensors="pt")
    dec = tok("", return_tensors="pt").input_ids
    # Tuple sample so converters use positional path
    sample = (enc.input_ids, dec)
    return model, sample, {"hf_dir": str(cache_dir), "tokenizer": tok,
                           "input_names": ["input_ids", "decoder_input_ids"]}


def _hf_causal(repo: str, cache_dir: Path, prompt: str):
    transformers = import_module("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(
        repo, cache_dir=str(cache_dir), trust_remote_code=True,
    )
    # On unified-memory systems (e.g. NVIDIA GB10) loading 14B params in fp32 and
    # then copying to GPU exhausts the shared CPU+GPU RAM pool. Default to fp16
    # for causal LMs; override via CONVERT_EXP_DTYPE=float32 if you have the budget.
    dtype_env = os.environ.get("CONVERT_EXP_DTYPE", "float16").strip().lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(dtype_env, torch.float16)
    inner = transformers.AutoModelForCausalLM.from_pretrained(
        repo, cache_dir=str(cache_dir), trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    inner.eval()
    model = _CausalLogits(inner).eval()
    enc = tok(prompt, return_tensors="pt")
    sample = (enc.input_ids,)
    return model, sample, {"hf_dir": str(cache_dir), "tokenizer": tok,
                           "input_names": ["input_ids"]}


def _timm(repo: str, cache_dir: Path):
    timm = import_module("timm")
    name = repo.split("/", 1)[1] if "/" in repo else repo
    model = timm.create_model(name, pretrained=True, cache_dir=str(cache_dir))
    model.eval()
    img = torch.rand(1, 3, 224, 224)
    return model, img, {"timm_name": name}


def load(model_id: str, cache_root: Path):
    cfg = REGISTRY[model_id]
    cache_dir = cache_root / cfg["category"] / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg["source"] == "toy":
        model, sample = build_toy(cfg["repo"])
        return model, sample, {"category": cfg["category"], "kind": "tensor"}
    if cfg["loader"] == "image-classification":
        m, s, meta = _hf_image_classifier(cfg["repo"], cache_dir)
    elif cfg["loader"] == "seq2seq-lm":
        m, s, meta = _hf_seq2seq(cfg["repo"], cache_dir, cfg["input"]["prompt"])
    elif cfg["loader"] == "causal-lm":
        m, s, meta = _hf_causal(cfg["repo"], cache_dir, cfg["input"]["prompt"])
    elif cfg["loader"] == "timm":
        m, s, meta = _timm(cfg["repo"], cache_dir)
    else:
        raise ValueError(f"unknown loader {cfg['loader']}")
    meta["category"] = cfg["category"]
    meta["kind"] = cfg["input"]["kind"]
    return m, s, meta
