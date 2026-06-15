#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax M2.7 single-prompt inference on Ascend 910B.

The full 228.7B model runs on one die via the streaming MoE path: expert weights stay FP8 on the
host and only the routed experts are dequantized to BF16 per forward, through the in-repo model
(`MiniMaxM2ForCausalLM`, no trust_remote_code) and the PyPTO fused grouped-GEMM kernel.

    MODEL_PATH=/path/to/MiniMax-M2.7 TILE_FWK_DEVICE_ID=0 python ask_minimax_m27.py
"""

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax_m27.ask")

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, "src"))

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu required (Ascend NPU).") from exc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/path/to/MiniMax-M2.7"))
    ap.add_argument("--device", type=int, default=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    ap.add_argument("--prompt", default="Explain mixture-of-experts routing in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=None, help="limit layers for a quick demo")
    return ap.parse_args()


def main():
    from transformers import AutoTokenizer
    import pypto_gym.ops.pypto_tile.minimax_m27 as _pto
    _pto.USE_PTO_GROUPED_GEMM = True        # enable the fused grouped-GEMM kernel for this run
    from pypto_gym.transformers.minimax_m27.modeling_minimax_m27 import (
        build_model, load_streaming_state_dict, attach_expert_fp8, materialize_meta_buffers, patch_moe,
    )

    args = parse_args()
    os.environ.setdefault("PYPTO_VEC_TILE", "128")          # UB-fitting vector tile (910B)
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(args.device)
    dev = f"npu:{args.device}"

    logger.info("Loading %s (streaming FP8 experts) ...", args.model_path)
    model, _ = build_model(args.model_path, max_layers=args.max_layers)
    sd, expert_fp8 = load_streaming_state_dict(args.model_path, max_layers=args.max_layers)
    model.load_state_dict(sd, strict=False, assign=True)
    attach_expert_fp8(model, expert_fp8)
    materialize_meta_buffers(model, "cpu")
    model.to(dev).eval()
    n_patched = patch_moe(model, streaming=True)
    logger.info("  patched %d MoE block(s) with the streaming PyPTO kernel", n_patched)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    logger.info("=" * 60)
    logger.info(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
