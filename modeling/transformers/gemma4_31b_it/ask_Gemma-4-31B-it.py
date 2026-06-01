#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""Gemma-4-31B-it inference script with optional PyPTO kernel acceleration."""

import argparse
import json
import os
import shutil
import sys
import time

import torch


def setup_pypto(repo_root, model_path):
    """Install patched modeling file and inject PyPTO kernel switch."""
    sys.path.insert(0, os.path.join(repo_root, "src"))
    # Install the modified modeling file into model directory
    patched_src = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                               "gemma4_31b_it", "modeling_gemma4.py")
    patched_dst = os.path.join(model_path, "modeling_gemma4.py")
    if os.path.exists(patched_src):
        backup = patched_dst + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(patched_dst, backup)
        shutil.copy2(patched_src, patched_dst)
        print(f"[PyPTO] Installed patched modeling to {patched_dst}")
    # Import real kernel adapter module and register under expected name
    from pypto_gym.ops.pypto_tile import gemma4_31b_it as gemma4_kernels
    gemma4_kernels.USE_PTO_SOFTMAX = True
    gemma4_kernels.USE_PTO_GQA = True
    sys.modules["gemma4_pto_kernels"] = gemma4_kernels
    print("[PyPTO] Attention SoftMax and GQA Decode kernels enabled")


def main():
    parser = argparse.ArgumentParser(description="Gemma-4-31B-it inference")
    parser.add_argument("--prompt", type=str, default="Explain machine learning in simple terms.")
    parser.add_argument("--device", type=str, default="0", help="NPU device ID")
    parser.add_argument("--model-path", type=str, required=True, help="Model weight path")
    parser.add_argument("--sentence_file", type=str, default=None, help="Input prompts file")
    parser.add_argument("--output_length", type=int, default=100, help="Max new tokens")
    parser.add_argument("--use_pypto", action="store_true", help="Enable PyPTO fused kernels")
    parser.add_argument("--report_file", type=str, default=None, help="JSON report output path")
    parser.add_argument("--show_outputs", action="store_true", help="Print generated text")
    args = parser.parse_args()

    # Setup device
    device_str = f"npu:{args.device}"
    os.environ.setdefault("TILE_FWK_DEVICE_ID", args.device)

    # Optional: enable PyPTO kernels (must happen BEFORE transformers import)
    if args.use_pypto:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        setup_pypto(repo_root, args.model_path)

    # Now import transformers
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    # Load tokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok_time = time.time() - t0

    # Load model
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device_str,
        trust_remote_code=True,
    )
    model_time = time.time() - t0

    # Read input
    if args.sentence_file and os.path.exists(args.sentence_file):
        with open(args.sentence_file, "r") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)

    # Timed generation
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.output_length)
    gen_time = time.time() - t0

    output_tokens = outputs.shape[1] - input_len
    tps = output_tokens / gen_time if gen_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Model       : {args.model_path}")
    print(f"PyPTO       : {'ON' if args.use_pypto else 'OFF'}")
    print(f"Input tokens: {input_len}")
    print(f"Output tokens: {output_tokens}")
    print(f"Generate time: {gen_time:.2f}s")
    print(f"Tokens/sec   : {tps:.1f}")
    print(f"Tokenizer load: {tok_time:.2f}s")
    print(f"Model load    : {model_time:.2f}s")
    print(f"{'='*60}")

    if args.show_outputs:
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated:\n{text}")

    if args.report_file:
        report = {
            "model": args.model_path,
            "pypto": args.use_pypto,
            "input_tokens": input_len,
            "output_tokens": output_tokens,
            "generate_time_s": gen_time,
            "tokens_per_sec": tps,
            "tokenizer_load_s": tok_time,
            "model_load_s": model_time,
        }
        with open(args.report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report_file}")


if __name__ == "__main__":
    main()
