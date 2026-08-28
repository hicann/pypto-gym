# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Qwen3.5-9B 推理脚本
用法: python3 ask_Qwen3.5-9B.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
       [--report-file out.json] [--per-step-timing] [--profile-dir dir] [--use-acl-graph]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch_npu

parser = argparse.ArgumentParser(description="Qwen3.5-9B 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""),
                    help="模型权重路径（默认取环境变量 MODEL_PATH）")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pypto", action="store_true", help="PyPTO融合算子模式")
parser.add_argument("--report-file", type=str, default=None,
                    help="将性能指标(JSON: 加载/推理/吞吐/峰值显存)写入此文件")
parser.add_argument("--per-step-timing", action="store_true",
                    help="记录逐token耗时(prefill->首token / decode均值)")
parser.add_argument("--profile-dir", type=str, default=None,
                    help="torch_npu profiler trace 输出目录")
parser.add_argument("--use-acl-graph", action="store_true",
                    help="尝试 torchair aclgraph 整网捕获；本模型 vendored modeling 含原地算子，"
                         "通常回退 eager（算子级 aclgraph 见 tests/ops/qwen3_5_9b 的 aclgraph 测试）")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\n".join(lines)
    logging.info(f"Prompt read from file: {args.sentence_file} ({len(prompt)} chars)")
else:
    prompt = "你好，请介绍一下自己。"

if not args.model_path:
    parser.error("model path required: pass --model-path or set MODEL_PATH")

# The sys.modules injection must happen BEFORE transformers is imported. The
# USE_PTO flag, however, is enabled only AFTER the model is on the NPU (skill
# step-22 timing rule), just before the forward hook is installed below.
if args.use_pypto:
    sys.path.insert(0, args.model_path)
    import qwen3_5_9b_pto_kernels as pto_kernels
    sys.modules["qwen3_5_9b_pto_kernels"] = pto_kernels
    logging.info("PyPTO module injected (fused GatedDeltaRule enabled after model load)")

from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

logging.info(f"Using device: npu:{args.device}")
logging.info(f"Model path: {args.model_path}")

torch.npu.set_device(args.device)
torch.npu.reset_peak_memory_stats(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

_t_load = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.model_path, local_files_only=True, trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(f"npu:{args.device}").eval()
torch.npu.synchronize(args.device)
load_time = time.time() - _t_load
logging.info(f"Model load time: {load_time:.3f}s")

# Monkey-patch Qwen3_5GatedDeltaNet to route the chunk-prefill path through
# the PyPTO wrapper. The hook is also embedded in the bundled modeling file
# (src/pypto_gym/transformers/qwen3_5_9b/modeling_qwen3_5.py); this runtime
# patch covers users who load the model with the upstream transformers code.
if args.use_pypto:
    import types
    # Enable the fused path now that the model is on the NPU (step-22 timing).
    pto_kernels.USE_PTO_GATED_DELTA_RULE = True
    patched_count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5GatedDeltaNet":

            orig_forward = module.forward

            def new_forward(self, *fwd_args, _orig=orig_forward, _pk=pto_kernels, **fwd_kwargs):
                _orig_chunk = self.chunk_gated_delta_rule
                if getattr(_pk, "USE_PTO_GATED_DELTA_RULE", False):
                    def _patched(q, k, v, **kw):
                        try:
                            return _pk.gated_delta_rule_wrapper(q, k, v, **kw)
                        except NotImplementedError:
                            return _orig_chunk(q, k, v, **kw)
                    self.chunk_gated_delta_rule = _patched
                try:
                    return _orig(*fwd_args, **fwd_kwargs)
                finally:
                    self.chunk_gated_delta_rule = _orig_chunk
            module.forward = types.MethodType(new_forward, module)
            patched_count += 1
    logging.info(f"PyPTO GatedDeltaRule monkey-patched to {patched_count} layers")

# Optional torchair aclgraph capture. Whole-model capture is currently blocked by
# in-place ops in the vendored modeling; we attempt it and fall back to eager so
# the flag is honest rather than fatal.
eager_model = model
gen_model = model
if args.use_acl_graph:
    try:
        import torchair  # noqa: F401  (also importable as torch_npu.dynamo.torchair)
        gen_model = torch.compile(model, backend=torchair.get_npu_backend(),
                                  mode="reduce-overhead", dynamic=False)
        logging.info("torchair aclgraph enabled (reduce-overhead)")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"torchair aclgraph init failed, falling back to eager: {e}")
        gen_model = eager_model

inputs = tokenizer(text, return_tensors="pt").to(f"npu:{args.device}")
logging.info(f"Input token count: {inputs.input_ids.shape[1]}")

# Optional per-token timing via a lightweight streamer. transformers echoes the prompt
# via put(input_ids) BEFORE prefill, then one put per generated token; record numel to
# tell the echo from real tokens.
streamer = None
step_times = []
step_nums = []
if args.per_step_timing:
    from transformers.generation.streamers import BaseStreamer

    class _TimingStreamer(BaseStreamer):
        def put(self, value):
            step_times.append(time.time())
            try:
                step_nums.append(int(value.numel()))
            except Exception:
                step_nums.append(1)

        def end(self):
            pass

    streamer = _TimingStreamer()


def _generate(m):
    with torch.no_grad():
        return m.generate(**inputs, max_new_tokens=args.output_length,
                          temperature=0.7, do_sample=True, top_p=0.8, streamer=streamer)


torch.npu.synchronize(args.device)
_t_inf = time.time()
if args.profile_dir:
    os.makedirs(args.profile_dir, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
    ):
        outputs = _generate(gen_model)
else:
    try:
        outputs = _generate(gen_model)
    except Exception as e:  # noqa: BLE001
        if args.use_acl_graph and gen_model is not eager_model:
            logging.warning(
                "aclgraph full-graph capture failed (vendored modeling "
                f"in-place ops), falling back to eager: {e}"
            )
            outputs = _generate(eager_model)
        else:
            raise
torch.npu.synchronize(args.device)
infer_time = time.time() - _t_inf

num_new = int(outputs.shape[1] - inputs.input_ids.shape[1])
throughput = num_new / infer_time if infer_time > 0 else 0.0
peak_mem_mb = torch.npu.max_memory_allocated(args.device) / 1024 / 1024

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
logging.info(f"[PERF] mode={'pto' if args.use_pypto else 'baseline'} load={load_time:.3f}s "
             f"infer={infer_time:.3f}s new_tokens={num_new} "
             f"throughput={throughput:.2f}tok/s peak_mem={peak_mem_mb:.1f}MB")

if args.per_step_timing and len(step_times) >= 2:
    # If the prompt was echoed first (numel>1), step_times[0] is the echo and step_times[1]
    # is the first real token -> prefill = step_times[1]-step_times[0]; otherwise step_times[0]
    # is already the first token (prefill = step_times[0]-_t_inf).
    if step_nums and step_nums[0] > 1:
        first = step_times[1] - step_times[0]
        deltas = [step_times[i] - step_times[i - 1] for i in range(2, len(step_times))]
    else:
        first = step_times[0] - _t_inf
        deltas = [step_times[i] - step_times[i - 1] for i in range(1, len(step_times))]
    decode_mean = sum(deltas) / len(deltas) if deltas else float("nan")
    logging.info(f"[PER-STEP] prefill->1st_tok={first:.3f}s decode_mean={decode_mean:.4f}s "
                 f"over {len(deltas)} steps")

if args.report_file:
    report = {
        "model": "Qwen3.5-9B",
        "mode": "pto" if args.use_pypto else "baseline",
        "device": args.device,
        "output_length": args.output_length,
        "new_tokens": num_new,
        "load_time_s": round(load_time, 3),
        "infer_time_s": round(infer_time, 3),
        "throughput_tok_s": round(throughput, 2),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "use_pypto": bool(args.use_pypto),
        "use_acl_graph": bool(args.use_acl_graph),
        "command": "python3 " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    }
    with open(args.report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"[REPORT] written to {args.report_file}")
