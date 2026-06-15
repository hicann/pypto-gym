#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax M2.7 full-model E2E benchmark on Ascend 910B — 8-die pipeline-parallel, PyPTO vs eager MoE.

Multi-process pipeline parallelism (one rank per die): the 62 decoder layers are split into
WORLD_SIZE contiguous stages; rank 0 holds the embedding, the last rank the final norm + lm_head.
A forward streams the hidden state rank->rank over HCCL. Single-process device_map cannot run the
kernel multi-die (torch_npu JIT kernels bind to the compiling die -> rtBinaryGetFunction 107000),
so one process per die is required. Measures prefill forward latency / throughput.

  --moe-impl pypto|eager|vectorized   pypto = fused kernel (default); eager = stock per-expert loop;
                                      vectorized = torch matmul loop on the same flat
  --route natural|uniform|fixed --active N   force a fixed expert load for a fair comparison
                                      (fixed N reproduces a measured real-data average, e.g. ~160/256)
  --graph                             NPU-graph-capture each rank's stage compute and replay it
  --prompt-text "..."                 real text input (sets seq from the tokenized length)

Run (8 dies):
  MODEL_PATH=/path/to/MiniMax-M2.7 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa PYPTO_VEC_TILE=128 \
  TE_PARALLEL_COMPILER=1 torchrun --nproc_per_node=8 bench_minimax_m27.py --moe-impl pypto
  torchrun --nproc_per_node=8 bench_minimax_m27.py --moe-impl eager      # the baseline (15.3x ref)
  # quick mechanism check (2 dies, 4 layers):
  torchrun --nproc_per_node=2 bench_minimax_m27.py --layers 4 --seq 16 --iters 2
"""
import argparse
from dataclasses import dataclass
import gc
import json
import logging
import os
import sys
import time
import typing
import typing_extensions

import torch
import torch.distributed as dist
import torch.nn.functional as functional

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu required (Ascend NPU).") from exc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax_m27.bench")

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8
for _nm in ("Unpack", "Required", "NotRequired", "Self", "TypeVarTuple"):
    if not hasattr(typing, _nm) and hasattr(typing_extensions, _nm):
        setattr(typing, _nm, getattr(typing_extensions, _nm))

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, "src"))

MODEL_PATH = os.environ.get("MODEL_PATH", "/path/to/MiniMax-M2.7")
_BLK = 128


@dataclass
class PPContext:
    """Shared per-rank pipeline-parallel state threaded through the main()-internal helpers.

    A plain mutable bundle (no behavior of its own) so the helpers can read the common pipeline
    state via ctx.<field> instead of long positional arg lists. Fields are populated as main()
    computes them: the topology/config fields up front, then seq_len / pos_emb / mask / input_ids
    once the sequence length and stage compute are resolved.
    """
    rank: int
    world: int
    local: int
    dev: str
    is_first: bool
    is_last: bool
    stage: object
    cfg: object
    my_lo: int
    my_hi: int
    n_layers: int
    hidden_size: int
    moe_impl: str
    args: object
    seq_len: int = 0
    real_ids: object = None
    input_ids: object = None
    position_ids: object = None
    pos_emb: object = None
    mask: object = None


@dataclass
class _FlatBuildCtx:
    """Per-die constants threaded through _build_layer_flat (one layer's flat grouped-GEMM
    build). Bundled so the per-layer body stays a small helper without a long arg list."""
    get: object
    num_experts: int
    hidden_size: int
    dev: str
    patch_minimax_m2_moe: object


def _dequant_fp8_block(w_fp8, scale_inv, blk=_BLK):
    out_dim, in_dim = w_fp8.shape
    w = w_fp8.to(torch.float32)
    s = scale_inv.repeat_interleave(blk, 0)[:out_dim].repeat_interleave(blk, 1)[:, :in_dim]
    return (w * s).to(torch.bfloat16)


def split_layers(n_layers, world):
    """Contiguously split the decoder layers into `world` pipeline stages.

    Extra layers go to the MIDDLE ranks (ranks 0 and world-1 also carry the embedding /
    lm_head, so keep them lighter); each die holds ~7.25 GB of BF16 experts per layer, so we
    avoid handing any single die more than it can fit. Returns a list of (start, end).
    """
    base, extra = divmod(n_layers, world)
    counts = [base] * world
    # distribute `extra` to middle ranks first (avoid loading up the embed/head ranks)
    order = sorted(range(world), key=lambda r: min(r, world - 1 - r), reverse=True)
    for i in range(extra):
        counts[order[i % world]] += 1
    out, s = [], 0
    for c in counts:
        out.append((s, s + c))
        s += c
    return out


def load_stage_state_dict(my_layers, want_embed, want_head, skip_experts=False):
    """Read only this rank's tensors from the FP8 checkpoint, dequantizing to BF16 on CPU.

    If skip_experts, the bulky expert weights are left out (the PyPTO path builds its flat
    buffers directly from the checkpoint instead — see build_pypto_flats_direct — to avoid
    holding both the per-expert Linears and the flat at once).
    """
    from safetensors import safe_open
    with open(f"{MODEL_PATH}/model.safetensors.index.json") as idx_file:
        idx = json.load(idx_file)["weight_map"]
    lo, hi = my_layers

    def want(name):
        if ".layers." in name:
            li = int(name.split(".layers.")[1].split(".")[0])
            if not (lo <= li < hi):
                return False
            if skip_experts and ".block_sparse_moe.experts." in name:
                return False
            return True
        if name.startswith("model.embed_tokens."):
            return want_embed
        if name.startswith("lm_head.") or name == "model.norm.weight":
            return want_head
        return False

    by_shard = {}
    for name, shard in idx.items():
        if name.endswith(".weight_scale_inv"):
            continue
        if want(name):
            by_shard.setdefault(shard, []).append(name)

    sd = {}
    for shard, names in sorted(by_shard.items()):
        f = safe_open(f"{MODEL_PATH}/{shard}", "pt")
        scaled = set(f.keys())
        for name in names:
            t = f.get_tensor(name)
            sname = name + "_scale_inv"
            if t.dtype == torch.float8_e4m3fn and sname in scaled:
                sd[name] = _dequant_fp8_block(t, f.get_tensor(sname))
            else:
                sd[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
        del f
        gc.collect()
    return sd


def _null_expert_weight_slots(experts, num_experts):
    """Null the registered Parameter slot of every expert's w1/w3/w2 Linear (the flat buffers
    replace them). Used both after the meta build and after a layer's flat is built."""
    for e in range(num_experts):
        exp = experts[e]
        for w in ("w1", "w3", "w2"):
            getattr(exp, w).weight = None


def _build_layer_flat(fb, stage, li):
    """Build the flat grouped-GEMM buffers for ONE layer `li` (the body of the per-layer loop
    in build_pypto_flats_direct), attach them to the block's experts and mark it PyPTO-ready.
    `fb.get(name)` reads a checkpoint tensor. Behavior-identical to the original inlined loop."""
    get = fb.get
    num_experts, hidden_size, dev = fb.num_experts, fb.hidden_size, fb.dev
    patch_minimax_m2_moe = fb.patch_minimax_m2_moe
    pfx = f"model.layers.{li}.block_sparse_moe.experts"
    # infer intermediate_size from expert 0 (w1: [I, H])
    intermediate_size = get(f"{pfx}.0.w1.weight").shape[0]
    w13 = torch.empty(num_experts * hidden_size, 2 * intermediate_size, dtype=torch.bfloat16, device=dev)
    w2f = torch.empty(num_experts * intermediate_size, hidden_size, dtype=torch.bfloat16, device=dev)
    for e in range(num_experts):
        p = f"{pfx}.{e}"
        w1 = _dequant_fp8_block(get(f"{p}.w1.weight"), get(f"{p}.w1.weight_scale_inv")).to(dev)
        w3 = _dequant_fp8_block(get(f"{p}.w3.weight"), get(f"{p}.w3.weight_scale_inv")).to(dev)
        w2 = _dequant_fp8_block(get(f"{p}.w2.weight"), get(f"{p}.w2.weight_scale_inv")).to(dev)
        w13[e * hidden_size:(e + 1) * hidden_size] = torch.cat([w1, w3], dim=0).t().contiguous()  # [H,2I]
        w2f[e * intermediate_size:(e + 1) * intermediate_size] = w2.t().contiguous()  # [I, H]
        del w1, w3, w2
    block = stage.layers[str(li)].block_sparse_moe
    patch_minimax_m2_moe(block)                     # binds the PyPTO forward
    ex = block.experts
    ex.pypto_w13_flat = w13
    ex.pypto_w2_flat = w2f
    ex.pypto_num_experts, ex.pypto_intermediate, ex.pypto_hidden = num_experts, intermediate_size, hidden_size
    ex.pypto_streaming = False
    ex.pypto_ready = True                          # skip the lazy build
    # free the meta/empty expert Linears' param slots (the flat replaces them)
    _null_expert_weight_slots(ex, num_experts)
    gc.collect()


def build_pypto_flats_direct(stage, my_layers, cfg, dev):
    """Build one preallocated flat grouped-GEMM buffer per owned layer, directly from FP8.

    Reads the checkpoint expert-by-expert into a single [E*H,2I]/[E*I,H] BF16 buffer per
    layer on the die — never materializing the per-expert Linears, so the die holds ~one
    layer's worth of transient (a flat + one expert) instead of the experts AND the flat at
    once (which would 2x the peak and OOM at 8 layers/die). Attaches the flats onto each
    block's experts module and marks it PyPTO-ready. Returns the number of layers built.
    """
    from safetensors import safe_open
    from pypto_gym.transformers.minimax_m27.modeling_minimax_m27 import patch_minimax_m2_moe
    with open(f"{MODEL_PATH}/model.safetensors.index.json") as idx_file:
        idx = json.load(idx_file)["weight_map"]
    lo, hi = my_layers
    num_experts = cfg.num_local_experts
    hidden_size = cfg.hidden_size
    handles = {}

    def get(name):
        sh = idx[name]
        if sh not in handles:
            handles[sh] = safe_open(f"{MODEL_PATH}/{sh}", "pt")
        return handles[sh].get_tensor(name)

    fb = _FlatBuildCtx(get, num_experts, hidden_size, dev, patch_minimax_m2_moe)
    n = 0
    for li in range(lo, hi):
        _build_layer_flat(fb, stage, li)
        n += 1
    return n


def grouped_ffn_vectorized(self, sorted_x, counts):
    """Vectorized eager expert FFN over the SAME PyPTO flat weights.

    A python loop of contiguous-slice torch matmuls (one host sync for counts, no per-expert
    gather). Isolates the fused PyPTO kernel vs plain torch matmul on an identical layout.
    """
    num_tokens, hidden_size = sorted_x.shape
    num_experts, intermediate_size = self.pypto_num_experts, self.pypto_intermediate
    w13, w2 = self.pypto_w13_flat, self.pypto_w2_flat
    out = torch.empty_like(sorted_x)
    cnt = counts.to("cpu")
    off = 0
    for e in range(num_experts):
        c = int(cnt[e])
        if c == 0:
            continue
        xe = sorted_x[off:off + c]                     # [c, H]
        gu = xe @ w13[e * hidden_size:(e + 1) * hidden_size]
        g, u = gu[:, :intermediate_size], gu[:, intermediate_size:]
        out[off:off + c] = (functional.silu(g) * u) @ w2[e * intermediate_size:(e + 1) * intermediate_size]
        off += c
    return out


def make_static_moe_forward(tok_src, inv, offsets, top_k):
    """Build a fully graph-capturable MoE forward for a fixed (controlled) route.

    All routing — the argsort (which runs on AICPU and synchronizes, breaking NPU-graph
    capture), bincount and cumsum — is precomputed on the host OUTSIDE the captured region.
    Inside, only device ops remain: gather (index_select) -> per-expert matmul/SwiGLU over
    static slices -> scatter -> weighted sum. No host sync, no AICPU op -> capturable.
    """
    def forward(self, hidden, top_k_index, top_k_weights):
        hidden_size = hidden.shape[1]
        intermediate_size = self.pypto_intermediate
        w13, w2 = self.pypto_w13_flat, self.pypto_w2_flat
        sorted_x = hidden.index_select(0, tok_src)          # [N*K, H]
        out = torch.empty_like(sorted_x)
        for e, off, c in offsets:
            xe = sorted_x[off:off + c]
            gu = xe @ w13[e * hidden_size:(e + 1) * hidden_size]
            g, u = gu[:, :intermediate_size], gu[:, intermediate_size:]
            out[off:off + c] = (functional.silu(g) * u) @ w2[e * intermediate_size:(e + 1) * intermediate_size]
        combined = out.index_select(0, inv).view(-1, top_k, hidden_size)
        return (combined * top_k_weights.reshape(-1, top_k).unsqueeze(-1)).sum(1).to(hidden.dtype)
    return forward


def _route_width(mode, num_experts, top_k, active):
    """Number of distinct experts the controlled route spreads tokens over.
      uniform : all E   |   fixed : exactly `active` experts
    `fixed` lets us reproduce a measured real-data load (e.g. ~160/256) as a STATIC route so it
    is NPU-graph-capturable — real dynamic routing varies per forward and can't be captured."""
    if mode == "uniform":
        return num_experts
    if mode == "fixed":
        return max(top_k, min(int(active), num_experts))
    return num_experts


def precompute_static_routing(width, top_k, num_experts, num_tokens, dev):
    """Build host-side perm/inv/offsets for the deterministic controlled route.

    `width` is the number of distinct experts to spread the tokens over (see _route_width).
    """
    flat = (torch.arange(num_tokens * top_k) % width)
    perm = flat.argsort()
    tok_src = (perm // top_k).to(dev)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(num_tokens * top_k)
    counts = torch.bincount(flat[perm], minlength=num_experts)
    offs, off = [], 0
    for e in range(num_experts):
        c = int(counts[e])
        if c:
            offs.append((e, off, c))
            off += c
    return tok_src, inv.to(dev), offs


def make_controlled_route(mode, top_k, num_experts, active=None):
    """Return a drop-in route_tokens_to_experts that forces a fixed expert-load distribution.

    Instead of the gate's natural routing, the same load hits BOTH the PyPTO and eager expert
    FFNs for a fair E2E comparison — uniform spreads over all experts, fixed uses exactly
    `active` experts (reproducing a measured real-data load, e.g. ~160/256, as a static
    route). top_k_weights are set uniform (1/top_k) — magnitudes don't affect timing.
    """
    width = _route_width(mode, num_experts, top_k, active)

    def route(self, router_logits):
        num_tokens = router_logits.shape[0]
        dev = router_logits.device
        ar = torch.arange(num_tokens * top_k, device=dev).view(num_tokens, top_k)
        idx = (ar % width).to(torch.int64)
        w = torch.full((num_tokens, top_k), 1.0 / top_k, device=dev, dtype=router_logits.dtype)
        return idx, w

    return route


def build_rotary(cfg, device):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    rope_type = "default"
    if getattr(cfg, "rope_scaling", None):
        rope_type = cfg.rope_scaling.get("rope_type", cfg.rope_scaling.get("type", "default"))
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[rope_type](cfg, device)
    return inv_freq.to(device), float(scaling)


def _parse_args():
    """Parse the benchmark CLI flags (behavior-identical to the original inline parser)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="limit total layers (testing)")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-pypto", action="store_true", help="alias for --moe-impl eager")
    ap.add_argument("--moe-impl", choices=["pypto", "eager", "vectorized"], default=None,
                    help="pypto=fused kernel; eager=stock per-expert loop; "
                         "vectorized=python loop of contiguous matmuls on the same flat")
    ap.add_argument("--graph", action="store_true",
                    help="NPU-graph-capture each rank's stage compute and replay it")
    ap.add_argument("--route", choices=["natural", "uniform", "fixed"], default="natural",
                    help="force a fixed expert-load distribution for the comparison "
                         "(fixed = exactly --active experts, e.g. the measured real-data average)")
    ap.add_argument("--active", type=int, default=160,
                    help="for --route fixed: number of experts to activate (default 160)")
    ap.add_argument("--prompt-text", default=None,
                    help="real text to tokenize as input (instead of random ids) — for measuring "
                         "the real-data expert load; sets seq from the tokenized length")
    ap.add_argument("--count-experts", action="store_true",
                    help="report active experts per layer (one forward); use with natural route")
    ap.add_argument("--report-file", default=None)
    return ap.parse_args()


def _init_distributed():
    """Bind this process to its die, init the HCCL group, and pre-create the comms.

    Pre-creating the HCCL comms (collective + pipeline P2P) NOW, while device memory is
    free, is required: HCCL allocates buffer space at comm-init; if we let it init lazily
    AFTER the ~58 GB weight load it fails with hcclCommInitRootInfoConfig error code 1 (no
    room). Returns (rank, world, local, dev)."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    dev = f"npu:{local}"
    torch_npu.npu.config.allow_internal_format = True
    os.environ.setdefault("PYPTO_VEC_TILE", "128")
    dist.init_process_group("hccl", rank=rank, world_size=world)

    _w = torch.zeros(1, device=dev)
    dist.all_reduce(_w)
    _reqs = []
    if rank + 1 < world:
        _reqs.append(dist.isend(_w, dst=rank + 1))
    if rank - 1 >= 0:
        _reqs.append(dist.irecv(torch.zeros(1, device=dev), src=rank - 1))
    for _q in _reqs:
        _q.wait()
    torch.npu.synchronize()
    del _w
    return rank, world, local, dev


def _load_config(args):
    """Load the in-repo MiniMaxM27Config from the checkpoint (no trust_remote_code / no HF
    download), applying the layer-count override. Returns (cfg, n_layers)."""
    from pypto_gym.transformers.minimax_m27.configuration_minimax_m27 import MiniMaxM27Config
    with open(f"{MODEL_PATH}/config.json") as cfg_file:
        _cj = json.load(cfg_file)
    _drop = ("architectures", "auto_map", "quantization_config", "_pre_quantization_dtype",
             "dtype", "torch_dtype", "transformers_version", "model_type")
    cfg = MiniMaxM27Config(**{k: v for k, v in _cj.items() if k not in _drop})
    setattr(cfg, "_attn_implementation", "eager")  # eager attention (no flash_attn on NPU)
    n_layers = args.layers or cfg.num_hidden_layers
    cfg.num_hidden_layers = n_layers
    return cfg, n_layers


def _build_stage(cfg, my_layers, is_first, is_last, moe_impl):
    """Build only this rank's stage module on the die: keep just the owned decoder layers
    (+ embedding on rank 0, norm/lm_head on the last rank), load their checkpoint tensors,
    and null the skipped expert param slots for the flat path. Returns the stage module."""
    my_lo, my_hi = my_layers
    from pypto_gym.transformers.minimax_m27.modeling_minimax_m27 import MiniMaxM2ForCausalLM
    with torch.device("meta"):
        full = MiniMaxM2ForCausalLM(cfg)
    model = full.model

    # keep only my layers (drop the rest to avoid materializing them)
    my_layer_mods = {li: model.layers[li] for li in range(my_lo, my_hi)}

    # pypto and vectorized both use the flat layout (build it directly); eager keeps Linears
    use_flat = moe_impl in ("pypto", "vectorized")
    sd = load_stage_state_dict((my_lo, my_hi), is_first, is_last, skip_experts=use_flat)
    # assemble a stage module to load_state_dict into (subset)
    import torch.nn as nn
    stage = nn.Module()
    stage.layers = nn.ModuleDict({str(li): my_layer_mods[li] for li in range(my_lo, my_hi)})
    if is_first:
        stage.embed_tokens = model.embed_tokens
    if is_last:
        stage.norm = model.norm
        stage.lm_head = full.lm_head

    # remap checkpoint names -> stage names
    remap = {}
    for k, v in sd.items():
        if k.startswith("model.layers."):
            li = int(k.split(".layers.")[1].split(".")[0])
            rest = k.split(f".layers.{li}.", 1)[1]
            remap[f"layers.{li}.{rest}"] = v
        elif k.startswith("model.embed_tokens."):
            remap[k[len("model."):]] = v
        elif k == "model.norm.weight":
            remap["norm.weight"] = v
        elif k.startswith("lm_head."):
            remap[k] = v
    stage.load_state_dict(remap, strict=False, assign=True)
    del sd, remap
    gc.collect()

    if use_flat:
        # expert Linears were skipped above and are still on meta; null their param slots so
        # stage.to(dev) doesn't try to move meta tensors. The flats replace them.
        num_experts = cfg.num_local_experts
        for li in range(my_lo, my_hi):
            ex = stage.layers[str(li)].block_sparse_moe.experts
            _null_expert_weight_slots(ex, num_experts)
    return stage


def _configure_moe(ctx, log):
    """Build the PyPTO flats (pypto/vectorized), select the vectorized FFN, and install the
    controlled / counting route wrappers. Returns the per-layer active-expert log list."""
    stage, cfg, dev, moe_impl, args = ctx.stage, ctx.cfg, ctx.dev, ctx.moe_impl, ctx.args
    my_lo, my_hi = ctx.my_lo, ctx.my_hi
    use_flat = moe_impl in ("pypto", "vectorized")
    if use_flat:
        nb = build_pypto_flats_direct(stage, (my_lo, my_hi), cfg, dev)
        if moe_impl == "vectorized":
            # same flat weights, but run a plain torch-matmul loop instead of the fused kernel
            for li in range(my_lo, my_hi):
                ex = stage.layers[str(li)].block_sparse_moe.experts
                ex.grouped_ffn = grouped_ffn_vectorized.__get__(ex)
        log(f"[pp] rank0 built {nb} MoE flats; impl={moe_impl}")

    # force a fixed expert-load distribution (same for pypto and eager) for the comparison
    if args.route != "natural":
        top_k = cfg.num_experts_per_tok
        num_experts = cfg.num_local_experts
        route_fn = make_controlled_route(args.route, top_k, num_experts, args.active)
        for li in range(my_lo, my_hi):
            blk = stage.layers[str(li)].block_sparse_moe
            blk.route_tokens_to_experts = route_fn.__get__(blk)
        log(f"[pp] forced route={args.route}"
            f"{('(active='+str(args.active)+')') if args.route=='fixed' else ''} on all MoE blocks")

    # active-expert counter (natural routing): wrap each block's route to record how many
    # distinct experts a forward lights up — measures the real-data load on the 256 experts.
    active_log = []
    if args.count_experts:
        for li in range(my_lo, my_hi):
            blk = stage.layers[str(li)].block_sparse_moe
            orig = blk.route_tokens_to_experts

            def counting(self, router_logits, _orig=orig, _li=li):
                idx, w = _orig(router_logits)
                active_log.append((_li, int(idx.reshape(-1).unique().numel())))
                return idx, w
            blk.route_tokens_to_experts = counting.__get__(blk)
    return active_log


def _resolve_seq(args, is_first, dev):
    """Determine the sequence length (and rank-0 real token ids). Real text input (rank 0
    tokenizes) sets the length and is broadcast to all ranks. Returns (real_ids, seq_len)."""
    if args.prompt_text:
        if is_first:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            real_ids = _tok(args.prompt_text, return_tensors="pt").input_ids.to(dev)
            s_t = torch.tensor([real_ids.shape[1]], device=dev)
        else:
            real_ids = None
            s_t = torch.zeros(1, dtype=torch.int64, device=dev)
        dist.broadcast(s_t, src=0)
        seq_len = int(s_t.item())
    else:
        real_ids = None
        seq_len = args.seq
    return real_ids, seq_len


def _make_stage_run(ctx, log):
    """Assemble the captured/uncaptured local stage compute. Installs the graph-friendly
    static MoE forward when needed, builds rotary cos/sin and the causal mask, then either
    NPU-graph-captures the stage or returns the plain forward. Returns stage_run(hidden)."""
    stage, cfg, dev, moe_impl, args = ctx.stage, ctx.cfg, ctx.dev, ctx.moe_impl, ctx.args
    my_lo, my_hi = ctx.my_lo, ctx.my_hi
    hidden_size, seq_len = ctx.hidden_size, ctx.seq_len
    inv_freq, attn_scaling = build_rotary(cfg, dev)
    position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)

    # graph-friendly vectorized: replace the experts.forward with a fully-static one that
    # precomputes ALL routing (argsort runs on AICPU and synchronizes -> would break capture)
    # on the host, leaving only device ops (gather/matmul/scatter) inside the captured region.
    if moe_impl == "vectorized" and args.graph:
        if args.route == "natural":
            raise ValueError("--graph with vectorized needs --route uniform|fixed")
        top_k = cfg.num_experts_per_tok
        route_width = _route_width(args.route, cfg.num_local_experts, top_k, args.active)
        tok_src, inv, offs = precompute_static_routing(
            route_width, top_k, cfg.num_local_experts, seq_len, dev)
        for li in range(my_lo, my_hi):
            ex = stage.layers[str(li)].block_sparse_moe.experts
            ex.forward = make_static_moe_forward(tok_src, inv, offs, top_k).__get__(ex)
        log(f"[pp] static MoE forward for graph ({len(offs)} active experts/layer)")
    # rotary cos/sin
    with torch.no_grad():
        freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = (emb.cos() * attn_scaling).to(torch.bfloat16)
        sin = (emb.sin() * attn_scaling).to(torch.bfloat16)
    pos_emb = (cos, sin)
    # additive causal mask [1,1,S,S]
    mask = torch.full((seq_len, seq_len), float("-inf"), device=dev, dtype=torch.bfloat16).triu(1)
    mask = mask.view(1, 1, seq_len, seq_len)

    def stage_forward(hidden):
        for li in range(my_lo, my_hi):
            hidden = stage.layers[str(li)](
                hidden, position_embeddings=pos_emb, attention_mask=mask,
                position_ids=position_ids, use_cache=False)
        return hidden

    # Optional NPU-graph capture of the (comm-free) local stage compute. The cross-rank
    # send/recv stays OUTSIDE the graph; we copy the incoming hidden into a static buffer,
    # replay, and read a static output. Requires a static op sequence — host-sync-free MoE
    # (pypto/eager) and a fixed routing (use --route uniform|fixed).
    stage_run = stage_forward
    if args.graph:
        hin = torch.zeros(1, seq_len, hidden_size, dtype=torch.bfloat16, device=dev)
        with torch.no_grad():
            for _ in range(3):
                stage_forward(hin)
        torch.npu.synchronize()
        gobj = torch.npu.NPUGraph()
        with torch.no_grad(), torch.npu.graph(gobj):
            hout = stage_forward(hin)

        def stage_run(hidden):
            hin.copy_(hidden)
            gobj.replay()
            return hout
        log(f"[pp] NPU-graph captured stage compute (impl={moe_impl})")
    return stage_run


def _make_pipeline_once(ctx, stage_run):
    """Build the single pipeline forward closure: rank 0 embeds, others recv the hidden over
    HCCL; run the local stage; forward to the next rank or finish with norm + lm_head."""
    stage, dev = ctx.stage, ctx.dev
    hidden_size, seq_len = ctx.hidden_size, ctx.seq_len
    rank, is_first, is_last = ctx.rank, ctx.is_first, ctx.is_last

    def pipeline_once(input_ids):
        if is_first:
            hidden = stage.embed_tokens(input_ids)
        else:
            hidden = torch.empty(1, seq_len, hidden_size, dtype=torch.bfloat16, device=dev)
            dist.recv(hidden, src=rank - 1)
        hidden = stage_run(hidden)
        if not is_last:
            dist.send(hidden.contiguous(), dst=rank + 1)
            return None
        hidden = stage.norm(hidden)
        logits = stage.lm_head(hidden[:, -1:, :])
        return logits
    return pipeline_once


def _run_count_experts(ctx, pipeline_once, input_ids, active_log):
    """One forward to record active-expert counts, then report (skips the timing loop)."""
    cfg, rank, seq_len = ctx.cfg, ctx.rank, ctx.seq_len
    my_lo, my_hi = ctx.my_lo, ctx.my_hi
    with torch.no_grad():
        pipeline_once(input_ids)
    torch.npu.synchronize()
    if active_log:
        cnts = [c for _, c in active_log]
        import statistics
        logger.info("[pp ACTIVE rank%d layers%d-%d] per-layer active experts (of %d): "
                    "min=%d max=%d mean=%.1f seq=%d tokens*topk=%d",
                    rank, my_lo, my_hi - 1, cfg.num_local_experts,
                    min(cnts), max(cnts), statistics.mean(cnts), seq_len,
                    seq_len * cfg.num_experts_per_tok)


def _run_timing(ctx, pipeline_once, input_ids):
    """Warmup then time the prefill forward over args.iters; the last rank logs the result
    and optionally writes the JSON report. Identical loop/sync/timing to the original."""
    args, cfg, moe_impl = ctx.args, ctx.cfg, ctx.moe_impl
    world, n_layers, seq_len, is_last = ctx.world, ctx.n_layers, ctx.seq_len, ctx.is_last
    # warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            pipeline_once(input_ids)
        torch.npu.synchronize()
        dist.barrier()

    # timed
    times = []
    for _ in range(args.iters):
        dist.barrier()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipeline_once(input_ids)
        torch.npu.synchronize()
        dist.barrier()
        times.append(time.perf_counter() - t0)

    if is_last:
        best = min(times)
        avg = sum(times) / len(times)
        tps = seq_len / best
        tag = moe_impl + ("+graph" if args.graph else "")
        rlabel = args.route + (f"({args.active})" if args.route == "fixed" else "")
        logger.info("\n[pp RESULT] world=%d layers=%d seq=%d route=%s impl=%s",
                    world, n_layers, seq_len, rlabel, tag)
        logger.info("  forward: best=%.1f ms  avg=%.1f ms  prefill_throughput=%.1f tok/s",
                    best * 1e3, avg * 1e3, tps)
        logger.info("  logits shape=%s argmax=%d", tuple(out.shape), int(out[0, -1].argmax()))
        if args.report_file:
            with open(args.report_file, "w") as report:
                json.dump({"world": world, "layers": n_layers, "seq": seq_len, "route": args.route,
                           "active": (args.active if args.route == "fixed" else None),
                           "impl": moe_impl, "graph": bool(args.graph),
                           "best_ms": best * 1e3, "avg_ms": avg * 1e3, "prefill_tok_s": tps},
                          report, indent=2)
            logger.info("  report: %s", args.report_file)


def main():
    # in-repo import (deferred until here so it follows the sys.path bootstrap above);
    # the flag must be set before any model is built, which main() guarantees as the entry.
    import pypto_gym.ops.pypto_tile.minimax_m27 as _pto
    _pto.USE_PTO_GROUPED_GEMM = True        # enable the fused grouped-GEMM kernel for the pypto path

    args = _parse_args()
    rank, world, local, dev = _init_distributed()

    def log(*a):
        if rank == 0:
            logger.info(" ".join(str(x) for x in a))

    cfg, n_layers = _load_config(args)
    hidden_size = cfg.hidden_size
    stages = split_layers(n_layers, world)
    my_lo, my_hi = stages[rank]
    my_layers = (my_lo, my_hi)
    is_first, is_last = rank == 0, rank == world - 1
    log(f"[pp] world={world} layers={n_layers} stages={stages} H={hidden_size} pypto={not args.no_pypto}")

    moe_impl = args.moe_impl or ("eager" if args.no_pypto else "pypto")
    stage = _build_stage(cfg, my_layers, is_first, is_last, moe_impl)
    stage = stage.to(dev).eval()

    # bundle the shared per-rank pipeline state once; later-computed fields (seq_len, input_ids)
    # are assigned on the ctx as main() resolves them.
    ctx = PPContext(rank=rank, world=world, local=local, dev=dev, is_first=is_first, is_last=is_last,
                    stage=stage, cfg=cfg, my_lo=my_lo, my_hi=my_hi, n_layers=n_layers,
                    hidden_size=hidden_size, moe_impl=moe_impl, args=args)

    active_log = _configure_moe(ctx, log)

    dist.barrier()

    # ---- pipeline forward (prefill, no cache) ----
    real_ids, seq_len = _resolve_seq(args, is_first, dev)
    ctx.real_ids, ctx.seq_len = real_ids, seq_len
    stage_run = _make_stage_run(ctx, log)
    pipeline_once = _make_pipeline_once(ctx, stage_run)

    torch.manual_seed(0)
    if is_first:
        input_ids = real_ids if real_ids is not None else torch.randint(0, cfg.vocab_size, (1, seq_len), device=dev)
    else:
        input_ids = None
    ctx.input_ids = input_ids

    if args.count_experts:
        _run_count_experts(ctx, pipeline_once, input_ids, active_log)
    else:
        _run_timing(ctx, pipeline_once, input_ids)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
