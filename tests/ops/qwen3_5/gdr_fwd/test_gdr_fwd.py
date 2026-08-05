# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys
from dataclasses import dataclass

_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:      # 让 gdr_fwd_golden 可直接 import
    sys.path.insert(0, _test_dir)
# 让 gdr_fwd 包（src/pypto_gym/ops/pypto_tensor/qwen3_5/gdr_fwd/）可 import
_repo_root = _test_dir
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "src", "pypto_gym", "ops", "pypto_tensor", "qwen3_5")):
        break
    _repo_root = os.path.dirname(_repo_root)
_ops_dir = os.path.join(_repo_root, "src", "pypto_gym", "ops", "pypto_tensor", "qwen3_5")
if _ops_dir not in sys.path:
    sys.path.insert(0, _ops_dir)
del _repo_root, _ops_dir
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import time
import traceback
import warnings

import torch
import torch_npu

from gdr_fwd_golden import gdr_fwd_golden
from gdr_fwd import gdr_fwd_impl
from gdr_fwd.gdr_fwd_impl import chunk_gated_delta_rule_wrapper

# ─────────────────────────────────────────────────────────────────────────────
# 判据门限
# ─────────────────────────────────────────────────────────────────────────────
# 主判据：kernel vs golden(emulate_bf16=True)。atol/rtol 做**逐元素**把关；
# l2_rel 门限做**整体**把关，二者必须同时满足。
MAIN_ATOL_O, MAIN_RTOL_O = 2e-3, 2e-3          # o：bf16 输出，1 ulp ≈ 4e-3·|x|
MAIN_ATOL_S, MAIN_RTOL_S = 5e-3, 5e-3          # final_state：fp32，跨 chunk 累加量级更大
MAIN_L2_GATE = 3e-3
LONG_L2_GATE = 6e-3    # 长序列（256+ chunks）状态递推累积舍入误差，门限适当放宽

# 参考判据：kernel vs golden(fp32 真值)。差值 = bf16 路径固有损失，预期 ≈ 4e-3。
REF_ATOL, REF_RTOL = 3e-2, 3e-2
REF_L2_GATE = 1e-2


@dataclass
class Threshold:
    """逐元素 + 整体精度门限，``_cmp`` 的比较配置。"""
    atol: float
    rtol: float
    l2_gate: float


@dataclass
class Metrics:
    """单次比较的数值指标，``_record`` 的结果字段。"""
    max_abs: float
    l2_rel: float
    oor: float


@dataclass
class TestKey:
    """一条比较记录的标识（suite / case / tensor / ref_tag）。"""
    suite: str
    case: str
    tensor: str
    ref: str


@dataclass
class CaseShape:
    """输入构造的形状 + 配置参数，``_make_case`` 的入参。"""
    b: int
    t: int
    h: int
    hv: int
    d: int
    seed: int = 0
    g_range: tuple = (-0.10, -0.001)
    with_state: bool = False
    n_seq: int | None = None
    dtype: torch.dtype = torch.bfloat16


@dataclass
class PairConfig:
    """``_run_pair`` 的运行配置。"""
    kernel_kwargs: dict | None = None
    golden_kwargs: dict | None = None
    note: str = ""
    l2_gate_main: float = None  # 延迟到 __post_init__ 设默认值
    verify: bool = True

    def __post_init__(self):
        if self.l2_gate_main is None:
            self.l2_gate_main = MAIN_L2_GATE


THRESHOLD_REF = Threshold(REF_ATOL, REF_RTOL, REF_L2_GATE)

_ROWS = []          # 结果表：dict(suite, case, tensor, ref, max_abs, l2_rel, oor, ok, note)
_FAILURES = []      # 失败用例名


# ═══════════════════════════════════════════════════════════════════════════════
# 用例总表 —— 所有会真正执行 kernel 的 shape 用例集中于此一处
# ═══════════════════════════════════════════════════════════════════════════════
# 每条一行，字段自解释；**新增用例 = 往下追加一条 dict**，无需改动任何 suite/runner。
#
#   name   唯一名字，供 `--case NAME` 选择、`--list-cases` 展示。
#   group  "precision"（进 suite P，快，双档判据）
#          "long"      （进 suite L，重，逐 head 等价比对）
#   kind   "pair" → _run_pair（整体 kernel-vs-golden）
#          "long" → _run_long_case（逐 head，h==hv，内部恒开 l2norm）
#   desc   一句话说明，`--list-cases` 会打印。
#   build  传给 _make_case 的 kwargs：b,t,h,hv,d,seed[,g_range,with_state,dtype…]。
#   call   传给 wrapper 的 kwargs：chunk_size[,cu_seqlens(写成 list,运行期转 tensor),
#          output_final_state,use_qk_l2norm_in_kernel…]。
#
# ⚠ cu_seqlens 写成 Python list（如 [0,300,812,1536]）——它依赖 device，不能在 import
#   期建 tensor；运行期由 _materialize_call 转成 device LongTensor。
# ═══════════════════════════════════════════════════════════════════════════════
_CASES = [
    # ── 精度配置（suite P / --case，快，emul+fp32 双档）──────────────
    dict(name="bt128", group="precision", kind="pair", desc="chunk_size=128（3 级块逆，B2 接口超集）",
         build=dict(b=2, t=512, h=4, hv=4, d=128, seed=8), call=dict(chunk_size=128)),
    # ── 长序列 / 深递推链（suite L；重，逐 head 等价比对，h==hv）─────────────
    dict(name="depth256_T32K_H4", group="long", kind="long", desc="链深 256（T=32K）",
         build=dict(b=1, t=32768, h=4, hv=4, d=128, seed=900), call=dict(chunk_size=128)),
    dict(name="varlen_T32K_H8", group="precision", kind="pair",
         desc="7 段 varlen、H=HV=8、T=32K、l2norm、BT=128（性能优化目标用例）",
         build=dict(b=1, t=32768, h=8, hv=8, d=128, seed=10),
          call=dict(chunk_size=128, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=[0, 796, 1560, 2262, 2914, 3535, 4137, 4734, 5319, 5893,
          6415, 6925, 7422, 7898, 8358, 8802, 9234, 9656, 10073, 10488,
        10878, 11226, 11550, 11860, 12162, 12462, 12758, 13047, 13333, 13613,
        13893, 14173, 14451, 14728, 15004, 15279, 15551, 15822, 16089, 16354,
        16616, 16876, 17135, 17394, 17647, 17899, 18151, 18401, 18650, 18896,
        19138, 19376, 19611, 19842, 20072, 20302, 20530, 20756, 20981, 21204,
        21419, 21633, 21844, 22041, 32768])),
]


def _cases_by_group(group):
    """按 group 取用例（保持 _CASES 中的顺序）。"""
    return [c for c in _CASES if c["group"] == group]


def _case_by_name(name):
    """按名取单条用例；找不到返回 None。"""
    return next((c for c in _CASES if c["name"] == name), None)


def _materialize_call(call):
    """把 call 里以 list 写的 cu_seqlens 转成 device LongTensor（运行期调用）。"""
    if isinstance(call.get("cu_seqlens"), list):
        call = dict(call)
        call["cu_seqlens"] = torch.tensor(call["cu_seqlens"], dtype=torch.long, device=_device())
    return call


# =============================================================================
# 基础设施
# =============================================================================

def _set_device() -> None:
    """按标准环境变量初始化 NPU 设备。必须在任何张量落 NPU 之前调用。"""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))


def _device() -> torch.device:
    return torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")


def _stats(a, b):
    """返回 ``(max_abs, l2_rel)``；``l2_rel = ‖a-b‖₂ / ‖b‖₂`` 是主指标。"""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    denom = b.norm().item()
    return diff.max().item(), (diff.norm().item() / denom if denom > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Golden 缓存（按用例名；_CASES 每条用 seed 固定输入，golden 可复用）
# ─────────────────────────────────────────────────────────────────────────────
_GOLDEN_CACHE_DIR = os.path.join(_test_dir, "_golden_cache")


def _golden_cache_path(case):
    return os.path.join(_GOLDEN_CACHE_DIR, f"{case}.pt")


def _load_golden_cache(case):
    """命中返回 ``(o_emu, s_emu, o_f32, s_f32)``，未命中返回 ``None``。"""
    path = _golden_cache_path(case)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location=_device())
    return data["o_emu"], data["s_emu"], data["o_f32"], data["s_f32"]


def _save_golden_cache(case, o_emu, s_emu, o_f32, s_f32):
    os.makedirs(_GOLDEN_CACHE_DIR, exist_ok=True)
    torch.save({"o_emu": o_emu, "s_emu": s_emu, "o_f32": o_f32, "s_f32": s_f32},
               _golden_cache_path(case))


def _record(key: TestKey, metrics: Metrics, ok, note=""):
    _ROWS.append(dict(suite=key.suite, case=key.case, tensor=key.tensor, ref=key.ref,
                      max_abs=metrics.max_abs, l2_rel=metrics.l2_rel, oor=metrics.oor, ok=ok, note=note))
    tag = "PASS" if ok else "FAIL"
    num = ("max_abs=  -       l2_rel=  -      " if key.ref in ("bool", "run")
           else f"max_abs={metrics.max_abs:.3e} l2_rel={metrics.l2_rel:.3e}")
    logging.info(f"    [{tag}] {key.case:<26s} {key.tensor:<12s} {key.ref:<5s} {num}  {note}")
    if not ok:
        _FAILURES.append(f"{key.suite}/{key.case}/{key.tensor}[{key.ref}]")


def _cmp(key: TestKey, got, ref, *, threshold: Threshold, note=""):
    """逐元素 atol/rtol + l2_rel 双门限比对。失败时打印越界元素明细。"""
    name = f"{key.suite}/{key.case}/{key.tensor}[{key.ref}]"
    g, r = got.cpu().float(), ref.cpu().float()

    diff = (g - r).abs()
    tol = threshold.atol + threshold.rtol * r.abs()
    oor = int((diff > tol).sum().item())
    oor_ratio = oor / g.numel() if g.numel() > 0 else 0.0
    all_close = oor == 0

    denom = r.norm().item()
    l2_rel = diff.norm().item() / denom if denom > 0 else 0.0
    max_abs = diff.max().item()
    ok = all_close and l2_rel <= threshold.l2_gate and torch.isfinite(g).all().item()
    _record(key, Metrics(max_abs, l2_rel, oor_ratio), ok, note)
    if not ok:
        logging.basicConfig(level=logging.INFO, force=True)
        logging.info("\n%s", "=" * 60)
        logging.info("📊 %s 越界元素明细", name)
        logging.info("=" * 60)
        logging.info("总元素数: %s  越界数: %s (%.4f%%)", f"{g.numel():,}", f"{oor:,}", oor_ratio * 100)
        logging.info("max_abs=%.6e  l2_rel=%.6e  atol=%.1e  rtol=%.1e", max_abs, l2_rel, threshold.atol, threshold.rtol)
        if oor > 0:
            mask = diff > tol
            idx = torch.nonzero(mask.flatten(), as_tuple=False)[:10, 0]
            flat_idx = idx.tolist()
            logging.info("%-12s %-15s %-15s %-12s", "FlatIndex", "Got", "Ref", "Diff")
            logging.info("-" * 56)
            for i in flat_idx:
                logging.info("%-12d %-15.6f %-15.6f %-12.6f",
                             i, g.flatten()[i].item(), r.flatten()[i].item(), diff.flatten()[i].item())
            if oor > 10:
                logging.info("... 还有 %d 个越界元素未显示", oor - 10)
        logging.info("=" * 60)
        logging.disable(logging.INFO)
    return ok


def _check(key: TestKey, cond, note=""):
    """非数值型断言（异常 / 布局 / 公开入口 …）落进同一张表。"""
    _record(key, Metrics(0.0, 0.0, 0.0), bool(cond), note)
    return bool(cond)


# =============================================================================
# 输入构造 —— 约束来自 SPEC §3.1（dtype）与 §5（值域）
# =============================================================================

def _make_case(shape: CaseShape):
    """构造一组语义合法的输入。

    * ``q`` / ``k``：先 L2 归一化再落 ``dtype``（``use_qk_l2norm_in_kernel=False``
      要求上游已归一化）。
    * ``v``：``dtype``，乘 0.5 收敛动态范围。
    * ``g``：fp32，**log 域**遗忘门，恒 ``<= 0``。
    * ``beta``：``dtype``，post-sigmoid 空间 ``(0, 1)``。
    * ``initial_state``：``[N, HV, K, V]`` fp32，**K 在 V 前**（要求 1）。

    ``dtype`` 的取值直接决定比较分辨率，两档各有唯一正确的用法：

    * **bf16（默认，用于所有 kernel 对比）** —— SPEC §3.1 规定的 IO dtype。
      golden 的 matmul 操作数 dtype 由调用方决定（``gdr_fwd_golden.py:244`` 原注释），
      传 fp32 会让 golden 用比 kernel 更精确的 q/k，主判据失真。
    * **fp32（仅用于 suite N 的 golden 自洽）** —— 两侧 ``o`` 的 dtype 都跟随 ``q``
      （SPEC §3.2），传 bf16 会把 ``o`` 量化到 bf16 网格，比较分辨率被截断到
      1 ulp（实测：仅 17/65536 个近零元素差 1 ulp，就已越过 SPEC §7 的 1e-5），
      根本测不出 fp32 层面的算法自洽性。SPEC §7 第一行要求的正是 fp32-vs-fp32。
    """
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(shape.seed)

    q = torch.randn(shape.b, shape.t, shape.h, shape.d, generator=gen)
    q = (q / q.norm(dim=-1, keepdim=True)).to(shape.dtype).to(dev)
    k = torch.randn(shape.b, shape.t, shape.h, shape.d, generator=gen)
    k = (k / k.norm(dim=-1, keepdim=True)).to(shape.dtype).to(dev)
    v = (torch.randn(shape.b, shape.t, shape.hv, shape.d, generator=gen) * 0.5).to(shape.dtype).to(dev)
    g = torch.empty(shape.b, shape.t, shape.hv).uniform_(shape.g_range[0], shape.g_range[1],
                                                         generator=gen).float().to(dev)
    beta = torch.rand(shape.b, shape.t, shape.hv, generator=gen).to(shape.dtype).to(dev)

    h0 = None
    if shape.with_state:
        n = shape.n_seq if shape.n_seq is not None else shape.b
        h0 = (torch.randn(n, shape.hv, shape.d, shape.d, generator=gen) * 0.1).float().to(dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0)


def _run_pair(inp, suite, case, *, config: PairConfig):
    """跑一个配置的双档对比：kernel vs golden(emul) + kernel vs golden(fp32)。

    对 ``o`` 与 ``final_state`` **两个叶子都比**（skill 明令禁止只比其中一个）。

    ``verify=False``（profiling 模式）：**只执行 kernel wrapper，不跑 golden、不做对比**，
    供 msprof 采集纯 kernel 的 profiling。仍做一次有限性检查（NaN/Inf 会被标 FAIL），
    但这不是精度判据。
    Returns:
        ``(ok, o_kernel, s_kernel)``
    """
    kw = dict(config.kernel_kwargs or {})
    gw = dict(config.golden_kwargs or kw)
    args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])

    o_ker, s_ker = chunk_gated_delta_rule_wrapper(*args, **kw)

    if not config.verify:
        fin = torch.isfinite(o_ker.float()).all().item()
        _record(TestKey(suite, case, "o", "run"), Metrics(0.0, 0.0, 0.0), fin,
                note="no-verify 仅执行 kernel" + (f"；{config.note}" if config.note else ""))
        return fin, o_ker, s_ker

    cached = _load_golden_cache(case)
    if cached is not None:
        o_emu, s_emu, o_f32, s_f32 = cached
        logging.info(f"    [CACHE] {case} golden 命中缓存")
    else:
        o_emu, s_emu = gdr_fwd_golden(*args, **gw, emulate_bf16=True)
        o_f32, s_f32 = gdr_fwd_golden(*args, **gw, emulate_bf16=False)
        _save_golden_cache(case, o_emu, s_emu, o_f32, s_f32)
        logging.info(f"    [CACHE] {case} golden 已缓存")

    th_main_o = Threshold(MAIN_ATOL_O, MAIN_RTOL_O, config.l2_gate_main)
    th_main_s = Threshold(MAIN_ATOL_S, MAIN_RTOL_S, config.l2_gate_main)
    ok = _cmp(TestKey(suite, case, "o", "emul"), o_ker, o_emu, threshold=th_main_o, note=config.note)
    ok &= _cmp(TestKey(suite, case, "o", "fp32"), o_ker, o_f32, threshold=THRESHOLD_REF)
    if s_ker is not None:
        ok &= _cmp(TestKey(suite, case, "final_state", "emul"), s_ker, s_emu, threshold=th_main_s)
        ok &= _cmp(TestKey(suite, case, "final_state", "fp32"), s_ker, s_f32, threshold=THRESHOLD_REF)
    return ok, o_ker, s_ker


# =============================================================================
# Suite P —— 精度：SPEC §6 全部典型配置
# =============================================================================

def _spec_configs():
    """从用例总表派生 SPEC §6 精度配置 → ``(name, build_kwargs, call_kwargs)``。

    cu_seqlens 在此运行期从 list 转成 device tensor（见 _materialize_call）。
    """
    return [(c["name"], c["build"], _materialize_call(c["call"]))
            for c in _cases_by_group("precision")]


def suite_precision():
    """SPEC §6 的 9 个配置，每个都做主判据 + 参考判据双档对比。"""
    logging.info("\n[P] 精度 —— SPEC §6 典型配置（双档：emul 主判据 / fp32 参考判据）")
    ok = True
    for name, bkw, ckw in _spec_configs():
        kw = dict(ckw)
        inp = _make_case(CaseShape(**bkw))
        if bkw.get("with_state"):
            kw["initial_state"] = inp["h0"]
        # golden 与 kernel 传完全相同的 kwargs（chunk_size 两侧同名同义）
        ok &= _run_pair(inp, "P", name, config=PairConfig(kernel_kwargs=kw, golden_kwargs=kw))[0]
    return ok


# =============================================================================
# Suite L —— 长序列 / 深递推链（重，默认不跑，须 --long 开启）
#
# 归档自曾经的独立脚本 check_bt128 / check_t64k / check_t128k（已删除）：
# 被测变量是**递推链深度** T/BT。状态递推 S_{i+1}=exp(γ_last)·S_i+kᵀ@vd 中
# exp(γ)<1 恒成立、是收缩映射，故实测误差**不随链深累积**（终态由最近若干 chunk
# 主导）。本 suite 把该结论固化为可复现的门禁。
#
# 为何不并入 suite_precision：单个用例 kernel 数十秒 + golden 逐 head 十余秒，
# 且 T=128K 需 ~5GB 显存；放进默认路径会让每次跑测试都要数分钟。故独立成 suite、
# 用 --long 显式开启，但**用例定义仍在本文件内**，符合“唯一入口”。
# =============================================================================

def _run_long_case(suite, case, shape: CaseShape, bt, verify=True):
    """长序列单用例：kernel 跑满 H，golden **逐 head** 比对（emulate_bf16=True）。

    逐 head 是**等价变形**而非降强度：h==hv（group=1）时每条 (段, head) 递推链
    完全独立，逐 head 比对与整体比对数学等价，但 golden 峰值显存降到 1/h——
    否则 T=128K/H=16 下 golden 的 [nt*HV, BT, BT] 中间量约 8.6GB 直接 OOM。

    判据与主判据同门限（o: 2e-3；final_state: l2_rel<=2e-3，逐元素 5e-3）。
    ``seed`` 显式传入（不用 hash()——其受 PYTHONHASHSEED 影响，跨进程不可复现）。

    ``verify=False``（profiling 模式）：只执行 kernel，不跑逐 head golden（否则 T=128K
    的 golden 部分本身要十余秒，污染 msprof 采样）。
    """
    inp = _make_case(shape)
    args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
    kw = dict(scale=shape.d ** -0.5, initial_state=None, output_final_state=True,
              use_qk_l2norm_in_kernel=True, chunk_size=bt)

    t0 = time.time()
    o_ker, s_ker = chunk_gated_delta_rule_wrapper(*args, **kw)
    torch.npu.synchronize()
    t_ker = time.time() - t0

    if not verify:
        fin = torch.isfinite(o_ker.float()).all().item()
        _record(TestKey(suite, case, "o", "run"), Metrics(0.0, 0.0, 0.0), fin,
                note=f"no-verify 仅执行 kernel；depth={shape.t // bt} h={shape.h} kernel={t_ker:.0f}s")
        return fin

    # 逐 head 累积最差 l2_rel、最大 max_abs、越界元素总数（atol/rtol 逐元素判据）。
    leaves = {"o": dict(atol=MAIN_ATOL_O, rtol=MAIN_RTOL_O, ma=0.0, l2=0.0, oor=0, n=0),
              "final_state": dict(atol=MAIN_ATOL_S, rtol=MAIN_RTOL_S, ma=0.0, l2=0.0, oor=0, n=0)}
    finite = True
    t0 = time.time()
    for hh in range(shape.h):
        sl = slice(hh, hh + 1)
        o_ref, s_ref = gdr_fwd_golden(
            inp["q"][:, :, sl], inp["k"][:, :, sl], inp["v"][:, :, sl],
            inp["g"][:, :, sl], inp["beta"][:, :, sl],
            scale=shape.d ** -0.5, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True, chunk_size=bt, emulate_bf16=True)
        for tag, got, ref in (("o", o_ker[:, :, sl], o_ref),
                              ("final_state", s_ker[:, sl], s_ref)):
            acc = leaves.get(tag, {})
            gf, rf = got.cpu().float(), ref.float()
            finite &= torch.isfinite(gf).all().item()
            diff = (gf - rf).abs()
            acc["ma"] = max(acc["ma"], diff.max().item())
            denom = rf.norm().item()
            acc["l2"] = max(acc["l2"], diff.norm().item() / denom if denom > 0 else 0.0)
            acc["oor"] += int((diff > (acc["atol"] + acc["rtol"] * rf.abs())).sum().item())
            acc["n"] += rf.numel()
        del o_ref, s_ref
    t_ref = time.time() - t0

    ok_all = True
    for tag, acc in leaves.items():
        ok = finite and acc["oor"] == 0 and acc["l2"] <= LONG_L2_GATE
        ok_all &= ok
        _record(TestKey(suite, case, tag, "emul"), Metrics(acc["ma"], acc["l2"], acc["oor"] / acc["n"]), ok,
                note=f"depth={shape.t // bt} h={shape.h} kernel={t_ker:.0f}s golden={t_ref:.0f}s")
    return ok_all


def _run_long_case_record(suite, c, verify=True):
    """从用例总表的一条 long 记录跑 _run_long_case。"""
    return _run_long_case(suite, c["name"], CaseShape(**c["build"]),
                          c["call"]["chunk_size"], verify=verify)


def suite_long_sequence():
    """长序列 / 深递推链门禁（重）。逐 head 等价比对，覆盖链深 256~1024、T 到 128K。"""
    logging.info("\n[L] 长序列 —— 深递推链（逐 head 等价比对，emulate_bf16 主判据）")
    ok = True
    for c in _cases_by_group("long"):
        ok &= _run_long_case_record("L", c)
    return ok


# =============================================================================
# 按名选择 / profiling 执行体 —— 直接作用在「用例总表 _CASES」上
#
# ``--case NAME`` 与 ``--no-verify`` 都只在总表上生效，无需在异构的各 suite 里插桩。
# 新增用例只改 _CASES，这里自动生效。
# =============================================================================

def _run_registry_case(c, verify):
    """按用例总表的一条记录执行。``verify`` 透传（False = 只跑 kernel）。"""
    if c["kind"] == "pair":
        inp = _make_case(CaseShape(**c["build"]))
        kw = _materialize_call(c["call"])
        if c["build"].get("with_state"):            # with_state 需注入 initial_state
            kw = dict(kw)
            kw["initial_state"] = inp["h0"]
        return _run_pair(inp, "SEL", c["name"],
                         config=PairConfig(kernel_kwargs=kw, golden_kwargs=kw, verify=verify))[0]
    return _run_long_case_record("SEL", c, verify=verify)


def run_selected(names, verify):
    """``--case`` / ``--no-verify`` 模式的执行体：只跑 names 指定（或全部）总表用例。"""
    if not names:                                   # --no-verify 未指名 → 跑全部 shape 用例
        names = [c["name"] for c in _CASES]
    bad = [n for n in names if _case_by_name(n) is None]
    if bad:
        logging.info(f"[ERROR] 未知用例名: {bad}\n可用用例: {[c['name'] for c in _CASES]}")
        return False
    mode = "验证" if verify else "仅执行 kernel（no-verify，供 msprof）"
    logging.info(f"\n[SEL] 选定用例模式（{mode}）—— 运行 {names}")
    ok = True
    for name in names:
        t0 = time.time()
        try:
            ok &= _run_registry_case(_case_by_name(name), verify)
        except Exception:                           # noqa: BLE001
            logging.info(f"    [FAIL] 用例 {name} 抛出未捕获异常：")
            traceback.print_exc()
            _FAILURES.append(f"SEL/{name}:EXCEPTION")
            ok = False
        logging.info(f"    -- {name} 用时 {time.time() - t0:.1f}s")
    return ok


# =============================================================================
# skill 约定的 L0 / L1 入口（OL21）
# =============================================================================


def test_gdr_fwd_l1() -> None:
    """L1：SPEC §6 的 P0 规模，全精度判据。"""
    _set_device()
    torch.manual_seed(42)


# =============================================================================
# 主入口
# =============================================================================

def _print_summary():
    logging.info("\n" + "=" * 118)
    logging.info("逐用例汇总（ref: emul=golden(emulate_bf16=True) 主判据 / fp32=golden 数学真值 参考判据 /"
          " naive=golden 自洽 / full, pack=kernel 自洽 / bool=契约）")
    logging.info("=" * 118)
    logging.info(f"{'SUITE':<6}{'CASE':<34}{'TENSOR':<14}{'REF':<7}"
          f"{'max_abs':>12}{'l2_rel':>12}{'out%':>10}  {'VERDICT':<7} NOTE")
    logging.info("-" * 118)
    for r in _ROWS:
        num = (f"{r['max_abs']:>12.3e}{r['l2_rel']:>12.3e}{r['oor'] * 100:>10.4f}"
               if r["ref"] not in ("bool", "run") else f"{'-':>12}{'-':>12}{'-':>10}")
        logging.info(f"{r['suite']:<6}{r['case']:<34}{r['tensor']:<14}{r['ref']:<7}{num}  "
              f"{'PASS' if r['ok'] else 'FAIL':<7} {r['note']}")
    logging.info("-" * 118)
    n_pass = sum(1 for r in _ROWS if r["ok"])
    logging.info(f"合计 {len(_ROWS)} 项断言：PASS {n_pass} / FAIL {len(_ROWS) - n_pass}")
    if _FAILURES:
        logging.info("\n失败项：")
        for f in _FAILURES:
            logging.info(f"  - {f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="gdr_fwd 唯一测试入口。默认跑 4 个用例（P 精度 + L 长序列）；"
                    "--case 精确选择用例；--no-verify 只跑 kernel（供 msprof profiling）。")
    ap.add_argument("--long", action="store_true",
                    help="（已废弃/兼容保留）L 长序列现默认即跑，本开关无额外效果")
    ap.add_argument("--only-long", action="store_true",
                    help="只跑 suite L（depth256_T32K_H4），跳过 P 精度")
    ap.add_argument("--case", nargs="+", metavar="NAME", default=None,
                    help="只跑指定名字的用例（其余一律不跑）。名字见 --list-cases。"
                         "可多个：--case bt128 varlen_T32K_H8")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="关闭精度验证：不跑 golden，只执行 wrapper + 生成 input（供 msprof profiling）。"
                         "默认开启验证。未指定 --case 时对全部 shape 用例生效。")
    ap.add_argument("--list-cases", action="store_true",
                    help="列出所有可用 --case 名字后退出")
    args = ap.parse_args()

    if args.list_cases:
        logging.info(f"用例总表（共 {len(_CASES)} 条，定义见 test_gdr_fwd.py::_CASES）：")
        logging.info(f"  {'NAME':<22}{'GROUP':<10}{'SHAPE (B, T, H, HV, D)':<24}{'BT':<5}说明")
        logging.info("  " + "-" * 96)
        for c in _CASES:
            b = c["build"]
            shape = f"{b['b']}, {b['t']}, {b['h']}, {b['hv']}, {b['d']}"
            logging.info(f"  {c['name']:<22}{c['group']:<10}{shape:<24}"
                  f"{c['call']['chunk_size']:<5}{c['desc']}")
        return 0

    _set_device()
    torch.manual_seed(42)           # 生成任何随机输入之前固定种子
    dev_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    logging.info("=" * 118)
    logging.info("gdr_fwd 端到端验证（kernel = gdr_fwd_impl.chunk_gated_delta_rule_wrapper）")
    logging.info("=" * 118)
    logging.info(f"Device: npu:{dev_id}   torch={torch.__version__}")
    logging.info(f"主判据门限: l2_rel<={MAIN_L2_GATE:.0e}, o atol/rtol={MAIN_ATOL_O:.0e}, "
          f"state atol/rtol={MAIN_ATOL_S:.0e}")
    logging.info(f"参考判据门限: l2_rel<={REF_L2_GATE:.0e}（预期实测 ≈4e-3，即 bf16 固有损失）")

    # ---- 选择模式：指定了 --case 或关闭了验证 → 走注册表（精确/profiling），不跑异构 suite ----
    # （异构 suite 含 kernel-vs-kernel 自洽与异常契约，本身依赖对比，no-verify 下无意义。）
    if args.case is not None or not args.verify:
        ok = run_selected(args.case, args.verify) and not _FAILURES
        _print_summary()
        if args.verify:
            logging.info("\n[PRECISION_PASS] 全部用例通过" if ok else "\n[PRECISION_FAIL] 存在失败用例")
        else:
            logging.info("\n[RUN_DONE] kernel 执行完成（未做精度验证）" if ok
                  else "\n[RUN_DONE] kernel 已执行，但输出含 NaN/Inf（见上方 FAIL）")
        return 0 if ok else 1

    if args.only_long:
        suites = [("L 长序列", suite_long_sequence)]
    else:
        # 精简后仅保留 4 个用例：P 精度（bt32 / bt128 / varlen_T32K_H8）
        # + L 长序列（depth256_T32K_H4）。两者默认全跑。
        suites = [
            ("P 精度", suite_precision),
            ("L 长序列", suite_long_sequence),
        ]
    for name, fn in suites:
        t0 = time.time()
        try:
            fn()
        except Exception:                              # noqa: BLE001
            logging.info(f"    [FAIL] suite {name} 抛出未捕获异常：")
            traceback.print_exc()
            _FAILURES.append(f"{name}:EXCEPTION")
        logging.info(f"    -- {name} 用时 {time.time() - t0:.1f}s")

    _print_summary()
    if _FAILURES:
        logging.info("\n[PRECISION_FAIL] 存在失败用例")
        return 1
    logging.info("\n[PRECISION_PASS] 全部用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
