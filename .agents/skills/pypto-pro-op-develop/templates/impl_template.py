# {op_name} — PyPTO-Pro Kernel 实现模板
#
# 使用说明：所有 {{...}} 占位符需替换为 DESIGN.md 中的实际值。
# 开发流程见 SKILL.md：以 DESIGN.md（施工图）和 EXPLORE_REPORT.md（API 约束/样例模式）为主要依据，
# PRO_MATERIAL_INDEX.md / API 文档 / 官方指定算子 / 教学文档按需取用。
#
# ⚠️ 两条性能强制（违反即性能不可接受）：
#   1. buffer 管理：需要 buffer 切换/轮转的 tile 一律用 make_tile_group + auto_mutex，
#      make_tile 仅限单次使用 scratch tile（不参与轮转），禁止 make_tile + 手动 sync 管 buffer 轮转
#   2. Vector 计算：Vector 数值计算用 vf.* 指令手写（在 section_vector() 内通过 @pl.vector_function 执行）
#      （@pl.vector_function 装饰器 或 @pl.inline + with pl.section_vf(): 块，以 vf API 文档为准）

import logging
import os
import sys
import torch
import torch_npu
import pypto_pro.language as pl


def _assert_precision(actual, *inputs, label="", **kwargs):
    """方案A精度校验（混合容差标准）。

    内部完成: CPU golden 计算 + precision_compare 对比 + PASS/FAIL 判定。
    阈值由 precision_compare 按 actual.dtype 自动查表，禁止外部覆盖。

    Args:
        actual: 算子输出 tensor（NPU 或 CPU）
        *inputs: 传给 golden_cpu 的位置参数（按 golden 签名顺序，tensor 类型）
        label: 测试标签（用于日志输出）
        **kwargs: 传给 golden_cpu 的关键字参数（如 dim、eps 等 scalar 参数）

    Raises:
        AssertionError: 精度不达标时抛出
    """
    # ⚠️ 这两个 import 必须留在函数体内，禁止提到模块顶层：
    # 算子的交付单元仅含 test_{op}.py + {op}_golden.py，不含 precision_compare.py /
    # {op}_golden_cpu.py（这两个是 dev-only 自测工具，只在 custom/<op>/ 本地自测用）。
    # 交付单元被作为模块加载时会执行所有顶层代码——顶层 import 会直接
    # ModuleNotFoundError，导致交付态全部 case 0 分。
    from precision_compare import check_precision
    from {op}_golden_cpu import {op}_golden_cpu
    inputs_cpu = [i.cpu() if hasattr(i, "cpu") else i for i in inputs]
    golden = {op}_golden_cpu(*inputs_cpu, **kwargs)
    actual_cpu = actual.cpu() if hasattr(actual, "cpu") else actual
    passed, summary = check_precision(actual_cpu, golden)
    if not passed:
        raise AssertionError(f"精度不达标: {summary}")
    if label:
        logging.info("[{}] PASS ({})".format(label, summary))
    return summary

# ============================================================================
# 动态维度声明 (来自 DESIGN.md §0)
# ============================================================================
{DYN_DIM_DECLS}  # 动态维度声明（具体 API 以 docs/ 和官方指定算子样例为准）

# ============================================================================
# 模块级常量 (来自 DESIGN.md §2 Tile 规划)
# ============================================================================
{TILE_CONSTANTS}  # 例如: TS = 128, TD = 128, SCALE = 1.0 / math.sqrt(64)

# ============================================================================
# Kernel 函数
# ============================================================================
# jit 装饰器二选一（按 DESIGN.md §3 分配方式决定；官方指定算子中无 codegen_mode 参数）：
#   手动 sync 模式：       @pl.jit()
#   auto_mutex 模式：      @pl.jit(auto_mutex=True)
@pl.jit(auto_mutex=True)
def {op}_kernel(
    # 输入/输出签名来自 DESIGN.md §0 I/O 规格
    {KERNEL_PARAMS}):
    # 例如: inp1: pl.Tensor[[S, D_PAD], pl.DT_FP16],
    #       out: pl.Tensor[[S, D_PAD], pl.DT_FP16],

    # ---- Tile 声明 (逐条复制 DESIGN.md §3 地址映射表) ----
    # ⚠️ 性能强制：优先使用 make_tile_group + auto_mutex，非必要不用 make_tile + 手动 sync。
    {TILE_DECLS}
    # 例如:
    # tile_type = pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    # a_db  = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])

    # ---- SPMD 原语获取 (位置按 DESIGN.md §4：多 section 在外，单 section 在内) ----
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()

    # ---- Section 声明 (来自 DESIGN.md §0 Phase 划分) ----
    with {SECTION_TYPE}():
        # ⚠️ 性能强制：Vector 数值计算用 @pl.vector_function + vf.* 手写（pl.* 计算API不得用于Vector数值计算）。
        # ---- 循环结构 (来自 DESIGN.md §4) ----
        {TILE_LOOP_NESTING}
        # 例如:
        # for work_id in pl.range(core_id, {TOTAL_WORK}, num_cores):  # 分核方式来自 §5
        #     # sync: {同步点占位} — 按 §4-6 标注，步骤 5 填入具体 pl.system.* API
        #     # Phase 实现 — 步骤 5 逐行翻译（含 §7 尾块处理、§1 API 调用）

    return {OUTPUT_VAR}


# ============================================================================
# 入口函数 — 外部调用入口（host 适配 + kernel launch）
# ============================================================================
def {op}_wrapper(inp: torch.Tensor) -> torch.Tensor:
    """host 适配 + kernel launch，外部调用入口。

    host 端仅做必要适配（输出分配 / num_cores 计算），核心计算全部在
    kernel 内完成。wrapper 只调用一次 kernel（核心原则 #3）。
    多输入算子按 DESIGN.md §0 I/O 规格扩展签名（如 layernorm 的 gamma/beta）。
    """
    out = torch.zeros_like(inp)
    block_dim = {BLOCK_DIM}  # num_cores 计算式来自 DESIGN.md §5
    {op}_kernel[None, block_dim](inp, out)
    torch.npu.synchronize()
    return out


# ============================================================================
# 测试函数 (至少 4 个 case：整除 / 单轴尾块 / 双轴尾块 / 跨多 tile+尾块)
# 详见 SKILL.md 步骤 6「测试 shape 选择」。跨 tile case 取能触发多 tile
# 迭代的最小规模即可，不必放大，避免拖慢编译/执行或触发 OOM。
# 单动态轴算子凑不出 4 个有区分度的 case 时，按 SKILL 例外说明处理。
# ============================================================================
def test_{op}_aligned():
    """整除 case：所有 tile 维度均可整除，如 [TILE_A, TILE_B]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_DIVISIBLE}, device=device, dtype={DTYPE})
    out = {op}_wrapper(inp)

    _assert_precision(out, inp, label="{op} aligned")


def test_{op}_tail():
    """单轴尾块 case：一个 tile 维度存在尾块，如 [TILE_A + 尾块余数, TILE_B]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_WITH_TAIL}, device=device, dtype={DTYPE})
    out = {op}_wrapper(inp)

    _assert_precision(out, inp, label="{op} tail")


def test_{op}_tail2d():
    """双轴尾块 case：两个 tile 维度均存在尾块，如 [TILE_A + 尾块余数, TILE_B - 尾块余数]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_WITH_TAIL_2D}, device=device, dtype={DTYPE})
    out = {op}_wrapper(inp)

    _assert_precision(out, inp, label="{op} tail2d")


def test_{op}_multitile():
    """跨多 tile + 尾块 case：动态轴跨越 2-3 个 tile 并带尾块，
    验证跨 tile 循环与状态持久化，如 [2~3 × TILE_A + 尾块余数, TILE_B - 尾块余数]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_MULTITILE}, device=device, dtype={DTYPE})
    out = {op}_wrapper(inp)

    _assert_precision(out, inp, label="{op} multitile")


if __name__ == "__main__":
    test_{op}_aligned()
    test_{op}_tail()
    test_{op}_tail2d()
    test_{op}_multitile()
    logging.info("All tests passed!")
