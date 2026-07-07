# {op_name} — PyPTO-Pro Kernel 实现模板
#
# 使用说明：所有 {{...}} 占位符需替换为 DESIGN.md 中的实际值。
# 开发流程见 SKILL.md：以 DESIGN.md（施工图）和 EXPLORE_REPORT.md（API 约束/样例模式）为主要依据，
# PRO_MATERIAL_INDEX.md / API 文档 / pro_ops 样例 / 教程按需取用。遇到编译/精度问题查阅 references/pitfalls.md。

import logging
import torch
import torch_npu
import pypto_pro.frontend as fe
import pypto_pro.language as pl

# ============================================================================
# 动态维度声明 (来自 DESIGN.md §0)
# ============================================================================
{DYN_DIM_DECLS}  # 例如: B = pl.DynVar('B'), N = pl.DynVar('N'), S = pl.DynVar('S')

# ============================================================================
# 模块级常量 (来自 DESIGN.md §2 Tile 规划)
# ============================================================================
{TILE_CONSTANTS}  # 例如: TS = 128, TD = 128, SCALE = 1.0 / math.sqrt(64)

# ============================================================================
# Kernel 函数
# ============================================================================
# jit 装饰器二选一（按 DESIGN.md §3 分配方式决定；pro_ops 样例中无 codegen_mode 参数）：
#   手动 sync 模式：       @pl.jit()
#   auto_mutex 模式：      @pl.jit(auto_mutex=True)
@pl.jit(auto_mutex=True)
def {op}_kernel(
    # 输入/输出签名来自 DESIGN.md §0 I/O 规格
    {KERNEL_PARAMS}):
    # 例如: inp1: pl.Tensor[[S, D_PAD], pl.FP16],
    #       out: pl.Tensor[[S, D_PAD], pl.FP16],

    # ---- Tile 声明 (逐条复制 DESIGN.md §3 UB 地址映射表) ----
    # 首选 make_tile_group + auto_mutex，由框架自动管理 buffer 切换与 core 内互斥。
    # 次选 make_tile（手动 addr/size）。
    {TILE_DECLS}
    # 例如:
    # tile_type = pl.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
    #                         valid_shape=[-1, -1])
    # 首选: a_db  = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])
    # 次选: tile_a = pl.make_tile(tile_type, addr=0x00000, size=TS * TD * 2)

    # ---- SPMD 原语获取 (位置按 DESIGN.md §4：多 section 在外，单 section 在内) ----
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()

    # ---- Section 声明 (来自 DESIGN.md §0 Phase 划分) ----
    with {SECTION_TYPE}():
        # ---- 循环结构 (来自 DESIGN.md §4) ----
        {TILE_LOOP_NESTING}
        # 例如:
        # for work_id in pl.range(core_id, {TOTAL_WORK}, num_cores):  # 分核方式来自 §5
        #     # sync: {同步点占位} — 按 §4-6 标注，步骤 5 填入具体 pl.system.* API
        #     # Phase 实现 — 步骤 5 逐行翻译（含 §7 尾块处理、§1 API 调用）

    return {OUTPUT_VAR}


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
    out = torch.zeros_like(inp)
    block_dim = {BLOCK_DIM}

    {op}_kernel[None, block_dim](inp, out)
    torch.npu.synchronize()

    out_ref = {op}_golden(inp)
    torch.testing.assert_close(out, out_ref, rtol={RTOL}, atol={ATOL})
    logging.info("{op} aligned PASS")


def test_{op}_tail():
    """单轴尾块 case：一个 tile 维度存在尾块，如 [TILE_A + 22, TILE_B]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_WITH_TAIL}, device=device, dtype={DTYPE})
    out = torch.zeros_like(inp)
    block_dim = {BLOCK_DIM}

    {op}_kernel[None, block_dim](inp, out)
    torch.npu.synchronize()

    out_ref = {op}_golden(inp)
    torch.testing.assert_close(out, out_ref, rtol={RTOL}, atol={ATOL})
    logging.info("{op} tail PASS")


def test_{op}_tail2d():
    """双轴尾块 case：两个 tile 维度均存在尾块，如 [TILE_A + 22, TILE_B - 30]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_WITH_TAIL_2D}, device=device, dtype={DTYPE})
    out = torch.zeros_like(inp)
    block_dim = {BLOCK_DIM}

    {op}_kernel[None, block_dim](inp, out)
    torch.npu.synchronize()

    out_ref = {op}_golden(inp)
    torch.testing.assert_close(out, out_ref, rtol={RTOL}, atol={ATOL})
    logging.info("{op} tail2d PASS")


def test_{op}_multitile():
    """跨多 tile + 尾块 case：动态轴跨越 2-3 个 tile 并带尾块，
    验证跨 tile 循环与状态持久化，如 [2 * TILE_A + 13, TILE_B - 7]"""
    from {op}_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)

    inp = torch.randn({SHAPE_MULTITILE}, device=device, dtype={DTYPE})
    out = torch.zeros_like(inp)
    block_dim = {BLOCK_DIM}

    {op}_kernel[None, block_dim](inp, out)
    torch.npu.synchronize()

    out_ref = {op}_golden(inp)
    torch.testing.assert_close(out, out_ref, rtol={RTOL}, atol={ATOL})
    logging.info("{op} multitile PASS")


if __name__ == "__main__":
    test_{op}_aligned()
    test_{op}_tail()
    test_{op}_tail2d()
    test_{op}_multitile()
    logging.info("All tests passed!")
