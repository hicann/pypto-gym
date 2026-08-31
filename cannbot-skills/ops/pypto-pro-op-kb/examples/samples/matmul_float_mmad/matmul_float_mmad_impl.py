# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# STATUS: VALIDATED on Ascend a5 (950) NPU — 2026-07-17. pl tile DSL (pypto_pro.language).
#   FP32 -> max_abs_diff = 0.000e+00 (bit-exact vs golden); PASS.
#   Modeled on the real pl matmul test t01 (python/tests/ut/block/frontend/a5/matmul/
#   test_matmul_api_basic.py:92-125): GM->L1(Mat)->L0A(Left,NZ)/L0B(Right,ZN)->matmul->L0C(Acc)->GM.
#   pl has NO RunMode.SIM -> validated ONLY on Tier-3 NPU (bisheng compile + run).
#   loops per test_doc_quickstart.py matmul_example for larger shapes.
#   NOT A DELIVERY SHAPE: the `*_wrapper` below is this sample's own driver so the
#   file runs standalone. Its host-side allocation, layout and synchronize calls are
#   harness, not a licensed pattern -- a delivered wrapper may call only torch.empty.
#   See ../../../constraints/wrapper-boundary.md. Copy the kernel, not the driver.
#   RE-STAMP BASIS (this PR): the edit that moved this hash was **text only** -- a
#   docstring/comment change with no executable difference, proved by comparing the
#   parsed AST with docstrings stripped before and after. The recorded result above was
#   NOT re-measured on a board. So this stamp asserts "semantically the same code as the
#   one that produced that record", which is weaker than `--stamp-validation-hashes`'s
#   normal meaning ("this exact code was just re-validated"). Re-run on target before
#   relying on it as a fresh result.
# VALIDATED-CODE-SHA256: 4728b84f81f7e40d8b38f708c3791aef03b730b020064e02985f69c51e631b20
import pypto_pro.language as pl
import torch
import torch_npu  # noqa: F401  required for NPU device init

M, K, N = 32, 16, 48  # smoke shapes (multiples of the 16 cube fractal)


@pl.jit(auto_mutex=True)
def matmul_float_mmad_kernel(
    a: pl.Tensor[[M, K], pl.DT_FP32],  # [M, K]  (= x)
    b: pl.Tensor[[K, N], pl.DT_FP32],  # [K, N]  (= y.T; wrapper transposes y[N,K])
    out: pl.Tensor[[M, N], pl.DT_FP32],  # [M, N]  = a @ b = x @ y.T, FP32 accum in L0C
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        ca, cb = a_l1.current(), b_l1.current()
        al, br, ac = a_l0a.current(), b_l0b.current(), c_l0c.current()
        pl.load(ca, a, [0, 0])
        pl.load(cb, b, [0, 0])  # element offsets (single tile)
        pl.move(al, ca)
        pl.move(br, cb)  # L1 -> L0A/L0B
        pl.matmul(ac, al, br)  # a @ b -> L0C
        pl.store(out, ac, [0, 0])


def matmul_float_mmad_wrapper(x, y):
    """contract z = x @ y.T. x:[M,K], y:[N,K] -> z:[M,N]. Pass y.T as the B operand."""
    b = y.t().contiguous()  # [K, N] = y.T
    out = torch.zeros((x.shape[0], y.shape[0]), dtype=torch.float32, device=x.device)
    matmul_float_mmad_kernel(x, b, out)  # plain call = single default core
    torch.npu.synchronize()
    return out
