import logging
import os
import torch
import torch_npu
import pypto_pro.language as pl
from kda_test_config import HV, C, DEVICE

B = 1
T = 256

UB_T = 0x00000
UB_O = 0x10000


@pl.jit()
def load_strided_kernel(
    src: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    head_id: pl.DT_INT32,
    t_base: pl.DT_INT32,
):
    t_in = pl.make_tile(
        pl.TileType(shape=[1, C], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=UB_T, size=4096)
    t_out = pl.make_tile(
        pl.TileType(shape=[1, C], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=UB_O, size=4096)

    with pl.section_vector():
        pl.load(t_in, src, [0, t_base, head_id], order=[0, 1])
        pl.move(t_out, t_in)
        pl.store(out, t_out, [0, 0])


def test_strided_load():
    device = DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    src = torch.arange(B * T * HV, dtype=torch.float16).reshape(B, T, HV).to(device)
    out = torch.zeros(1, C, device=device, dtype=torch.float16)

    head_id = 2
    t_base = 0

    load_strided_kernel(src, out, head_id, t_base)
    torch.npu.synchronize()

    expected = src[0, t_base:t_base + C, head_id].unsqueeze(0)
    npu = out.cpu()
    exp = expected.cpu()

    diff = (npu - exp).abs().max().item()
    logging.info("tile_dims=[1,2] from [B,T,HV] load [1,C] at [0,%d,%d]", t_base, head_id)
    logging.info("  expected[0,:8] = %s", exp[0, :8].tolist())
    logging.info("  npu[0,:8]      = %s", npu[0, :8].tolist())
    logging.info("  max diff: %.3e", diff)

    if diff < 1e-3:
        logging.info("  PASS")
    else:
        logging.info("  FAIL (diff=%.3e)", diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_strided_load()
