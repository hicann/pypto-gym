import os
import sys
import logging

import torch
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _HERE
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pypto_gym.ops.pypto_pro.experimental.ops_transformer.kda.kda_common import (
    C, K, V, HC, K_DIM, V_DIM, DEVICE,
    make_cu_seqlens_tensor, build_chunk_tables, alloc_chunk_h_workspaces,
)

CHUNK_SIZE = C
HV = 4

DEVICE_ID = os.environ.get('TILE_FWK_DEVICE_ID', '0')

TEST_SHAPES = [
    (8192, None),
    (4117, None),
    (1024, [0,63,156,618,799,1024]),
    (3118, [0,129,197,315,618,3113,3118]),
    (385, None),
    (512, [0, 256, 512]),
    (768, [0, 128, 384, 768]),
]


def require_a5(device=None):
    if device is None:
        device = DEVICE
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def make_kda_base_inputs(T, device):
    """Canonical KDA test inputs shared across all single-stage and e2e tests.

    Returns (q, k, v, g_log, beta_sig, scale) on ``device``.  Uses seed 0
    and matches the distribution previously hardcoded in test_kda_e2e so
    that single-kernel tests and the e2e test exercise identical data.

    Shapes:
        q:        [1, T, HV, K_DIM]  fp32  L2-normalised (NOT pre-scaled)
        k:        [1, T, HV, K_DIM]  fp32  L2-normalised
        v:        [1, T, HV, V_DIM]  fp32  randn
        g_log:    [1, T, HV, K_DIM]  fp32  -rand  (log-space decay gates)
        beta_sig: [1, T, HV]         fp32  sigmoid(randn)
        scale:    float               K_DIM**-0.5
    """
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(
        torch.randn(1, T, HV, K_DIM, dtype=torch.float32), dim=-1, p=2
    ).to(device)
    k = torch.nn.functional.normalize(
        torch.randn(1, T, HV, K_DIM, dtype=torch.float32), dim=-1, p=2
    ).to(device)
    v = torch.randn(1, T, HV, V_DIM, dtype=torch.float32).to(device)
    g_log = -torch.rand(1, T, HV, K_DIM, dtype=torch.float32).to(device)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV, dtype=torch.float32)).to(device)
    scale = K_DIM ** -0.5
    return q, k, v, g_log, beta_sig, scale


def make_tril_mask(C, device, diagonal=0):
    return torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=diagonal).to(device)


def run_main(title, shapes, run_case_fn):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(title)
    logging.info("=" * 60)
    idx = None
    device = DEVICE
    for arg in sys.argv[1:]:
        if arg.isdigit():
            idx = int(arg) - 1
        elif arg == "cpu":
            device = "cpu"
    selected = [shapes[idx]] if idx is not None else shapes
    for T, cu in selected:
        cu_str = f"cu={cu}" if cu else "cu=None"
        run_case_fn(T, cu, f"T={T}, {cu_str}", device)
