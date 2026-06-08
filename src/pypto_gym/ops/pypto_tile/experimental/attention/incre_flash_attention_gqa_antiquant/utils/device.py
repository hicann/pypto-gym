#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import os
from typing import Optional

import torch
import torch_npu

from .logger import create_logger


logger = create_logger(__name__)


def get_env_device_id() -> int:
    """Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        The device ID if valid.

    Raises:
        ValueError: If TILE_FWK_DEVICE_ID is not set or invalid.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logger.info("If no NPU environment is available, set --run_mode sim "
                    "to run in simulation mode;")
        logger.info("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        logger.info("Please set it before running this example:")
        logger.info("  export TILE_FWK_DEVICE_ID=0")
        raise ValueError("Please set TILE_FWK_DEVICE_ID.")

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError as e:
        error_msg = (f"ERROR: TILE_FWK_DEVICE_ID must be an integer, "
                     f"got: {os.environ['TILE_FWK_DEVICE_ID']}")
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_device(device_id: Optional[int] = None, run_mode: str = "npu") -> str:
    """Get the appropriate device string for computation.

    Args:
        device_id: Explicit device ID. If None, will read from environment.
        run_mode: Execution mode - "npu" for hardware, "sim" for simulation.

    Returns:
        Device string (e.g., "npu:0" or "cpu").
    """
    if device_id is not None:
        cur_device_id = device_id
    else:
        cur_device_id = get_env_device_id()

    torch.npu.set_device(int(cur_device_id))
    device = f"npu:{cur_device_id}" if (run_mode == "npu" and cur_device_id is not None) else "cpu"
    return device