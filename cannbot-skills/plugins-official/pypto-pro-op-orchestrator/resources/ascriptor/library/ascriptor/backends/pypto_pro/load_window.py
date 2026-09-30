# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offset DMA destinations keep physical alias geometry and local valid extents."""


def load_window(emitter, op, name, tile):
    from .emit import PyptoGap

    shape = list(tile['shape'])
    capacity = list(tile['base_shape'])
    l1_band = tile['space'] == 'l1' and emitter._off_fold(tile['offs_ir'][0]) == 0
    partial_height = l1_band and shape[0] != capacity[0]
    if all(isinstance(x, int) for x in shape) and not partial_height:
        return emitter._strip_tile(op, name, tile), [], []
    if tile['space'] == 'l1' and isinstance(shape[1], int):
        capacity[1] = shape[1]
    if any(isinstance(want, int) and isinstance(have, int) and not 0 <= want <= have
           for want, have in zip(shape, capacity, strict=True)):
        raise PyptoGap(op, 'DMA window valid extent exceeds its physical alias capacity', owner='ours')
    full = emitter._strip_tile(op, name + '.origin', {**tile, 'shape': capacity})
    # Validshape belongs to the alias being loaded, not another Tile at its
    # parent's address. Its declaration keeps the original NZ row pitch.
    pre, post = emitter._vs_wrap({**tile, 'base_py': full['py'], 'base_shape': capacity})
    return {**full, 'shape': shape}, pre, post
