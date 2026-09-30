# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pl.system.dcci for core.clean_dcache (RFC-0015 "Data cache clean"), kept out of emit.py for its size bound."""
from ...ir import Value
from ...ir.types import MemType


def clean_dcache(side, op) -> None:
    """pl.system.dcci prints dcci(pointer + element offset, CacheLine, DcciDst): the address and policy cce's
    DataCacheCleanAndInvalid(window, ...) passes, with its defaults."""
    from .emit import PyptoGap  # emit.py imports this module

    dst = op.attrs.get("dst")
    line, target = (str(op.attrs.get(key, default)) for key, default in
                    (("entire_type", "ENTIRE_DATA_CACHE"), ("dcci_dst", "CACHELINE_OUT")))
    if not isinstance(dst, Value) or not isinstance(dst.type, MemType) or dst.type.space != "gm":
        raise PyptoGap(op, "clean_dcache needs a GM window in 'dst'")
    if "mode" in op.attrs or line not in ("SINGLE_CACHE_LINE", "ENTIRE_DATA_CACHE") or target not in (
            "CACHELINE_ALL", "CACHELINE_UB", "CACHELINE_OUT", "CACHELINE_ATOMIC"):
        raise PyptoGap(op, f"clean_dcache policy {line}/{target} (mode {op.attrs.get('mode')!r}) has no pl.system.dcci spelling")
    base, offset = side.scalar_address(op, dst, 0)
    side.emit(f"pl.system.dcci({base}, {offset}, cache_line=pl.CacheLine.{line}, dst=pl.DcciDst.{target})")
