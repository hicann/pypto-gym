# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit local synchronization printers shared by the PTO kernel frame."""
from typing import Any

from . import types as pt


class ExplicitLocalSync:
    def op_sync_set_flag(self, op: Any) -> None:
        self.emit(f"set_flag({pt.PIPE[str(op.attrs['src'])]}, {pt.PIPE[str(op.attrs['dst'])]}, (event_t){self.val(op.attrs['event_id'])});", op)

    def op_sync_wait_flag(self, op: Any) -> None:
        self.emit(f"wait_flag({pt.PIPE[str(op.attrs['src'])]}, {pt.PIPE[str(op.attrs['dst'])]}, (event_t){self.val(op.attrs['event_id'])});", op)
