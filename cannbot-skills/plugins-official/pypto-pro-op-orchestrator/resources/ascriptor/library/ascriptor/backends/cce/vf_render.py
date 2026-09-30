# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep a terminal void return outside the compiler's vector region."""
from ...ir import Block


def render_vf(printer):
    head = printer.signature()
    ops = printer.fn.body.ops
    terminal = ops[-1] if ops and ops[-1].opcode == 'cf.return' and not ops[-1].operands else None
    printer.indent = 2
    printer.run_block(Block(ops[:-1] if terminal else ops))
    body = '\n'.join(printer.lines)
    tail = ''
    if terminal:
        start = len(printer.lines)
        printer.indent = 1
        printer.run_op(terminal)
        tail = '\n'.join(printer.lines[start:]) + '\n'
    return f'{head}\n{{\n    __VEC_SCOPE__\n    {{\n{body}\n    }}\n{tail}}}\n'
