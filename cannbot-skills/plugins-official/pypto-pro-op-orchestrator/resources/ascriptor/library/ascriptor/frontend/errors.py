# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frontend diagnostics (RFC-0002 §5): ``path:line:col: error[E0xxx]: message`` with the source line."""

from __future__ import annotations

import ast
import linecache


class CompileError(Exception):
    def __init__(self, code: str, message: str, file: str | None = None, node: ast.AST | None = None,
                 note: str | None = None) -> None:
        self.code = code
        self.message = message
        self.file = file
        self.line = getattr(node, "lineno", None)
        self.col = getattr(node, "col_offset", None)
        self.note = note
        super().__init__(self.format())

    def format(self) -> str:
        where = self.file or "<unknown>"
        if self.line is not None:
            where += f":{self.line}"
            if self.col is not None:
                where += f":{self.col + 1}"
        text = f"{where}: error[{self.code}]: {self.message}"
        if self.file and self.line is not None:
            src = linecache.getline(self.file, self.line).rstrip("\n")
            if src:
                text += f"\n    {src}\n    {' ' * (self.col or 0)}^"
        if self.note:
            text += f"\nnote: {self.note}"
        return text


# Codes (RFC-0002 §5): E00xx names and scope, E01xx binding and dataflow, E02xx unsupported syntax,
# E03xx types and shapes, E04xx device / capability, E05xx static evaluation failures.
E_UNKNOWN_NAME = "E0001"
E_NOT_CALLABLE = "E0002"
E_RESERVED_NAME = "E0003"
E_BAD_SIGNATURE = "E0010"
E_ACLNN_NAME_COLLISION = "E0011"
E_REBIND_CELL = "E0102"
E_CONDITIONAL_REBIND = "E0110"
E_OUT_OF_REGION = "E0111"
E_DYNAMIC_BREAK_IN_UNROLL = "E0120"
E_RETURN_NOT_LAST = "E0130"
E_UNSUPPORTED = "E0201"
E_BAD_OPERAND = "E0301"
E_BAD_SHAPE = "E0302"
E_BAD_COPY = "E0303"
E_STATIC_EVAL = "E0501"
