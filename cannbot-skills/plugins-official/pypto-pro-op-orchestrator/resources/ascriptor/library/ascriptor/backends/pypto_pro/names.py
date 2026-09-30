# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable local names, independent of unrelated generated Python functions."""

import re

from ...ir import Value
from ...ir.types import ScalarType


def initializer_names(function):
    """Give shared anonymous initializer expressions a meaningful local spelling."""
    result = {}
    generic = r"(?:add|sub|mul|div|mod|min|max|neg|select|cast|and|or|xor|not|ceil_div|align)(?:[._]\d+)*"
    for op in function.walk():
        if op.opcode != "scalar.cell":
            continue
        value, cell = op.attrs.get("init"), op.results[0]
        if (isinstance(value, Value) and isinstance(value.type, ScalarType)
                and value.type.dtype.is_integer and value.type.dtype == cell.type.dtype
                and re.fullmatch(generic, value.name) and not re.fullmatch(r"v(?:[._]\d+)*", cell.name)):
            result.setdefault(value.name, f"initial_{cell.name}")
    return result


class LocalNames:
    def __init__(self, reserved=()):
        self.reserved = set(reserved)
        self.reset()

    def reset(self, parameters=()):
        self.used = self.reserved | set(parameters)

    def unique(self, name):
        base, index = name, 1
        while name in self.used:
            name = f"{base}_{index}"
            index += 1
        self.used.add(name)
        return name
