# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The ascriptor IR: types, values, the op registry, modules, printer, parser, verifier.

Specification: ``docs/rfc/0001-ir.md``. Nothing in this package may import from
``ascriptor.frontend``, ``ascriptor.passes``, ``ascriptor.backends`` or ``ascriptor.runtime``:
the IR is the bottom of the dependency graph (``ascriptor.devices`` is data and may be read).
"""

from . import types
from .builder import Builder, FunctionBuilder, Rewriter
from .core import LOWERED, SURFACE, Block, FuncRef, Function, Ident, Literal, Loc, Module, Op, Origin, Value
from .jsonform import from_json, to_json
from .parser import ParseError, parse_module
from .printer import print_module
from .registry import REGISTRY, OpSpec, Registry, load_builtin_ops
from .verify import Diagnostic, VerifyError, check, verify

load_builtin_ops()

__all__ = [
    "types", "Builder", "FunctionBuilder", "Rewriter", "LOWERED", "SURFACE", "Block", "FuncRef", "Function", "Ident",
    "Literal", "Loc", "Module", "Op", "Origin", "Value", "from_json", "to_json", "ParseError", "parse_module", "print_module",
    "REGISTRY", "OpSpec", "Registry", "load_builtin_ops", "Diagnostic", "VerifyError", "check", "verify",
]
