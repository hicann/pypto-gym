# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the original typed ABI before the Pro printer expands list members."""
from .common import ContractError
from .public_io import canonical_params, decode_scalar


def validate_signature(module, data):
    import torch
    from ascriptor.ir.types import DimValue, MemType, Product, Ragged, ScalarType
    from ascriptor.backends.pypto_pro.emit import TORCH_DT

    params = list(next(fn for fn in module.functions if fn.kind == "func").params)
    args = data["args"]
    if len(args) > len(params):
        raise ContractError("case has more arguments than the kernel signature")
    for category in ("input", "output"):
        positions = list(data[f"{category}_indices"].values())
        if len(set(positions)) != len(positions):
            raise ContractError(f"duplicate {category} argument indices")
        for position in positions:
            if position >= len(params) or not isinstance(params[position].type, MemType):
                raise ContractError("public IO must map to a memory parameter in the original signature")
    bindings = {name: decode_scalar(value) for name, value in data.get("bindings", {}).items()}
    scalar_names = {p.name for p in params if isinstance(p.type, ScalarType)}
    if set(bindings) - scalar_names:
        raise ContractError("scalar bindings contain names outside the kernel signature")

    def bind(name, value):
        value = decode_scalar(value)
        if name in bindings and canonical_params({name: bindings[name]}) != canonical_params({name: value}):
            raise ContractError(f"scalar/shape binding disagrees with case arguments: {name}")
        bindings[name] = value

    dimensions = []
    for position, param in enumerate(params):
        kind = param.type
        if isinstance(kind, ScalarType):
            if position < len(args):
                bind(param.name, args[position])
            continue
        if position >= len(args):
            raise ContractError(f"missing tensor argument: {param.name}")
        value = args[position]
        is_list = kind.space == "gmlist"
        if is_list:
            if not isinstance(value, (list, tuple)) or not value:
                raise ContractError(f"kernel parameter {param.name} requires a non-empty TensorList")
            members = value
        else:
            members = (value,)
        for member in members:
            if not isinstance(member, torch.Tensor):
                raise ContractError(f"kernel parameter {param.name} requires tensors")
            if str(member.dtype).removeprefix("torch.") != TORCH_DT.get(kind.dtype.name):
                raise ContractError(f"dtype differs from kernel signature: {param.name}")
            if member.ndim != len(kind.dims):
                raise ContractError(f"rank differs from kernel signature: {param.name}")
            for declared, actual in zip(kind.dims, member.shape):
                if isinstance(declared, DimValue):
                    bind(declared.name, int(actual))
                dimensions.append((param.name, declared, int(actual)))
    for name, declared, actual in dimensions:
        if isinstance(declared, Ragged):
            continue
        if isinstance(declared, int):
            expected = declared
        elif isinstance(declared, DimValue):
            expected = bindings[declared.name]
        elif isinstance(declared, Product):
            expected = 1
            for factor in declared.factors:
                if isinstance(factor, int):
                    expected *= factor
                elif factor.name in bindings:
                    expected *= bindings[factor.name]
                else:
                    raise ContractError(f"provide a scalar binding for shape symbol {factor.name}")
        else:
            raise ContractError(f"unsupported kernel shape dimension: {declared}")
        if expected != actual:
            raise ContractError(f"shape differs from kernel signature: {name}")
    if scalar_names - bindings.keys():
        raise ContractError(f"missing scalar bindings: {sorted(scalar_names - bindings.keys())}")
    return bindings
