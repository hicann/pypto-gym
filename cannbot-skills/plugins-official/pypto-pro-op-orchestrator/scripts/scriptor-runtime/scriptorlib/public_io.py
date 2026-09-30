# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public IO validation, also embedded verbatim in standalone generated launchers.

Metadata checks never perform tensor arithmetic. Value checks are only for test
input preparation and CPU Golden preflight, outside the public wrapper.
"""


def validate_public_io(value, *, name, shape, dtype, is_list=False, device_type=None):
    import torch

    if is_list:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"TensorList {name} must be a non-empty list of tensors")
        if len(value) != len(shape):
            raise ValueError(f"TensorList {name} arity differs from SPEC")
        members, shapes = value, shape
    else:
        members, shapes = (value,), (shape,)
    first_device = None
    for index, (member, expected_shape) in enumerate(zip(members, shapes)):
        label = f"{name}[{index}]" if is_list else name
        if not isinstance(member, torch.Tensor):
            raise ValueError(f"public IO must be a tensor: {label}")
        if list(member.shape) != expected_shape:
            raise ValueError(f"case shape differs from SPEC: {label}")
        if str(member.dtype).removeprefix("torch.") != dtype:
            raise ValueError(f"case dtype differs from SPEC: {label}")
        if device_type is not None and member.device.type != device_type:
            raise ValueError(f"case device differs from {device_type}: {label}")
        if is_list:
            if member.layout != torch.strided or not member.is_contiguous():
                raise ValueError(f"TensorList member must be contiguous: {label}")
            if first_device is not None and member.device != first_device:
                raise ValueError(f"TensorList members must share a device: {name}")
            first_device = member.device
    return members


def normalize_outputs(result, outputs):
    """Distinguish one TensorList output from several public tensor outputs."""
    names = [item["name"] for item in outputs]
    if isinstance(result, dict):
        values = result
    elif len(outputs) == 1 and outputs[0].get("is_list", False):
        values = {names[0]: result}
    else:
        ordered = result if isinstance(result, (tuple, list)) else (result,)
        values = dict(zip(names, ordered)) if len(ordered) == len(names) else {}
    if set(values) != set(names):
        raise ValueError("public output names/count differ from SPEC")
    return values


def decode_scalar(value):
    # These strings are the JSON contract's reserved IEEE special-scalar spellings.
    if isinstance(value, str) and value.lower() in ("inf", "+inf", "-inf", "nan"):
        return float(value)
    return value


def decode_params(params):
    return {name: decode_scalar(value) for name, value in params.items()}


def json_safe(value):
    """Keep non-finite numbers out of JSON and compare NaN parameters reflexively."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {name: json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_params(params):
    return json_safe(decode_params(params))


def validate_input_values(members, *, name, is_list=False, value_range=None, special_values=()):
    """Test-only content checks; excluded from generated standalone launchers."""
    import torch

    required, observed = set(special_values), set()
    for index, member in enumerate(members):
        label = f"{name}[{index}]" if is_list else name
        if value_range is None and not special_values:
            continue
        finite = torch.isfinite(member)
        if required or not bool(finite.all()):
            for tag, predicate in (("-inf", torch.isneginf), ("+inf", torch.isposinf), ("nan", torch.isnan)):
                if bool(predicate(member).any()):
                    observed.add(tag)
        if value_range is not None:
            low, high = value_range
            if bool((finite & ((member < low) | (member > high))).any()):
                raise ValueError(f"generated input violates SPEC value_range: {label}")
    if observed != required:
        raise ValueError(f"generated input special values differ from SPEC: {name}; "
                         f"lacks {', '.join(sorted(required - observed))}; "
                         f"unexpected {sorted(observed - required)}")
