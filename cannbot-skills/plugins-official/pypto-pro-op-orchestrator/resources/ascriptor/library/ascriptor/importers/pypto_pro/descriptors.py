# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolate descriptor state and refuse unproved values at control-flow joins."""
from dataclasses import replace

from .lower import Memory


def clone_env(env):
    from .selection import Choice

    memories = {}

    def clone(value):
        if isinstance(value, Memory):
            return memories.setdefault(id(value), replace(value))
        if isinstance(value, Choice):
            return replace(value, items=tuple(clone(v) for v in value.items))
        if isinstance(value, tuple):
            return tuple(clone(v) for v in value)
        return value
    return {name: clone(value) for name, value in env.items()}


def updates(ctx, body):
    owners = set()
    for node in ctx.expr_nodes(body):
        if node['kind'] != 'Call' or node['fields']['name'] != 'block.set_validshape':
            continue
        ref = ctx.o.node(node['fields']['args'][0])
        value = ctx.env.get(ref['fields'].get('name')) if ref['kind'] == 'Var' else None
        ctx.o.need(isinstance(value, Memory) and value.value.type.space == 'ub' and value.pitch is None,
                   node, 'Descriptor mutation in control flow requires a known plain UB owner')
        owners.add(value.value.name)
    return owners


def invalidate(ctx, owners):
    from .selection import Choice

    def visit(value):
        if isinstance(value, Memory) and value.value.name in owners:
            value.valid = None
        elif isinstance(value, Choice):
            for item in value.items:
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)
    for value in ctx.env.values():
        visit(value)
