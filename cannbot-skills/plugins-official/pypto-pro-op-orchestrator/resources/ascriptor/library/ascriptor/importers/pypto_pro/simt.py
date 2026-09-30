# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT launches and thread bodies: element access, context queries, barriers and fences.

Admitted launches are one-dimensional and reached by every AIV. Pro context queries are UINT32;
the target queries are INT32 with the thread bounds of the callee's `max_threads`.
"""
from math import prod

from ...devices import SIMT_UB_CAP_KB
from ...ir import Block, FuncRef, Function, Value
from ...ir.types import MemType, ScalarType, dtype
from .lower import Context, Memory
from .memory import LAYOUTS
from .simt_math import TARGETS as MATH
from .simt_math import convert as math_call

QUERIES = {'simt.linear_thread_idx': 'simt.thread_id', 'simt.thread_idx': 'simt.thread_id',
           'simt.block_dim': 'simt.thread_num', 'simt.block_idx': 'simt.block_idx', 'simt.grid_dim': 'simt.block_num'}
STATEMENTS = {'simt.syncthreads': 'simt.barrier', 'simt.threadfence': 'simt.threadfence',
              'simt.threadfence_block': 'simt.threadfence_block'}
TARGETS = {'simt.launch': ('simt.launch',), 'block.getval': ('simt.load',), 'block.setval': ('simt.store', 'scalar.cast'),
           **{name: (target, 'scalar.cast') for name, target in QUERIES.items()},
           **{name: (target,) for name, target in STATEMENTS.items()}}
U32 = ScalarType(dtype('u32'))
ELEMENTS = {'f32', 'i32'}
UB_LIMIT = SIMT_UB_CAP_KB * 1024  # The target keeps UB above this for SIMT launches (addr_alloc and the model).


def launch(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {'callee', 'max_threads'})
    o.need(o.side == 'vec' and ctx.depth == 0, node,
           'Every AIV must reach a SIMT launch; launch at the top level of the vector section')
    o.need(len(args) >= 3 and all(type(n) is int for n in args[:3]), node, 'Expected static SIMT launch dimensions')
    o.need(all(end <= UB_LIMIT for _, end in o.allocations.get('ub', ())), node, f'SIMT launches reserve UB above {SIMT_UB_CAP_KB} KB')
    o.simt_launched = True
    threads = args[0]
    o.need(args[1:3] == [1, 1], node, 'SIMT launches admit one thread dimension')
    function, bound = convert(ctx, node, attrs.get('callee'))
    o.need(attrs.get('max_threads') == bound and 1 <= threads <= bound, node, 'SIMT launch threads contradict the callee bound')
    o.need(len(args) - 3 == len(function.params), node, 'SIMT launch arguments disagree with the callee parameters')
    operands = []
    for param, arg in zip(function.params, args[3:], strict=True):
        if isinstance(param.type, MemType):
            o.need(isinstance(arg, Memory) and arg.value.type == param.type and (arg.pitch is None or arg.root is not None), node,
                   'SIMT memory arguments must be whole storage of the parameter type')
            operands.append(arg.value if arg.root is None else view_argument(ctx, node, arg))
            continue
        value = ctx.snapshot(arg, node, param.type)
        o.need(isinstance(value, Value) and value.type == param.type, node, 'SIMT scalar arguments must have the parameter type')
        operands.append(value)
    ctx.emit('simt.launch', node, (FuncRef(function.name), *operands), attrs={'threads': threads})


def view_argument(ctx, node, view):
    """A GM view passed to a SIMT function is read row-major from its origin at the declared shape's pitch, ignoring its
    strides (view_simt_probe on A5): a reshaped window of the flattened root."""
    o, root = ctx.o, view.root
    o.need(root.value.type.space == 'gm' and view.value.type.dtype == root.value.type.dtype, node,
           'SIMT memory arguments cannot be tile aliases or dtype views')
    o.need(type(view.origin) is int, node, 'SIMT view arguments need a static origin')
    (rows, cols), numel, dt = view.shape, prod(root.shape), root.value.type.dtype
    flat = ctx.emit('mem.reshape', node, (root.value,), typ=MemType('gm', dt, (1, numel)), attrs={'shape': [1, numel]})
    window = ctx.emit('mem.slice', node, (flat,), typ=MemType('gm', dt, (1, rows * cols)),
                      attrs={'offsets': [0, view.origin], 'extents': [1, rows * cols]})
    return ctx.emit('mem.reshape', node, (window,), typ=MemType('gm', dt, (rows, cols)), attrs={'shape': [rows, cols]})


def storage_parameter(o, var):
    """A static FP32/INT32 GM tensor or whole canonical Vec tile, as Pro passes its data pointer."""
    typ = o.node(var['fields']['type'])
    o.need(typ['kind'] in {'TensorType', 'TileType'}, var, 'SIMT parameters are scalars, GM tensors or Vec tiles')
    f = typ['fields']
    shape = o.shape(f['shape'], var)
    dt = o.dt(f['dtype'], var)
    o.need(dt.name in ELEMENTS or (dt.name == 'u32' and typ['kind'] == 'TensorType'), var,
           'SIMT element access admits FP32 and INT32 storage and UINT32 GM tensors')
    if typ['kind'] == 'TensorType':
        view = o.node(f['tensor_view'])['fields'] if f['tensor_view'] else None
        # A view argument gives the type its strides; the body still reads the declared pitch (view_simt_probe on A5).
        o.need(f['memref'] is None and (view is None or view['layout']['name'] == 'ND' and not view['valid_shape']), var,
               'Only ordinary ND GM tensor parameters are admitted')
        return MemType('gm', dt, shape)
    memory = o.node(f['memref'])['fields']
    hardware = o.node(f['hardware_info'])['fields']
    view = o.node(f['tile_view'])['fields']
    canonical = (hardware['blayout']['name'], hardware['slayout']['name'], hardware['fractal'])
    o.need(memory['memory_space']['name'] == 'Vec' and canonical == LAYOUTS['Vec'] and hardware['pad']['name'] == 'null'
           and hardware['compact']['name'] == 'null' and not view['stride'] and view['start_offset'] is None, var,
           'SIMT tiles must be canonical Vec storage')
    return MemType('ub', dt, shape)


def convert(ctx, node, name):
    """Convert a launched SimtVF once; its thread bounds come from `max_threads`."""
    o = ctx.o
    if name in o.simt_functions:
        return o.simt_functions[name]
    source = o.simt_sources.get(name)
    o.need(source is not None and source['fields']['func_type']['name'] == 'SimtVF', node,
           'A SIMT launch needs a launchable SIMT function')
    f = source['fields']
    bound = f['attrs'].get('max_threads')
    o.need(set(f['attrs']) == {'max_threads'} and type(bound) is int and 1 <= bound <= 2048 and not f['return_types'],
           source, 'Expected the pinned SIMT function form')
    o.need(all(fn.name != name for fn in o.functions), source, 'SIMT function name collides with an imported function')
    o.unsigned_scope, outer = True, o.unsigned_scope
    child = Context(o, simt=bound)
    params = []
    for ref in f['params']:
        var = o.node(ref)
        pname = var['fields']['name']
        if o.node(var['fields']['type'])['kind'] == 'ScalarType':
            value = Value(pname, o.scalar_type(var['fields']['type']))
            child.env[pname] = value
        else:
            value = Value(pname, storage_parameter(o, var))
            child.env[pname] = Memory(value, value.type.dims, value.type.dims)
        params.append(value)
    child.prepare_cells(f['body'])
    child.block(f['body'])
    child.emit('cf.return', source)
    o.unsigned_scope = outer
    function = Function('simt', name, tuple(params), {}, Block(tuple(child.ops)))
    o.functions.append(function)
    o.simt_functions[name] = function, bound
    return function, bound


def declaration(ctx, ref):
    """Pro binds all three axes of a context query for a `.x` read; the unused tuple is never printed."""
    o = ctx.o
    node = o.node(ref)
    calls = [o.node(e) for e in node['fields']['elements']] if node['kind'] == 'MakeTuple' else []
    names = {c['fields'].get('name') for c in calls}
    if not (len(calls) == 3 and len(names) == 1 and names <= set(QUERIES) - {'simt.linear_thread_idx'}
            and all(c['kind'] == 'Call' and not c['fields']['args'] for c in calls)
            and [c['fields']['kwargs'] for c in calls] == [{'axis': axis} for axis in range(3)]):
        return False
    for call in calls:
        o.record(call, len(o.emitted))
    return True


def thread_call(ctx, node, args):
    from .selection import interval

    o, name = ctx.o, node['fields']['name']
    if name in QUERIES:
        attrs = ctx.attrs(node, {'axis'})
        o.need(not args and attrs == ({} if name == 'simt.linear_thread_idx' else {'axis': 0}), node,
               'Only x-axis SIMT context queries are admitted; admitted launches are one-dimensional')
        o.need(not o.mixed or name not in {'simt.block_idx', 'simt.grid_dim'}, node,
               'SIMT core queries in mixed launches need a measured AIC/AIV mapping')
        o.need(o.scalar_type(node['fields']['type']) == U32, node, 'Expected a UINT32 SIMT context query')
        opcode = QUERIES[name]
        value = ctx.emit('scalar.cast', node, (ctx.emit(opcode, node, typ=ScalarType(dtype('i32'))),), typ=U32)
        ctx.bounds[value.name] = {'simt.thread_id': (0, ctx.simt - 1), 'simt.thread_num': (1, ctx.simt),
                                  'simt.block_idx': (0, 2**31 - 1), 'simt.block_num': (1, 2**31 - 1)}[opcode]
        ctx.nonnegative.add(value.name)
        return value
    if name in STATEMENTS:
        ctx.attrs(node)
        o.need(not args, node, 'SIMT barriers and fences take no arguments')
        o.need(name != 'simt.syncthreads' or ctx.depth == 0, node,
               'Every thread must reach syncthreads; place it at the top level of the SIMT function')
        ctx.emit(STATEMENTS[name], node)
        return None
    if name in {'block.getval', 'block.setval'}:
        ctx.attrs(node)
        load = name == 'block.getval'
        o.need(len(args) == (2 if load else 3) and isinstance(args[0], Memory), node, 'Unadmitted SIMT element access')
        storage = args[0].value
        element = ScalarType(storage.type.dtype)
        index = ctx.snapshot(args[1], node)
        bounds = interval(ctx, index)
        o.need(bounds is not None and bounds[0] >= 0 and bounds[1] < prod(storage.type.dims), node,
               'SIMT element index is not proven inside its storage')
        if load:
            o.need(o.scalar_type(node['fields']['type']) == element, node, 'SIMT load type contradicts its storage')
            return ctx.emit('simt.load', node, (storage, index), typ=element)
        value = args[2]
        if type(value) is int:
            bits, kind = element.dtype.bits, element.dtype.kind
            low, high = {'int': (-(1 << (bits - 1)), 1 << (bits - 1)), 'uint': (0, 1 << bits)}.get(kind, (-(1 << 24), (1 << 24) + 1))
            o.need(low <= value < high, node, 'SIMT stored literal changes value in its storage type')
        value = ctx.snapshot(value, node, element)
        if isinstance(value, Value) and value.type != element:
            kinds = {value.type.dtype.kind, element.dtype.kind}
            o.need(kinds <= {'int', 'uint', 'bool'}, node, 'SIMT stores convert only between integer types implicitly')
            value = ctx.emit('scalar.cast', node, (value,), typ=element)  # C: modulo the width; BOOL is 0 or 1.
        o.need(isinstance(value, Value) and value.type == element, node, 'SIMT stored value does not match its storage')
        ctx.emit('simt.store', node, (storage, index, value))
        return None
    if name in MATH:
        return math_call(ctx, node, args)
    from .simt_exact import CALLS as EXACT
    from .simt_exact import convert as exact
    if name in EXACT:
        return exact(ctx, node, args)
    if name == 'block.tile_valid_shape':
        o.fail(node, 'Target SIMT launches do not pass tile valid shapes')
    if name in o.simt_sources:
        o.fail(node, 'SIMT helper calls need inlining')
    o.fail(node, f'Unadmitted operation {name} inside a SIMT function')
    return None
