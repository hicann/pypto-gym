# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a compiled kernel on the reference interpreter with host tensors (the ``sim`` launcher's core).

``run_kernel(kernel, *args)`` takes the same positional arguments the old ``OpExec`` took —
GM tensors in signature order, then explicit scalars — derives every shape symbol from the
real tensors (RFC-0002 §3.2), copies the inputs so the caller's tensors are untouched, runs the
module, and returns the outputs (fresh tensors) in ``return`` order.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ...ir import Function, Module, Value
from ...ir.types import DimValue, MemType, Product, Ragged
from .interp import Machine, torch_dtype


def bind_arguments(module: Module, args: tuple[Any, ...], *, seed_outputs: bool = False) -> dict[str, Any]:
    """Bind positional arguments (tensors, then scalars) to the kernel's parameters.

    Inputs are copied. Outputs are **poisoned** (0xFF bytes: NaN for floats, -1 / 255 for ints) so
    that every element the kernel forgets to write shows up in the result, as in the old simulator;
    ``seed_outputs=True`` copies the caller's tensor instead, for kernels that read their outputs.
    """
    kernel = entry_function(module)
    params = list(kernel.params)
    tensor_params = [p for p in params if isinstance(p.type, MemType)]
    scalar_params = [p for p in params if not isinstance(p.type, MemType)]
    outputs = {o.name for o in kernel.attrs.get("outputs", []) if isinstance(o, Value)}
    if len(args) < len(tensor_params):
        raise TypeError(f"{kernel.name} needs {len(tensor_params)} tensor argument(s), got {len(args)}")
    bound: dict[str, Any] = {}
    for p, a in zip(tensor_params, args[: len(tensor_params)], strict=True):
        if p.type.space == "gmlist":  # type: ignore[union-attr]
            if not isinstance(a, (list, tuple)) or not all(isinstance(m, torch.Tensor) for m in a):
                raise TypeError(f"argument {p.name} must be a list of torch tensors")
            bound[p.name] = [poison_like(m) if p.name in outputs and not seed_outputs else m.detach().clone().contiguous() for m in a]
            continue
        if not isinstance(a, torch.Tensor):
            raise TypeError(f"argument {p.name} must be a torch tensor")
        if p.name in outputs and not seed_outputs:
            bound[p.name] = poison_like(a)
        else:
            bound[p.name] = a.detach().clone().contiguous()
    explicit = args[len(tensor_params):]
    given: dict[str, Any] = {}
    for p, a in zip(scalar_params, explicit, strict=False):
        given[p.name] = a
    if len(explicit) > len(scalar_params):
        raise TypeError(f"{kernel.name}: too many scalar arguments")
    # shape symbols from the real tensors
    derived: dict[str, int] = {}
    for p in tensor_params:
        t = p.type
        if not isinstance(t, MemType):
            raise TypeError(f"{p.name}: expected a memory tensor parameter")
        members = bound[p.name] if t.space == "gmlist" else [bound[p.name]]
        for k, m in enumerate(members):
            what = f"{p.name}[{k}]" if t.space == "gmlist" else p.name
            shape = tuple(m.shape)
            if m.dtype != torch_dtype(t.dtype):
                raise TypeError(f"argument {what}: dtype {m.dtype} does not match the declared {t.dtype.name} (carried as {torch_dtype(t.dtype)})")
            if len(shape) != t.rank:
                raise TypeError(f"argument {what} has rank {len(shape)}, the kernel declares rank {t.rank}")
            for d, n in zip(t.dims, shape, strict=True):
                if isinstance(d, int):
                    if d != n:
                        raise TypeError(f"argument {what}: dimension {n} does not match the declared {d}")
                elif isinstance(d, DimValue):
                    if derived.setdefault(d.name, n) != n:
                        raise TypeError(f"shape symbol {d.name} is {derived[d.name]} in one argument and {n} in {what}")
                elif isinstance(d, Product):
                    syms = [f for f in d.factors if isinstance(f, DimValue)]
                    known = 1
                    for f in d.factors:
                        known *= f if isinstance(f, int) else derived.get(f.name, 1)
                    if len(syms) == 1 and syms[0].name not in derived and n % known == 0:
                        derived[syms[0].name] = n // known
                # a Ragged ('?') dim is per member: read at runtime through list.item_dim
    for p in scalar_params:
        if p.name in given:
            v = given[p.name]
            if p.name in derived and int(v) != derived[p.name]:
                raise TypeError(f"scalar {p.name}={v} disagrees with the tensor shapes ({derived[p.name]})")
            bound[p.name] = v
        elif p.name in derived:
            bound[p.name] = derived[p.name]
        else:
            raise TypeError(f"missing scalar argument {p.name}")
    for name in kernel.attrs.get("pro_dynamic_dimensions", []):
        if not 0 < derived[name] < 2**63:
            raise TypeError(f"Pro runtime dimension {name} must be a positive signed INDEX")
    if kernel.attrs.get('pro_workspace_layout'):
        from ...importers.pypto_pro.workspace import check_sizes
        check_sizes(kernel.attrs['pro_workspace_layout'], bound)
    return bound


def entry_function(module: Module) -> Function:
    """The kernel of a surface module, or a stand-in built from a lowered module's per-side funcs and ``meta``."""
    for f in module.functions:
        if f.kind == "kernel":
            return f
    funcs = [f for f in module.functions if f.kind == "func"]
    if not funcs:
        raise TypeError("the module has neither a kernel nor per-side funcs")
    meta = dict(module.attrs.get("meta", {}))
    return replace(funcs[0], kind="kernel", name=str(meta.get("kernel", funcs[0].name)), attrs=meta)


def poison_like(t: torch.Tensor) -> torch.Tensor:
    raw = torch.full((t.numel() * t.element_size(),), 255, dtype=torch.uint8)
    return raw.view(t.dtype).reshape(t.shape)


def run_module(module: Module, args: tuple[Any, ...], *, block_dim: int | None = None, timeout: float = 120.0,
               seed_outputs: bool = False, processes: bool | None = None) -> list[torch.Tensor]:
    bound = bind_arguments(module, args, seed_outputs=seed_outputs)
    machine = Machine(module, timeout=timeout)
    machine.run(bound, block_dim=block_dim, processes=processes)
    outputs = entry_function(module).attrs.get("outputs", [])
    return [bound[o.name] for o in outputs if isinstance(o, Value)]


def run_kernel(kernel: Any, *args: Any, block_dim: int | None = None, timeout: float = 120.0, seed_outputs: bool = False,
               processes: bool | None = None) -> Any:
    """Run a ``@kernel`` on the reference interpreter; returns one tensor or a list of them."""
    outs = run_module(kernel.ir(), args, block_dim=block_dim, timeout=timeout, seed_outputs=seed_outputs, processes=processes)
    return outs[0] if len(outs) == 1 else outs


__all__ = ["bind_arguments", "poison_like", "run_module", "run_kernel"]
