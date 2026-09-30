# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent reference for NsaCompress. Torch only; never imports ascriptor.

Semantics recovered from three upstream sources at ops-transformer ef85c0b5,
attention/nsa_compress, and cross-checked against each other:

  README.md / docs/aclnnNsaCompress.md   the formula, dtypes and the constraint list
  op_host/nsa_compress_infershape.cpp    the output-row count, which depends on actSeqLen VALUES
  op_kernel/nsa_compress_kernel.h        the arithmetic order, fp32 accumulation and CAST_ROUND

    out[i, n, :] = sum_{j=0}^{L-1} input[win_start[i] + j, n, :] * weight[j, n]

accumulated in fp32 and rounded once back to the input dtype.  Only whole windows
are emitted: batch b with sequence length S contributes floor((S - L) / d) + 1 rows
when S >= L, and none otherwise.

`win_start` is the host-derived table this port feeds the device instead of the
int64 prefix-sum tensor.  Deriving it is the same host work the upstream tiling
already does (nsa_compress_tiling_general.cpp computes kvStartTokenIdx and
PerCoreStartOutputOffset from actSeqLen the same way); it is recorded as permitted
host work in the task's authoring contract.
"""

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Significand bits including the implicit leading one, the exponent of the
# smallest subnormal step, and the largest finite value, per storage dtype.
SIGNIFICAND_BITS = {torch.float16: 11, torch.bfloat16: 8}
MIN_ULP_EXPONENT = {torch.float16: -24, torch.bfloat16: -133}
LARGEST_FINITE = {torch.float16: 65504.0, torch.bfloat16: 3.3895313892515355e38}


def round_half_away_from_zero(x32, dtype):
    """fp32 -> dtype with ties resolved away from zero.

    The upstream kernel's single writeback is
    ``Cast(..., AscendC::RoundMode::CAST_ROUND, ...)``, and CAST_ROUND breaks ties
    away from zero.  ``torch.Tensor.to`` breaks them to even (that is CAST_RINT),
    so the two disagree on every exactly-halfway value.  With bf16's eight
    significand bits those are common enough to be observable, and hiding the
    difference under a tolerance would be recording the wrong arithmetic.

    Subnormal results are rounded on the destination's fixed subnormal step, so
    the near-zero tail is exact too -- an fp16 output of this operator reaches it
    often enough to matter.  Overflow is not part of the declared domain and
    raises rather than saturating in silence.
    """
    p = SIGNIFICAND_BITS[dtype]
    largest = LARGEST_FINITE[dtype]

    magnitude = x32.abs()
    if bool((magnitude > largest).any()):
        raise ValueError(f"value overflows {dtype}: this port does not claim saturating casts")

    # frexp gives magnitude = fraction * 2**exponent with fraction in [0.5, 1),
    # so the step between neighbouring destination values is 2**(exponent - p) --
    # clamped, below the smallest normal, to the fixed subnormal step.
    _, exponent = torch.frexp(magnitude)
    ulp_exponent = torch.clamp(exponent - p, min=MIN_ULP_EXPONENT[dtype])

    # Scaling by a power of two is exact, and the scaled magnitude is
    # non-negative, so floor(v + 0.5) is round-half-up on the magnitude -- which
    # is round-half-away-from-zero once the sign goes back on.
    steps = torch.floor(torch.ldexp(magnitude, -ulp_exponent) + 0.5)
    result = torch.ldexp(steps, ulp_exponent)
    return torch.copysign(result, x32).to(dtype)


def domain(parameters):
    """Raise for any case the upstream operator does not accept.

    These are the constraints in README.md 约束说明, not a re-derivation: a case
    that violates one of them has no upstream answer to compare against.
    """
    N, D = parameters["N"], parameters["D"]
    L, d = parameters["L"], parameters["stride"]
    seq_lens = parameters["seq_lens"]

    if parameters["dtype"] not in DTYPES:
        raise ValueError(f"dtype {parameters['dtype']!r} is not one of {sorted(DTYPES)}")
    if not 1 <= N <= 128:
        raise ValueError(f"N={N} must satisfy 1 <= N <= 128 (weight.shape[1] == input.shape[1])")
    if D % 16 or not 0 < D <= 256:
        raise ValueError(f"D={D} must be a positive multiple of 16 and at most 256")
    if L % 16 or not 0 < L <= 128:
        raise ValueError(f"compressBlockSize={L} must be a positive multiple of 16 and at most 128")
    if d % 16 or d <= 0:
        raise ValueError(f"compressStride={d} must be a positive multiple of 16")
    if L < d:
        raise ValueError(f"compressBlockSize={L} must be >= compressStride={d}")
    if not seq_lens:
        raise ValueError("actSeqLen must not be empty")
    if any(s <= 0 for s in seq_lens):
        raise ValueError(f"every per-batch sequence length must be positive: {seq_lens}")


def window_starts(seq_lens, L, d):
    """The host table: absolute row index in T of each output window's first row.

    This is the infershape count expressed one window at a time.  Its length is
    exactly the `compressKvNum` that nsa_compress_infershape.cpp computes.
    """
    starts = []
    base = 0
    for seq_len in seq_lens:
        if seq_len >= L:
            for i in range((seq_len - L) // d + 1):
                starts.append(base + i * d)
        base += seq_len
    return starts


def make_inputs(case):
    parameters = case["parameters"]
    domain(parameters)

    N, D = parameters["N"], parameters["D"]
    L, d = parameters["L"], parameters["stride"]
    seq_lens = parameters["seq_lens"]
    dtype = DTYPES[parameters["dtype"]]

    T = sum(seq_lens)
    starts = window_starts(seq_lens, L, d)
    if not starts:
        raise ValueError(f"no whole window fits: seq_lens={seq_lens} all shorter than L={L}")

    generator = torch.Generator().manual_seed(case["seed"])
    # Values are kept small so that the fp32 accumulation of L terms stays well
    # inside the fp16 range; the comparison budget is about the single final
    # rounding, not about overflow.
    tensor = torch.randn(T, N, D, generator=generator).to(dtype)
    weight = (torch.randn(L, N, generator=generator) / L).to(dtype)

    return {
        "input": tensor,
        "weight": weight,
        "win_start": torch.tensor(starts, dtype=torch.int32),
        # Carried for the reference and for the upstream oracle; the device
        # signature does not take it.
        "act_seq_len": torch.tensor(torch.cumsum(torch.tensor(seq_lens), 0).tolist(), dtype=torch.int64),
        "seq_lens": seq_lens,
        "L": L,
        "stride": d,
        "N": N,
        "D": D,
        "T": T,
    }


def reduction_order(L):
    """The order upstream ReduceBlock folds L terms in, as an expression tree.

    This is observable, so it is semantics rather than an implementation choice:
    fp32 addition is not associative, and four of 9216 elements in the fp16
    48x100x1x32 case land close enough to an fp16 rounding boundary that the
    order decides which way they go. Measured against the vendor operator on A2 --
    all four match this tree and none match a serial left-to-right sum.

    nsa_compress_kernel.h ReduceBlock: align = ceilPow2(L)/2, fold the
    non-power-of-two remainder onto the front, then halve the stride to 1.
    """
    if L < 1:
        raise ValueError(f"L={L} must be positive")
    terms = [("leaf", j) for j in range(L)]
    align = 1
    while align < L:
        align <<= 1
    align //= 2
    for i in range(L - align):
        terms[i] = ("add", terms[i], terms[align + i])
    while align > 1:
        align >>= 1
        for i in range(align):
            terms[i] = ("add", terms[i], terms[align + i])
    return terms[0]


def _evaluate(node, products):
    if node[0] == "leaf":
        return products[node[1]]
    return _evaluate(node[1], products) + _evaluate(node[2], products)


def reference(inputs):
    tensor = inputs["input"]
    weight = inputs["weight"]
    starts = inputs["win_start"].tolist()
    L = inputs["L"]

    # fp32 accumulation, matching the kernel's castKvCacheLocal/subResultLocal path.
    body = tensor.float()
    coefficients = weight.float()
    tree = reduction_order(L)

    rows = []
    for start in starts:
        window = body[start:start + L]                      # [L, N, D]
        scaled = window * coefficients.unsqueeze(-1)        # weight[j, n] broadcast over D
        rows.append(_evaluate(tree, [scaled[j] for j in range(L)]))
    out32 = torch.stack(rows, dim=0)                        # [Tc, N, D]

    # One rounding back to the storage dtype, as the kernel's single
    # Cast(..., CAST_ROUND) does -- ties away from zero, not to even.
    return {"output": round_half_away_from_zero(out32, tensor.dtype)}
