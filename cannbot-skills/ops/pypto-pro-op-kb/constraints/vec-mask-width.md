# Converting a mask between element widths

A `MaskReg` can be converted to a different element width in **one instruction**. Two separate
efforts independently concluded it could not be, and between them that cost several development
rounds of avoidable work. This page exists because the capability is real but undiscoverable
from the place you would look for it.

## Why it is missed

Both operators audited the functions with `mask` in the name — `create_mask`, `update_mask`,
`unsqueeze`, `get_mask_spr`, `mask_gen_with_reg_tensor` — and correctly concluded that none of them
retypes an existing, data-dependent mask. That audit was right. The converter is filed under
**advanced computation**, not mask operations: `vf.interleave` and `vf.de_interleave` are *unified*
ops whose behaviour changes when their sources are `MaskReg`s.

`mask_reg.md`'s "Mask 设置方式" table lists five construction routes and conversion is not among
them.

## Why it works

From `docs/pypto_pro/api/SIMD-API/vf_computation/mask_reg.md`: a
`MaskReg` is a fixed **256-bit** object, and *width* is not a property of the register. It is the
**stride at which the consuming instruction reads it** — b8 reads bit *i*, b16 bit 2*i*, b32 bit
4*i*.

A width conversion is therefore nothing but **bit re-spacing**, and `pintlv` / `pdintlv` are exactly
the predicate re-spacing instructions. From `_vf_api.py:1312-1313` and `:687-688`:

> `dtype`: When src operands are MaskReg, specifies the interleave bit-width (selects
> `pintlv_b8`/`b16`/`b32`). Inferred from src0 if omitted.

A corollary that matters in practice: **the `dtype=` you pass to `create_mask` when declaring a
destination is bookkeeping, not semantics.** The shipped sample below declares its b16 result as
`DT_UINT8` and is still correct.

## The conversions

**`dtype=` names the *finer* of the two widths.** Narrowing discards `dst1`; widening uses both.

| Direction | Call | Ops |
|---|---|---|
| b32 (2×64 lanes) → b16 (128) | `m16, _ = vf.de_interleave(m32_lo, m32_hi, dtype=pl.DT_UINT16)` | 1 |
| b16 (2×128) → b8 (256) | `m8, _ = vf.de_interleave(m16_lo, m16_hi, dtype=pl.DT_UINT8)` | 1 |
| b16 (128) → b32 (2×64) | `m32_lo, m32_hi = vf.interleave(m16, m16, dtype=pl.DT_UINT16)` | 1 |
| b8 (256) → b16 (2×128) | `m16_lo, m16_hi = vf.interleave(m8, m8, dtype=pl.DT_UINT8)` | 1 |
| b32 → b8 | two `de_interleave(…, UINT16)` then one `de_interleave(…, UINT8)` | 3 |

`src0 == src1` is explicitly legal (`interleave.md:53-54`); `dst0 == dst1` is not.

## The shipped sample

`.devkit/pro_ops/fa/test_flex_attention.py:551-553` — two 64-lane b32 masks in, one 128-lane b16
mask out, one instruction:

```python
merge_bit        = vf.lt(index,        maskr_reg, preg_all)   # b32, lanes 0..63
merge_unroll_bit = vf.lt(index_unroll, maskr_reg, preg_all)   # b32, lanes 64..127
row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
```

Two details worth copying, both non-obvious: the destinations are pre-declared as **`DT_UINT8`**
(`:527-528`) rather than at the target width, and mask-level boolean combination works —
`vf.and_(m0, m1, preg)`, three args with a predicate third.

## Evidence classes — check before building

| Claim | Evidence |
|---|---|
| MaskReg is 256 raw bits; width is a read-stride | **documented** (`mask_reg.md:17,19`) |
| `interleave`/`de_interleave` take MaskRegs; `dtype` selects `pintlv/pdintlv_b8/16/32` | **documented** (`_vf_api.py:687,1312`; `de_interleave.md:31,111`) |
| `dst0` = lower part, `dst1` = higher part | **documented figure + numeric golden** (`test_vf_basic_ops.py:3865-3867` round-trips and recovers both 64-lane inputs) |
| **narrowing** b32 → b16 | **sample-backed, shipped** |
| **widening** b16 → b32 | **measured on device** — a retained lane-mask probe, 128/128 lanes correct with `dtype=pl.DT_UINT16`; `dtype=pl.DT_UINT32` is wrong on 32/128, confirming `dtype=` names the *finer* width |
| b32 ↔ b8 in three ops | **derived only** |

## It working does not mean it pays

One effort verified the widening on device, built a native fp16/bf16 path on it, got a **correct**
kernel — and measured it **slower on every target case it was meant to help**. Reverted.

A substantial gain had been projected for it and carried for several rounds. It was void, and it
was **that effort's own earlier fix that voided it**: a round before, it had found the narrow-dtype
penalty was mostly a UB bank conflict on the *work tile's* pitch rather than the fp32 round trip,
and fixing the pitch had already collected that value. What remained of the round trip costs less
than two `vf.interleave` calls plus doubled index-register traffic — the native step is 20 vector
ops per 128 elements against 26-plus-a-cast, and still measures 1.18–1.44× worse.

**The transferable rule: an estimate inherits the lifetime of the diagnosis it came from.**
Re-derive a carried-forward number before spending on it, especially after fixing something in the
same area.

## What the round-trip costs, for comparison

`vf.unsqueeze` → `vf.pack` → compare-back does close the loop, and it was correctly identified by
both operators — the error was concluding nothing cheaper existed. It runs ≈3 ops per 64 lanes and
fills only the lower half of the wider register, so covering 128 lanes takes 5–6 ops plus a
register-level merge. `de_interleave` does it in **one**.

`vf.mask_gen_with_reg_tensor` (`_vf_api.py:1885`) is a genuine, data-dependent register→mask
converter and is better than compare-against-zero for that route, but its source is restricted to
*uint16 or uint32*, so it cannot reach b8.

## A documented fallback, not the primary route

`load_align.md:82` gives `pl.LoadDist.US` (upsample, each bit repeated twice) and `pl.LoadDist.DS`
(downsample, every other bit discarded), complementary to `StoreDist.PACK`. That is a UB round-trip
— a store, a sync and a load — so it is slower. Its value is that it independently confirms ×2/÷2
mask conversion is a first-class concept in this API rather than an accidental property of
`pdintlv`.

## The residual gap is discoverability, not capability

Worth filing upstream, in decreasing value: add a **conversion row to `mask_reg.md`'s construction
table** (a one-line doc change that would have prevented all of this); a `vf.mask_astype(mask,
dtype=…)` alias over the `interleave`/`de_interleave` pair; a b8 form of
`mask_gen_with_reg_tensor`; and a `MaskWidth.B8` for `get_mask_spr`.
