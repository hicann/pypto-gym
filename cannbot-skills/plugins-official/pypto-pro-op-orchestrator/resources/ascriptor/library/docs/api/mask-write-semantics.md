# What `mask=` does to the lanes it turns off

Sixty-six vf operators take a predicate, and they do not agree on what an inactive lane means.
Most of them **write zero** there — the arithmetic, the casts, the register-to-register copies
and the mask-register booleans alike. A masked store does not: it **never writes that lane**, so
the memory keeps whatever it held. Nothing preserves a destination *register* lane; a register is
always written whole, and only memory can be left alone. Same keyword, opposite results, and
until now the answer was written down only in the reference interpreter.

Three lines, run side by side, with the destination holding `7.0` and a mask that is false
everywhere — so by any reading "nothing should happen":

```python
api.add(dst, a, a, mask=m)        # dst was 7.0 ... now 0.0    the lane was CLEARED
api.mask_mov(md, ms, mask=m)      # md  was all-1 ... now 0    the bit was CLEARED too
api.reg_to_ub(ub, a, mask=m)      # ub  was 7.0 ... still 7.0  the lane was SKIPPED
```

The middle line is the one that has already been got wrong once, in this repository: the
mask-register booleans look like they ought to merge into the destination, and they do not.
`_mask_blend` ANDs the result with the mask and writes the whole register.
The physical-predicate regressions
retain the M10-051 correction. [Registers](registers.md#execution-masks-and-selectors)
states the current inactive-zero rule, and
[precision guidance](../../../agent/en/references/precision.md) applies it.
Treating the destination bit as surviving produces the wrong value.

The nearest thing to a merging write anywhere on this part is the one the hardware refuses: a
cast can ask for `MaskMergeMode.MERGING`, and `backends/cce/arch/c310.py` records that every one
of the 55 rows in the Cast API's tables 6-9 reads `ZEROING`, with a compile probe agreeing on all
ten pairs it tried. The cce printer turns such a cast into a `CceGap` and tells you to convert
into a temporary and `select`. That is the shape of the answer everywhere in this table: if you
want an inactive destination lane to survive, you write it yourself.

Guessing wrong is not a crash; it is a wrong number, in the direction of whichever assumption
you brought. Guessing conservatively costs too: the sparse-attention kernels in this repository
spent two instructions — `select` to push masked lanes to `-inf`, then `exp` — where one would
do, because `expsub(v, v, zero, mask=live)` zeroes the inactive lanes itself and swallows
whatever the cube left in the padding rows. The rewrite was unsafe to make without knowing the
zeroing was guaranteed, and on a route that spends three quarters of its time issuing vector
instructions, that one line was a twelfth of them.

`tests/ir/test_mask_semantics.py` runs both halves of that contrast through the reference
interpreter, so the two roles a kernel can most easily get backwards are pinned by execution
rather than by reading this page back to itself.

`examples/api/mask_semantics/` is the natural home for a released demonstration of it, and does
not have one yet: that unit's `support` rows are release-acceptance records tied to a frozen
wheel and an audited campaign, so a new case belongs to the next such campaign rather than to a
local run.

## The roles

### `zero` — the inactive destination lane is written 0

`_store_masked` / `_store_num` / `_write_reg`, and `reg.zero_()` before the active lanes in the
gathering forms. For the mask-register booleans it is `_mask_blend`: `result & mask`, written
whole.

`abs` `abssub` `add` `adds` `and` `axpy` `cast` `copy` `cpadd` `div` `dup` `exp` `expsub` `gather_copy` `ln` `load` `log` `log10` `log2` `lrelu` `mask_and` `mask_mov` `mask_not` `mask_or` `mask_xor` `max` `maxs` `min` `mins` `mod` `mul` `muladddst` `muldstadd` `muls` `mulscast` `neg` `not` `or` `prelu` `relu` `shiftl` `shiftls` `shiftr` `shiftrs` `sqrt` `sub` `xor`  *(47)*

### `skip` — the inactive lane is not written; the destination memory keeps its bytes

the store handlers index the destination with `where[active]` and never touch the rest.

`scatter_copy` `store` `store_cont`  *(3)*

### `filter` — the mask selects source lanes; the destination is packed or reduced

`_masked_src`, `_group_reduce`, and the compressing handlers.

`cadd` `cgadd` `cgmax` `cgmin` `cmax` `cmin` `gathermask` `histograms` `squeeze`  *(9)*

### `select` — the mask is the selector, not a predicate: every lane is written

`op_vf_select` and `op_vf_mask_sel`: the mask becomes the `torch.where` selector, so both sources
contribute and no lane is off. `mask_sel` is the only member of the `mask_*` family that is not
`zero`; that is what makes the family easy to classify wrong as a block.

`mask_sel` `select`  *(2)*

### `predicate` — the result is a new mask; inactive lanes come out false

`_compare`: `bits = result & mask`.

`cmp` `cmps`  *(2)*

### `count` — the destination is derived from the mask bits themselves

`op_vf_unsqueeze`: an exclusive prefix sum of the mask bits.

`unsqueeze`  *(1)*

### `block` — the bit at each index lane selects a whole block; unselected blocks are written 0

`op_vf_gatherb`: physical predicate bit `4*b`, the first bit of UINT32 index lane `b`, selects
block `b`. The block is copied with every lane, including lanes whose own bit is off; A5 runs
measured this for b16 and b32 predicates (I014).

`gatherb`  *(1)*

### `ignored` — the instruction takes the mask but writes every element

`op_vf_store_interleave`: A5 wrote all 2L elements with empty, prefix, compare and b16 predicates
(RFC-0001), so the whole footprint must be in range.

`store_interleave`  *(1)*

## Keeping this true

`ascriptor/ir/mask_semantics.py` holds the table; this page is its prose. The classification is
checked against the op registry for completeness, so a new operator that takes a `mask` cannot
be added without choosing its role — which is the point. Adding one is the moment the question
is cheap to answer; every later moment is a kernel author guessing.

The roles were read from the only place the answer exists, the reference interpreter
(`backends/sim/vf_ops.py` and `backends/sim/interp.py`), and the three that a kernel can most
easily get backwards — the arithmetic `zero`, the store's `skip`, and the mask booleans that
look like they merge — are pinned by execution rather than by reading, in
`tests/ir/test_mask_semantics.py`.
