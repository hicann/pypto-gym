# A2-B32-ND2NZ-L1-OVERWRITE — an int32 L1 tile took the c220 cube's fp32 ZZ layout

Status: **closed**, 2026-09-24. Scope: A2 / A3 (`c220`) CCE, any 32-bit **integer** L1 tile.

The c220 cube keeps fp32 L1 tiles in the ZZ fractal layout and everything else in NZ. The backend
selected ZZ with `sizeof(T) == 4`, so an int32 tile — which is what an int4 operand's carriers are
— was written by `gm_to_l1_nd2nz` as ZZ, read back by `l1_to_l0` as ZZ, and folded by the printer's
window arithmetic as ZZ, while `addr_alloc` and the functional model both had it as NZ. A ZZ tile
of `[64, 8]` int32 occupies `align16(64) * align16(8) * 4 = 4096` bytes against an NZ footprint of
2048, so a narrow tile ran over whatever the allocator had placed after it.

The old framework tests `dst.dtype is Datatype.float` for exactly this (`asc_handlers/cube.py`),
and its `l1_to_l0` splits four ways — `L0ZZ2ZZ` / `L0ZZ2NZ` for float, `L0NZ2ZZ` / `L0NZ2NZ`
otherwise, L0A taking ZZ and L0B NZ. The port kept the four branches and lost the predicate.

**The ID is the handle the work was done under; two thirds of it are wrong.** It is not about
`nd2nz` — the NZ route was never at fault — and "B32" names the symptom's width rather than the
mistake, which was testing a width where a dtype was meant.

## How it presented, and the two wrong diagnoses

On an A2 card, two adjacent `[64, 8]` int32 tiles: whichever was **loaded first** lost rows 32-63.
Swapping which tile is declared first moves the damage from the M axis to the N axis, which is what
an overlapping write looks like and nothing else does. Every emitted instruction matched the
shipped `a2_int4_tail` unit's for the same shape, so the transfer arguments were never the
difference — only the two addresses were.

1. **"The trigger is the 32-bit width."** Read off the int32 cases alone. An `f16 [64, 8]` tile was
   then measured and *also* found damaged, which looked like a refutation and was taken as one.
2. **"`nd2nz` pads its destination's column count to 16 elements, dtype-independently."** This fit
   every measurement: f16 `[64, 8]` needs 2048 and got 1024; int32 `[64, 8]` needs 4096 and got
   2048. It was wrong about both cases for compensating reasons — f16's NZ granule `C0` *is* 16
   elements, and int32's 4096 came from ZZ, not from padding. `addr_alloc` reserved
   `align16(cols) * rows * width`, which is the ZZ footprint applied to every tile, so the card
   went clean and the diagnosis survived.

What broke it open was reading the CANN parameters the repair was supposedly based on
(`data_copy_wrapper_nd.h:99`, `kernel_operator_data_copy_impl.h:751`, `BLOCK_CUBE = 16`) and
finding they say a `[64, 8]` int32 NZ tile writes 2048 — so the measured damage had no mechanism
under the stated rule. **The rule explained the f16 rows and not the int32 one, and that was
visible in the arithmetic long before anyone looked.**

## The fix

Three sites tested the element width where they meant the dtype:

| | was | now |
| --- | --- | --- |
| `tensorutils_cce.h` `gm_to_l1_nd2nz` (c220) | `sizeof(T) == 4` | `is_same<T, float>::value` |
| `tensorutils_cce.h` `l1_to_l0` (c220) | `sizeof(T) == 4` | `is_same<T, float>::value` |
| `backends/cce/views.py` `elem_bytes_offset` | `esz == 4` | `g.dtype.name == "f32"` |

A 32-bit integer tile then needs the NZ write the c220 branch did not have — it would have fallen
through to the b8 form — so `copy_gm_to_cbuf_multi_nd2nz_b32s` joins the b16 and b8 cases with the
ordinary NZ parameters.

`addr_alloc` is re-derived from the layouts rather than from the symptom. Both pad, and differ only
in the column granule:

    bytes = align16(rows) * align_up(cols, G) * width        G = 16 for ZZ, else C0 = 32 / width

NZ keeps `ceil(cols / C0)` fractal columns of `dstNzC0Stride = align16(rows)` rows; ZZ keeps
`rows / 16` bands of `align16(cols) * 16` elements. This is smaller than the old reservation for
32-bit integers and **larger** for two shapes it had been under-reserving: int8 tiles with a
column count that is not a multiple of 32 (`C0` is 32 elements there), and any tile with fewer
than 16 rows.

## Evidence

On an A2 card, after the fix — the int32 rows at the *reduced* 2048 reservation, which is what
distinguishes a repair from a mask:

| probe | before | after |
| --- | --- | --- |
| int4 carriers = 8 (int32 NZ, 2048 B reserved) | rows 32-63 destroyed | **clean**, 0 of 4096 |
| int4 carriers = 16 | clean | clean |
| fp32 cols = 8, 16, 64 (genuinely ZZ, 4096 B) | — | **clean**, 0 of 4096 |

The int4 fused bias and its split-K form re-qualify exact on the same card. The gallery emits
byte-identically except for six a5 kernels (`online_mx` ×2, `simt_transpose` ×2, `a5_decode_fp8`,
`a5_mla`), established by digesting all 114 kernels under both revisions with the shared headers
excluded — including them made a header edit look like all 109 had moved.

The guard is `tests/passes/test_l1_tile_reserves_its_physical_footprint.py`. The assertion that
matters is `test_only_fp32_takes_the_c220_zz_layout`: an fp32 and an int32 tile of the same shape
on the same device must reserve **differently** (4096 against 2048). A test keyed on the element
width cannot express that, which is why the first one did not catch this.

## What stays open

- **Nothing below a card can see an overlapping write.** D-022 makes every DMA a logical window
  copy, so `sim` and `pipesim` model no physical layout by construction. The reservation is the
  only thing between a kernel and this class of failure, and a lint or a model warning would still
  be worth more than the reservation alone.
- The model labels every L1 tile `nz`; ZZ exists only inside the backend and `addr_alloc`'s
  device-and-dtype test. That is a second place where the two could drift apart.
