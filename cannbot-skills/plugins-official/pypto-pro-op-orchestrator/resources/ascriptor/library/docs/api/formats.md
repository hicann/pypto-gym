# Packed and exotic format references

The [standalone host example](../../examples/api/host_codecs) imports without
Torch/NumPy. Numerical checks load the declared tensor extra lazily and use
independent integer arithmetic, explicit value tables and midpoint intervals.
The example's kernel only transports UINT8 carriers; host codec arithmetic
and device conversion are separate evidence.

| Format | Carrier and checked meaning |
| --- | --- |
| signed int4 | eight low-first two's-complement nibbles per INT32 word; odd final counts retain explicit logical length |
| FP4 E2M1 | low-first UINT8 pairs; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| FP4 E1M2 | low-first UINT8 pairs; magnitudes 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 |
| legacy host E8M0 | code 0 denotes `2**-127`; code 255 denotes positive infinity |
| HiFloat8 TA | exponent/fraction class enumeration and nearest midpoint intervals with ties away |

The FP4 decode check covers all 256 carrier bytes and signed zero. Host
encoding canonicalizes negative zero to positive zero. The HiFloat8 check
covers all 256 decode codes, then 767 values below/at/above every midpoint and
overflow threshold, plus signed zero, infinities and NaN. FP32 and FP16 input
conversions run under all four saturation/NaN-to-zero settings. Default overflow
at 40960 selects infinity; saturation selects the largest finite value 32768.
NaN output payload bits are not a promise of the decoded FP32 reference.
Hybrid/SSR conversion exists in the library but lies outside this independent
TA reference; it must not inherit its acceptance.

Hardware register conversion can differ from a host helper with a similar
name. The [exponent-cast unit](../../examples/api/exponent_casts)
implements the reviewed BF16 exponent-bit sequences: code 0 decodes to zero
and code 255 to a canonical NaN there. Its reference checks raw bits instead
of calling the legacy host E8M0 decoder.

The [cast-format unit](../../examples/api/cast_formats) distinguishes
single-register 64-bit conversions from packed int4. The defined first 32
logical lanes of a single-register 64-bit form are published with LOWEST32;
an uninitialized second register is never part of its output contract.
INT64-to-FP32 RTZ reference arithmetic preserves large integer low bits until
rounding and corrects overshoot with `nextafter`. Packed int4 narrowing uses
ties to even and saturation, while widening sign-extends each nibble.

The [grouped-register unit](../../examples/api/register_groups)
uses two-register groups to cover full 64-lane INT64/UINT64/complex64 or
128-lane complex32 values. Grouped data and masks have matching extent.
Explicitly initialized scatter destinations retain unwritten lanes. Its raw
byte comparison preserves large integer bits and both complex components.

An A2 INT32-backed int4 cube view is a separate contract from A5 packed
register conversion. `reinterpret` labels a view; it does not pack or quantize
data. Typed GM carriers and logical K must agree with the consuming operation.
On A2/A3, odd-K INT4 MMAD reads the complete final byte; its unused high
nibble must be zero for a logical-K product. The [tail example](../../examples/api/a2_int4_tail)
normalizes private carriers on the device while preserving arbitrary input padding.
Packed logical elements have no ordinary byte-addressable allocation size.
Do not infer that a dtype name permits direct GM/local allocation in every
position or backend.

The dated vendor cast matrix beside `cast_formats` is retained for compiler
table consistency. Tests preserve documented pair coverage, layout/rounding
constraints, merge refusals and explicit synthesized E8M0 recipes. The table's
historical vendor and board notes are provenance, not successor hardware gates.

## Cast policy and destination state

`CastConfig` carries a round mode, register layout, saturation request, optional
name and mask merge mode. Configuration construction is not evidence that a
dtype pair or backend can express every field. The current C310 CCE pair table
rejects `MaskMergeMode.MERGING`; it does not inherit the old CANN-wrapper guide's
claim of support with layout ZERO. `tests/product/test_api_cast_matrix.py`
checks the captured pair/shape/rounding matrix and the refusal, while
`test_api_cast_controls.py` checks scalar CTRL values and dynamic restoration
through the source emitters.

Round modes include NONE, ties-to-even, ties-away, FLOOR, CEIL, TRUNC, ODD and
HYBRID, with pair-specific legality. A round mode does not supply a missing
instruction operand. The [saturation example](../../examples/api/cast_saturation)
checks per-instruction and CTRL-driven behavior from all four initial control
bit combinations. [Flag observation](../../examples/api/saturation_flags)
is a separate small case. Both restore the state they save; references do not
substitute a generic `torch.to` rule for saturation and rounding.

For ordinary byte-addressable width changes, widening selects one source slot
per destination lane and narrowing writes selected destination slots. Which slot
is `reg_layout`, and for the half/single pair it is a lane parity, not a half of
the register. Measured on an Ascend950 card through CCE (2026-09-24, two repeats,
identical), and the simulator agrees on every row. The 950 family shares this
behaviour, so the card is the device and not one board:

| `reg_layout` | f16 -> f32 widen (128 -> 64 lanes) | f32 -> f16 narrow (64 -> 128 lanes) | CCE |
| --- | --- | --- | --- |
| `ZERO` | reads source lanes 0, 2, ... 126 | writes destination lanes 0, 2, ... 126 | `PART_EVEN` |
| `ONE` | reads source lanes 1, 3, ... 127 | writes destination lanes 1, 3, ... 127 | `PART_ODD` |

`bf16`/`f32` behaves identically. Whether the field is read at all is a property
of the pair's **shape family** in `CAST_SHAPES` (`backends/cce/arch/c310.py`),
which names the instruction's tag sequence. One representative of every family
that plain GM dtypes can express was measured, in the model and on the card, and
the two agreed line for line:

| Families | Positions | What `reg_layout` does |
| --- | --- | --- |
| `part`, `sat_part`, `rnd_sat_part`, `rnd_part` | 2 | `ZERO` / `ONE` select the even / odd phase |
| `part_t`, `sat_part_t`, `rnd_sat_part_t`, `rnd_part_t` | 4 | `ZERO`…`THREE` select one of four phases (`PART_P0`…`P3`) |
| `rnd`, `rnd_sat`, `sat_rnd` | 1 | Same-size pair: destination lane k takes source lane k. **The field is ignored** |
| `b64_widen`, `b64_from_f32`, `f32_from_b64` | 1 | Two-register 64-bit form with no part selector: lane k takes lane k. **The field is ignored** |

The last two rows are where a layout is accepted and silently means `ZERO`: an
`i32 -> i64` widen emits `vcvt(w, a)` byte for byte under either layout, so a
reader reaching for the upper half gets the lower one with no diagnostic.
`pypto_pro` refuses the 64-bit pairs outright as an upstream gap, so only cce and
pto_isa express them at all. The four-position families are the packed and 8-bit
carriers; [cast_formats](../../examples/api/cast_formats) exercises
them.

**`ONE` needs a half-width mask.** The predicate is sampled at the selected
*source* element positions, and a b32 `MaskReg` has no active bit at an odd f16
position: the same cast that returns lanes 1, 3, ... under `MaskReg(DT.half)`
returns **all zeros, and raises nothing**, under `MaskReg(DT.float)`. That is the
card's own behaviour, not a model artefact — the board run above produced the
same all-zero register. A result that is entirely zero is the symptom of the mask
width, not of the layout.
`ZERO` emits on cce, pto_isa and pypto_pro; `ONE` is refused by pypto_pro at its
source line (`vf.cast layout 'one' has no CastLayout spelling`), so a kernel that
consumes both parities does not reach that backend.

Packed
INT4/FP4 mappings and grouped-register forms have their own carrier geometry.
The unit contracts identify which lanes/bytes are defined. Checking only a
leading prefix is valid only when that prefix is explicitly the output ABI;
unwritten storage cannot silently be treated as zero.

The historical plain-Cast ONE-layout restrictions and Hybrid/SSR observations
remain dated source context. Current declarations, generated coverage and
located backend refusals determine what can be emitted. Actual vendor and
board gates determine the supported hardware domain.
