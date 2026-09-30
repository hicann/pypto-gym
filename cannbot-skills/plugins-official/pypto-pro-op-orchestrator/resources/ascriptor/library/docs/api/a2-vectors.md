# A2/A3 tensor-vector operations

Import `ascriptor.a2` or `ascriptor.a3` before decorating an entry. These facades
share UB tensor-vector signatures and select different device profiles. An A5
`Reg` operation with the same name is a different operand family.

Facade helpers are compiled DSL calls. Use Python `builtins.abs(value)` for
host scalars when a facade import shadows the built-in name. The old host
fallback of `a2.abs(-1)` is retired. Current `add` accepts FP32, FP16 and INT32;
the old guide's INT16 entry contradicted its own source implementation.
INT16 is separately accepted by `vmax`; one operation's dtype list does not
generalize to another operation.

The [counted arithmetic and select example](../../examples/api/a2_vectors)
uses `count=70` for `2*x+y` and a separate packed predicate buffer for
`compare`/`select`. The [mask-state example](../../examples/api/a2_mask_state)
observes an explicit three-lane mask, a managed count spanning two repeats, and
a following full operation. A managed count restores normal full-mask state;
it does not restore an earlier custom three-lane mask.

`count` and `count_per_rep` are mutually exclusive. The first describes a
contiguous active element domain; the second describes active lanes within each
repeat. Repeat and block strides are in 32-byte blocks. Explicitly allocate the
physical footprint required by the selected instruction even when a logical
tail is shorter. The [reduction/broadcast example](../../examples/api/a2_reduce_broadcast)
reduces rows of 64, 128 and 256 FP32 values in two stages, then broadcasts the
result with `brcb`. Its small integer-valued inputs make the entire comparison
exact, including the intermediate totals.

`SelectMode.TENSOR_TENSOR` uses two UB data sources and an explicit UINT32 UB
`tmp_addr_buf` of at least eight elements. The destination must have a different
starting address from both data sources. `TENSOR_SCALAR` still receives a UB
second source and broadcasts its first element; it rejects address scratch.
The predicate is UINT8 storage containing packed comparison bits. These tokens
are enum-like values, not old `SelectModeType` objects or strings.

The [gather example](../../examples/api/a2_gather) distinguishes one
element per byte offset from one 32-byte block per offset. Element offsets in
its FP32 case are four-byte aligned; block offsets are 32-byte aligned. The
host validator checks the whole accessed interval against the source extent.
The block operation consumes the first eight offsets for one repeat; unused
offset storage does not enlarge the defined input domain. Repeated legal
offsets are supported by the example.

The exact signatures, including explicit repeat/stride overrides and declaration
only arithmetic forms, are in [the public manifest](manifest.json) and
`ascriptor/frontend/dsl_vec.pyi`. The examples cover their stated formats and
shapes. They do not establish every cast pair, aliasing form, scatter collision
rule or vendor instruction restriction from the historical API tables.

Run an exported directory with the installed simulator extra:

```sh
python run.py reference
python run.py check --device a3 --launcher sim
python run.py check --device a3 --launcher pipesim
python run.py emit --device a3 --backend cce
```

Pipe simulation checks the lowered memory/event schedule. Emission writes CCE
sources; vendor compilation and hardware execution are separate stages.
