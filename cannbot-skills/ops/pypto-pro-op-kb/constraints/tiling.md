# Tile legality

## Rule

Derive every tile from the selected API's documented shape, dtype, memory-space,
layout, alignment, and valid-shape contract. Do not carry limits from another
platform or SDK version.

## Checks

1. Detect the target platform and record the installed PyPTO/CANN version.
2. For each live tile, record shape, dtype bytes, memory space, layout, address,
   slot count, and lifetime.
3. Sum simultaneous allocations separately for each memory space.
4. Compare each sum with the matching platform source/configuration.
5. Confirm offset units independently for `pl.load`, `pl.load_tile`, stores, and
   vector-function APIs.
6. For dynamic tails, keep a legal physical tile and set the runtime valid
   window on every operation that reads or writes it.
7. Re-run correctness for aligned, single-tail, multi-tail, and multi-tile
   cases.

## Two structural ceilings the checks above will not derive

Both are properties of the codegen, not of a platform file, so no amount of
target detection surfaces them — they are found by hitting them.

- **A Tile may have at most two dimensions** (`TileType.md`). A rank-3 view has
  to be flattened into `[outer, inner]` on the host side of the tiling maths.
- **`mutex_ids` must lie in `[0, 31]` and be mutually distinct.** That makes
  **32** the hard ceiling on simultaneous rotating buffer slots. Budget it
  alongside the byte budget in step 3 above: a design that fits in memory can
  still be unbuildable because its slot count does not fit, and the two limits
  are reached by different designs.

Measured on Ascend950PR / CANN 9.2.0; see
[../references/pypto-pro-framework-findings.md](../references/pypto-pro-framework-findings.md)
§"A5 probe session findings". Note the related hazard recorded there: slot
identity is what `auto_mutex` orders against, so a slot counter that steps by
the wrong stride aliases two logical buffers onto one physical one while the
event machinery still issues two credits.

## Cube-specific check

Obtain dtype-dependent contraction geometry and Left/Right/Acc layout
requirements from the installed API and official matmul examples. Do not assume
one K alignment or one fractal shape for all dtypes.

## Evidence

- [single-block matmul](../examples/samples/matmul_float_mmad/matmul_float_mmad_impl.py)
- [BF16 operand reuse](../examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py)
- [dynamic row softmax](../examples/samples/softmax/softmax_impl.py)
- platform discovery: [arch-a5.md](arch-a5.md) for explicit or workflow-default A5;
  numerical limits still require exact target confirmation

The installed `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` pages and official
examples are the primary source for the current version.
