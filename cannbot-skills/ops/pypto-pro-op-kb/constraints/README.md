# Constraint index

Use these pages as focused supplements to the installed API documentation:

| Constraint | Use for |
|---|---|
| [precision.md](precision.md) | dtype, accumulation, cast, and quantization contracts |
| [tiling.md](tiling.md) | tile shape, layout, and memory-space legality |
| [memory-layout.md](memory-layout.md) | address ownership, layout conversion, and overlap |
| [sync-stitch.md](sync-stitch.md) | tile-group mutexes and section synchronization |
| [tail-validshape.md](tail-validshape.md) | dynamic dimensions and tail windows |
| [vec.md](vec.md) | conditional tile-op / vector-function authoring |
| [vec-alignment-and-rotation.md](vec-alignment-and-rotation.md) | vf lane and reduction-row alignment, buffer rotation, bare-tile sync |
| [vec-mask-width.md](vec-mask-width.md) | converting a mask between b8/b16/b32 element widths |
| [arch-a5.md](arch-a5.md) | A5-only platform discovery and evidence gate |

Do not load a platform-specific page until the target architecture is
confirmed. If a KB statement conflicts with the installed API documentation,
official sample, or a correctness run, use the target environment's evidence
and update the derived design.
