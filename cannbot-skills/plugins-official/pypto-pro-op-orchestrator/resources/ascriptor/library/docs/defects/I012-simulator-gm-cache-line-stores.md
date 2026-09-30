# I012 — Cross-core GM scalar stores sharing a cache line

Status: open model gap; the model keeps every store and warns when a writer drops
the protocol A5 needs. Owner: `backends/sim/pipesim.py`.

RFC-0006 checks GM store hazards on exact byte footprints, so two cores storing
different elements of one 64-byte cache line raise no hazard under
`check_gm=True`, and the interpreter keeps both values. On A5 a scalar store into
such a line is lost unless the writer cleans the line after the store and its
cross-core publication cannot run before the store. The publication rides a
memory pipe (`ffts_cross_core_sync` on MTE3, since an AIV set compiles only on
PIPE_V/MTE2/MTE3), while `setval` and `dcci` run on the scalar pipe, so without a
wait the next core is released before the store has happened.

Four AIVs, each storing one element of line 0 of an INT32 tensor whose data
pointer is 64-byte aligned, 20 launches per probe
(protocol probes,
controls):

| writer | producer dcci | publication | result | job |
| --- | --- | --- | --- | --- |
| scalar | after the store | MTE3 waits for S | every store kept | 363 |
| scalar | after the store | `bar_all` before it | every store kept | 363 |
| scalar | after the store | no wait | two of four lost, every launch | 363 |
| scalar | none | MTE3 waits for S | three of four lost, every launch | 366 |
| scalar | after the store only | MTE3 waits for S | every store kept | 366 |
| scalar | any placement | none, writers concurrent | two or three of four lost | 310–312, 363 |
| DMA (MTE3) | with, without or ENTIRE | ordered or concurrent | every store kept | 363, 366 |

The consumer's own dcci before its store is not needed: keeping only the
producer's kept every store. The wait of the scalar pipe for the storing pipe
before the dcci was not load-bearing in any measured variant, but every variant
that ran without it published on the storing pipe itself. Earlier probes agree:
concurrent stores lost two or three of four with dcci before or after the store,
on one line, the entire cache or another line (jobs 310–312), and a rendezvous
whose publication did not wait lost two or three of four with every dcci
placement (job 338, serialized probes).
In every probe a lost element read back its value from before the store, never a
value from another launch. Imported CCE runs fell in the same classes.

This is the dcci page's protocol: a producer syncs to the scalar pipe before its
dcci or its flag publication, and the publication establishes the order the next
writer waits on. With `check_gm`, pipesim warns per pair of source operations
when two cores' scalar stores share a line and either nothing orders them or the
earlier writer misses its clean or its publication order (RFC-0006 §9); the
warning names both stores and what is missing. DMA stores measured clean and are
not checked. One cache line per core also keeps every store, as
the identity kernels do.
