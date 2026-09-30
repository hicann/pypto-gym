# Current library status

This is the first public Ascriptor source release, version 0.1.0. The plugin's
[sources.json](../../sources.json) and [sources-index.json](../../sources-index.json)
identify every delivered file by SHA-256. The compiler runs from the source tree;
there is no Ascriptor wheel or private repository dependency.

The [API guide](api/README.md) describes declared operations. The
[API example index](../examples/api/index.json) and
[kernel gallery index](../../kernels/index.json) are navigation. A declaration or
example does not establish hardware, backend, numerical or performance support
for another workload. Run the requested workload on the assigned device and
record its source identity, device environment and comparison result.

The `sim` and `pipesim` launchers provide model evidence. Device launchers run
only on the machine holding the card, using local configuration; see the
[execution guide](api/execution.md#running-the-whole-unit-on-the-device-machine).
CANN, PyPTO-Pro, torch_npu and device drivers are external environment
requirements. The [defect index](defects/README.md) and
[upstream report](upstream.md) describe known implementation boundaries.

No general hardware or performance qualification is claimed for this first
source release. Device acceptance remains specific to each verified case.
