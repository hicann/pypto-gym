# Pipeline plan: <unit / candidate>

Formula and precision boundaries: <original order, state recurrences>.
Allowed changes / target: <correctness, compute overlap, latency; frozen cases>.
Core ownership: <assignment and active participants; items per core>.

Stage groups: <for one-to-one alternating stages, consecutive pairs share an item;
group g handles t-g with guard 0 <= t-g < N; G groups traverse N+G-1 rounds>.

| Stage | Resource / side | Work mapping | Reads / writes | Dependencies / precision |
|---|---|---|---|---|
| <stage> | <M/V/DMA, actual participant> | <f_s(t), domain> | <edges> | <same-item / carried> |

| Edge / storage | Producer | All readers / last physical reader | Live interval | Slots / rotation | Bytes / memory |
|---|---|---|---|---|---|
| <name> | <stage,item> | <stage,item,pipe> | <acquire..release> | <D, slot(item)> | <physical pitch and padding> |

| Handoff | Initial capacity | Publish / acquire | Last-reader release | Final state |
|---|---|---|---|---|
| <edge> | <credit, physical storage> | <actual API and pipes> | <actual API and pipe> | <balanced protocol> |

Warmup / steady / drain: <valid guards; every item exactly once per required stage>.
Recurrence / join: <work mapping and visibility of all needed producers>.
Capacity total: <resident inputs + all live storage, per participant and memory>.
Matched serial control: <same work for a scheduling comparison>.
Validation: <one item; depth boundaries; repeated wraps; supported tails; negative controls>.
Compute overlap: <same-core stage/item intervals; union across vector participants>.
Actual trace / source identity: <mapping and command; no interval inference from source order>.
Remaining restrictions: <unsupported domain or justified serial choice>.
