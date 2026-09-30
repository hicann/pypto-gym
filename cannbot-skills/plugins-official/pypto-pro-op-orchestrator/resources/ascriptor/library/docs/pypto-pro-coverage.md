# Historical PyPTO Pro coverage survey

The original survey and gap ladder describe the prototype investigation. Their
corpus counts, tentative mappings and later corrections are preserved verbatim
in the backend investigation archive, at
this document's original path. They do not describe a current support matrix.

Use the owner for the question being asked:

| Question | Current source |
| --- | --- |
| Which opcode has an emitter, subject to operand restrictions? | [Generated support table](pypto-pro-support.md) |
| How is an admitted IR operation spelled? | [Mapping reference](pypto-pro-mapping.md) |
| Which dependency form is missing, and what evidence supports that? | [Unified upstream report](upstream.md) |
| Which adapter repair or canonical case was validated? | [Backend qualification](a5-backend-coverage.md) |
| Which local dependency supplement is required? | [Dependency supplements](pypto-pro-supplements.md) |
| Can an existing Pro source be imported into IR? | [Import contract](rfc/0015-pypto-pro-import.md) and [fixture index](pypto-pro-import-support.md) |

The former survey is useful for explaining why a mapping was investigated, not
for selecting a runtime or overriding a current refusal. In particular, current
`mergesort4` policy is owned by the
[sorting subset](rfc/0013-pypto-native-synchronization.md#sorting-subset).
Source emission, vendor compilation, board correctness and performance remain
distinct claims with their original source/dependency identities.
