# Source disposition: <path>

Source owner/revision: <read-only input>.
Full SHA-256: <64 hexadecimal characters>.
Full read UTC: <actual ISO timestamp; list intervals if read in segments>.

Disposition: retain | adapt | rewrite | retire | regenerate.
File-specific rationale: <which claims survive, change or disappear and why>.

| Claim / source section | Disposition | Current code/specification/measurement |
|---|---|---|
| <specific behavior> | <retained/corrected/removed> | <versioned evidence> |

Destinations: <successor paths>. Replacement route when retired: <path and reason>.
Discovered links/dependencies: <one pending or resolved record per resource>.

| Destination check | Prerequisites / expected result | Exact command | Actual result / UTC |
|---|---|---|---|
| Python/shell/template | <scope> | <command> | <result> |
| Links and language counterpart | <both source reads and target semantics> | <check> | <result> |
| Isolation/hardware if relevant | <declared dependencies> | <command> | <separate stages> |

Final state: pending | source-reviewed | destination-verified | closed.
Remaining work: <explicit dependencies/checks>. Reading or hashing alone never closes the row.
