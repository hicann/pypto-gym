# Documentation map

Kernel authoring and substantial kernel work start with the
[compact kernel context](../context/kernel-authoring.zh-CN.md). Its route map and
authoring steps cover the initial Chinese single-kernel author route; other tasks
open the [English](../en/ROUTER.md) or [Chinese](../zh-CN/ROUTER.md) router and one
task playbook. Read the guide directly, or run `python tools/build_kernel_context.py --print`
in the selected agent directory. It has no source-hash dependency or refresh step.
The same context already embedded in the initial message can satisfy included reading
for its declared scope under [AGENTS.md](../AGENTS.md). Other tasks start at the router.
The compact guide is an authored start, not a substitute for task-specific owner
APIs and examples; update its prose when changes to those sources affect the guidance.
Compiler snapshot and validation artifact integrity checks remain separate.
The library owns [design constraints](../../library/docs/decisions.md) and
[product contracts](../../library/docs/rfc/0012-product-contracts.md); workflow
guidance links to those rules rather than maintaining a second specification.

| Need | Document |
| --- | --- |
| Source identity | [Snapshot manifest](../../sources.json) |
| API and example discovery | [Index guide](../index/README.md) |
| Backend limitations | [Library upstream report](../../library/docs/upstream.md) |
| Product contract | [RFC-0012](../../library/docs/rfc/0012-product-contracts.md) |

[Code anchors](code-anchors.json) follow exact owner functions.

Document repository Python commands relative to their owner checkout. For a
shell block that runs from another owner, put `<!-- checked-command: library -->`
(or `agent` / `kernels`) immediately before its `sh`/`bash` fence and state the
working directory in the prose. The checker validates script targets without
executing commands; copied scratch scripts, module invocations and runtime
arguments retain their separate validation requirements. Short code excerpts
continue to use the existing owner [code anchors](code-anchors.json).
