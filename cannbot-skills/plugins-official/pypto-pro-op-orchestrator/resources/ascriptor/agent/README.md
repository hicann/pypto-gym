# Ascriptor agent guidance

Choose [English](en/ROUTER.md) or [中文](zh-CN/ROUTER.md), then one task playbook.
Kernel work starts with the common-language baseline and implementation preflight;
follow the triggered references and one relevant runnable example.

These guides accompany the first public 0.1.0 source snapshot. Use its `library/`,
`agent/` and `kernels/` directories together; [sources.json](../sources.json) records
the exact bytes. Validate each device and workload before claiming a result.

| Need | English | 中文 |
| --- | --- | --- |
| API signatures and operations | [API entry](../library/docs/api/README.md) | [API 入口](../library/docs/api/README.md) |
| Runnable primitives and algorithms | [API examples](../library/examples/api/README.md), [kernel gallery](../kernels/index.json) | [API 样例](../library/examples/api/README.md)、[Kernel 画廊](../kernels/index.json) |
| First source run | [Quickstart](en/quickstart.md) | [快速开始](zh-CN/quickstart.md) |
| Run what you write, here and on a card | [Running what you are writing](en/references/development-execution.md) | [开发期怎么跑](zh-CN/references/development-execution.md) |
| Execution and ownership | [Concepts](en/concepts.md) | [执行模型](zh-CN/concepts.md) |
| Find a working example | [Patterns](en/references/patterns.md) | [样例选择](zh-CN/references/patterns.md) |
| Plan memory and pipelines | [Pipeline model](en/references/pipeline-model.md) | [流水方法](zh-CN/references/pipeline-model.md) |
| Understand cost | [Roofline](en/references/roofline.md) | [成本与 Roofline](zh-CN/references/roofline.md) |
| Write attention | [Attention](en/references/attention-authoring.md) | [Attention 编写](zh-CN/references/attention-authoring.md) |
| Diagnose and maintain | [Runtime and maintenance](en/runtime-and-maintenance.md) | [运行与维护](zh-CN/runtime-and-maintenance.md) |

The selector ranks ids, owner paths and maintained phrases. It no longer filters on
dtype, layout or backend, because a kernel demo declares none of them. For a library
unit, read its contract and recorded evidence before making a claim. A kernel demo
records nothing about where it has run; run its `main.py` on the machine whose answer
you need. Gallery size and a worked example do not establish support for a new input
domain. Historical qualification details remain with their owners; guidance trials are in
the evaluation results.

The library owns APIs, specifications and [defects](../library/docs/defects/README.md).
Kernels owns the runnable demos and their independent references. The
[upstream report](../library/docs/upstream.md) owns backend gaps and workarounds; the
gallery records no case outcomes at all. Where a backend cannot run a demo, the reason is
a `# pypto_pro:` comment in that demo's own `main.py`, next to the code it is about.
Coverage and the evaluation protocol
distinguish maintained guidance, deterministic checks and fresh-context trials.

Use the library, agent and kernels directories of this self-contained snapshot together.
The snapshot root's `sources.json` and `sources-index.json` define its exact file identity.
No external source checkout or release wheel is needed. Run fresh checks for the
selected device and workload.
