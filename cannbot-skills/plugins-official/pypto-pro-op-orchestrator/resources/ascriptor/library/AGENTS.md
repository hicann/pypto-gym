# Library maintenance

Start with `docs/README.md`, `docs/status.md`, the applicable RFC and
`docs/decisions.md`. Follow shared maintenance rules for
environment binding, independent references, evidence, privacy and coordination.

The RFC is the specification; amend it first with a justified semantic correction.
Use English for code/specifications. Keep signatures compact. Pass rewrites use
Rewriter and explanation notes. CCE prints Lowered IR; nontrivial lowering belongs
in passes. Simulator internals may be inspected and repaired, with model rules
distinguished from measured silicon behavior.

Public API facts belong to `docs/api/README.md` and `docs/api/manifest.json`;
teaching primitives to `examples/api/README.md`; implementation defects to
`docs/defects/`; dependency restrictions to `docs/upstream.md`.

Pro reverse import follows `docs/rfc/0015-pypto-pro-import.md` and its generated
support index. Admit forms from real Pro export fixtures and measured semantics;
refuse unsupported forms at their source span instead of approximating them.

Keep task artifacts in ignored `tmp/<task>/`. An example is a folder of
four files -- `main.py`, `kernel`, `reference`, `metadata.json` -- and `main.py` is both its
entry point and its precision check: `cd examples/api/NAME && python main.py`, with
`--launcher pipesim` for the pipe model and `--list` for its cases. There is no shared runner
and no contract file; `python tools/api_examples.py --check` is the gate on the shape and on
`examples/api/index.json`. Keep task output in ignored `tmp/<task>/` and run the documented
checks for the affected behavior.
