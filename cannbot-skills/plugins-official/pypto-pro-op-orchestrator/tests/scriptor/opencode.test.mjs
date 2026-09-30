// Run against a real temporary installation whose OpenCode dependencies are present.
import { test, after } from "node:test";
import assert from "node:assert/strict";
import { execFile, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const project = process.env.SCRIPTOR_TEST_PROJECT;
if (!project) throw new Error("set SCRIPTOR_TEST_PROJECT to an isolated installed test project");
const config = path.join(project, ".opencode");
const { PyptoProScriptorPlugin } = await import(pathToFileURL(path.join(config, "plugins/pypto-pro-scriptor.ts")));
const { PyptoProOpLintPlugin } = await import(pathToFileURL(path.join(config, "plugins/pypto-pro-op-lint.ts")));

function shell(_strings, ...values) {
  return { quiet() { return this; }, nothrow() {
    const [python, cli, root, cwd, actor, request] = values;
    return new Promise((resolve) => execFile(python, [cli, "--config-root", root, "state", "--project", cwd,
      "--actor", actor, "--request-json", request], { cwd }, (error, stdout, stderr) => resolve({
      exitCode: error ? 1 : 0, stdout: Buffer.from(stdout || ""), stderr: Buffer.from(stderr || ""),
    })));
  } };
}
const plugin = await PyptoProScriptorPlugin({ directory: project, worktree: project, $: shell });
const legacy = await PyptoProOpLintPlugin({ directory: project, worktree: project, $: shell, client: {} });
const relative = `custom/sdk_probe_${process.pid}`;
const op = path.join(project, relative);
const context = { agent: "build", sessionID: "primary" };
after(() => fs.rmSync(op, { recursive: true, force: true }));

test("registered primary tool executes the real Python state implementation", async () => {
  const text = await plugin.tool.scriptor_transition.execute({ request: JSON.stringify({ action: "init", opDir: relative }) }, context);
  const state = JSON.parse(text);
  assert.equal(state.current_stage, "prepare");
  assert.equal(state.optimization.max_iterations, 5);
  assert.equal(state.optimization.enabled, false);
  assert.deepEqual(state.next_request, { action: "complete_prepare", opDir: relative, report: "reports/sealed/prepare.json" });
  assert.match(state.requires, /verifier/);
  assert.equal(Object.hasOwn(state, "history"), false);
});

test("worker cannot call the primary state tool", async () => {
  await assert.rejects(plugin.tool.scriptor_transition.execute({ request: JSON.stringify({ action: "status", opDir: relative }) },
    { ...context, agent: "pypto-pro-scriptor-worker" }), /primary/);
});

test("workflow hook protects the ledger, frozen inputs and verifier reports", async () => {
  await plugin["chat.params"]({ sessionID: "worker", agent: "pypto-pro-scriptor-worker" }, {});
  const statePath = path.join(op, ".scriptor/state.json");
  const state = JSON.parse(fs.readFileSync(statePath));
  state.current_stage = "implement";
  state.frozen = { "SPEC.md": "test" };
  fs.writeFileSync(statePath, JSON.stringify(state));
  for (const file of [statePath, path.join(op, "SPEC.md"), path.join(op, "reports/sealed/forged.json")]) {
    await assert.rejects(plugin["tool.execute.before"]({ tool: "write", sessionID: "worker", callID: "w" },
      { args: { filePath: file, content: "test" } }));
  }
  await plugin["tool.execute.before"]({ tool: "write", sessionID: "worker", callID: "ok" },
    { args: { filePath: path.join(op, "scriptor/kernel.py"), content: "test" } });
});

test("legacy lint skips scriptor artifacts but retains its original write guard", async () => {
  const flag = path.join(project, ".pypto-pro-op-lint-enabled");
  const previous = fs.existsSync(flag) ? fs.readFileSync(flag) : undefined;
  fs.writeFileSync(flag, "true\n");
  try {
    await legacy["chat.params"]({ sessionID: "primary", agent: "build" }, {});
    await legacy["tool.execute.before"]({ tool: "bash", sessionID: "primary", callID: "new" },
      { args: { command: `cp input.py ${relative}/test_probe.py` } });
    await assert.rejects(legacy["tool.execute.before"]({ tool: "bash", sessionID: "primary", callID: "old" },
      { args: { command: "cp input.py custom/original_probe/test_original_probe.py" } }), /bash\/shell/);
  } finally {
    if (previous === undefined) fs.rmSync(flag, { force: true });
    else fs.writeFileSync(flag, previous);
  }
});

test("shell resource binding uses this installation", async () => {
  const output = { env: {} };
  await plugin["shell.env"]({ cwd: project }, output);
  assert.equal(output.env.CANNBOT_CONFIG_ROOT, config);
  assert.equal(output.env.CANNBOT_PYTHON, JSON.parse(fs.readFileSync(path.join(config, "scriptor-install.json"))).python);
});

test("generated final outputs and checkpoints reject structured edits by every role", async () => {
  for (const actor of ["build", "pypto-pro-scriptor-worker", "pypto-pro-scriptor-verifier", "unknown"]) {
    await plugin["chat.params"]({ sessionID: actor, agent: actor }, {});
    for (const file of ["reports/final/final.json", "reports/final/final.md", "reports/final/final.csv", ".scriptor/checkpoints/probe/checkpoint.json", ".scriptor/receipts/receipt.json"]) {
      await assert.rejects(plugin["tool.execute.before"]({ tool: "write", sessionID: actor, callID: "write" },
        { args: { filePath: path.join(op, file), content: "FIXTURE ONLY" } }));
      await assert.rejects(plugin["tool.execute.before"]({ tool: "apply_patch", sessionID: actor, callID: "patch" },
        { args: { patchText: `*** Begin Patch\n*** Add File: ${path.join(op,file)}\n+FIXTURE ONLY\n*** End Patch` } }));
    }
  }
});

test("sealed Pro inputs and the .scriptor parent require the managed restart CLI", async () => {
  const target = path.join(project, `custom/bootstrap_hook_${process.pid}`);
  const receiptPath = path.join(target, ".scriptor/receipts/pro-bootstrap.json");
  fs.mkdirSync(path.dirname(receiptPath), { recursive: true });
  fs.writeFileSync(receiptPath, "{}\n");
  await plugin["chat.params"]({ sessionID: "bootstrap-primary", agent: "build" }, {});
  await plugin["chat.params"]({ sessionID: "bootstrap-worker", agent: "pypto-pro-scriptor-worker" }, {});
  try {
    for (const file of [target, path.join(target, "sigmoid_golden_cpu.py"), path.join(target, "SPEC.md"),
      path.join(target, ".scriptor")]) {
      await assert.rejects(plugin["tool.execute.before"]({ tool: "write", sessionID: "bootstrap-primary", callID: "sealed" },
        { args: { filePath: file, content: "changed" } }));
    }
    for (const command of [`mv ${target}/.scriptor /tmp/hidden-scriptor`,
      `rm -rf ${target}/.scriptor`,
      `python -c 'Path("${target}/.scriptor").rename("/tmp/hidden-scriptor")'`,
      `python -c 'shutil.move("${target}/.scriptor", "/tmp/hidden-scriptor")'`]) {
      await assert.rejects(plugin["tool.execute.before"]({ tool: "bash", sessionID: "bootstrap-primary", callID: "move" },
        { args: { command } }), /\.scriptor directory/);
    }
    for (const command of [`mv ${target} /tmp/hidden-op`,
      `cd ${project} && rm -rf custom/bootstrap_hook_${process.pid}`]) {
      await assert.rejects(plugin["tool.execute.before"]({ tool: "bash", sessionID: "bootstrap-primary", callID: "move-op" },
        { args: { command } }), /operator directory/);
    }
    await plugin["tool.execute.before"]({ tool: "bash", sessionID: "bootstrap-primary", callID: "prototype-log" },
      { args: { command: `cd ${target} && mkdir -p reports/prototype-logs` } });
    const cli = path.join(config, "scriptor/scripts/scriptor.py");
    const restart = `python ${cli} bootstrap-restart --op-dir ${target} --reason 'repair Golden'`;
    await assert.rejects(plugin["tool.execute.before"]({ tool: "bash", sessionID: "bootstrap-worker", callID: "restart" },
      { args: { command: restart } }), /only the primary/);
    await plugin["tool.execute.before"]({ tool: "bash", sessionID: "bootstrap-primary", callID: "restart" },
      { args: { command: restart } });
    fs.rmSync(receiptPath);
    fs.writeFileSync(path.join(target, ".scriptor/bootstrap-restart.json"), "{}\n");
    await plugin["tool.execute.before"]({ tool: "write", sessionID: "bootstrap-primary", callID: "repair" },
      { args: { filePath: path.join(target, "sigmoid_golden_cpu.py"), content: "repaired" } });
  } finally {
    fs.rmSync(target, { recursive: true, force: true });
  }
});

test("state, seals and derived report regeneration retain their owning roles", async () => {
  const cli = path.join(config, "scriptor/scripts/scriptor.py");
  for (const [actor, command] of [
    ["pypto-pro-scriptor-worker", `python ${cli} state --actor build --request-json '{}'`],
    ["build", `python ${cli} seal --op-dir ${op}`],
    ["pypto-pro-scriptor-worker", `python ${cli} report --op-dir ${op}`],
    ["pypto-pro-scriptor-verifier", `python ${cli} report --op-dir ${op}`],
  ]) {
    await assert.rejects(plugin["tool.execute.before"]({ tool: "bash", sessionID: actor, callID: "forbidden" },
      { args: { command } }));
  }
  for (const [actor, command] of [
    ["build", `python ${cli} report --op-dir ${op}`],
    ["pypto-pro-scriptor-verifier", `python ${cli} seal --op-dir ${op}`],
    ["pypto-pro-scriptor-worker", `cat ${op}/reports/final/final.json`],
  ]) {
    await plugin["tool.execute.before"]({ tool: "bash", sessionID: actor, callID: "permitted" }, { args: { command } });
  }
});

test("direct shell writes to final outputs remain blocked", async () => {
  for (const actor of ["build", "pypto-pro-scriptor-worker", "pypto-pro-scriptor-verifier"]) {
    for (const command of [`cp fixture ${op}/reports/final/final.json`, `echo fixture > ${op}/reports/final/final.md`,
      `python -c 'Path("${op}/reports/final/final.csv").write_text("fixture")'`]) {
      await assert.rejects(plugin["tool.execute.before"]({ tool: "bash", sessionID: actor, callID: "direct" }, { args: { command } }));
    }
  }
});

test("patch renames cannot overwrite a final view or checkpoint", async () => {
  for (const target of ["reports/final/final.md", ".scriptor/checkpoints/best/checkpoint.json"]) {
    const patchText = `*** Begin Patch\n*** Update File: ${op}/scriptor/note.md\n*** Move to: ${op}/${target}\n@@\n-old\n+new\n*** End Patch`;
    await assert.rejects(plugin["tool.execute.before"]({ tool: "apply_patch", sessionID: "build", callID: "move" },
      { args: { patchText } }));
  }
});

test("optional full status returns the untouched v1 ledger through the real tool", async () => {
  const statePath = path.join(op,".scriptor/state.json");
  const before = fs.readFileSync(statePath, "utf8");
  const result = JSON.parse(await plugin.tool.scriptor_transition.execute({request:JSON.stringify({action:"status",opDir:relative,detail:"full"})},context));
  assert.deepEqual(result, JSON.parse(before));
  assert.equal(fs.readFileSync(statePath,"utf8"),before);
});

for (const scenario of ["normal", "unmet"]) {
  test(`registered tool closes ${scenario} fixture and generates the matching summary`, async () => {
    const target = `custom/host_fixture_${scenario}_${process.pid}`;
    const tests = path.dirname(fileURLToPath(import.meta.url));
    const receipt = JSON.parse(fs.readFileSync(path.join(config, "scriptor-install.json")));
    // Fixture creation is explicit; none of these measurements came from hardware.
    const setup = `
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from test_finalization import FinalizationFixture
from scriptorlib.common import atomic_json, read_json
fixture = FinalizationFixture()
fixture.setUp()
try:
    installation = read_json(Path(sys.argv[2]) / '.opencode/scriptor-install.json')
    atomic_json(fixture.config / 'scriptor-install.json', installation)
    original_report = fixture.report
    def report(stage, **kwargs):
        relative = original_report(stage, **kwargs)
        value = read_json(fixture.op / relative)
        value['source_id'] = installation['source_id']
        atomic_json(fixture.op / relative, value)
        return relative
    fixture.report = report
    if sys.argv[4] == 'unmet':
        fixture.set_goal(fixture.goal())
    fixture.advance({'enabled': True, 'max_iterations': 1})
    fixture.act('begin_round')
    fixture.act('finish_round', report=report('candidate', recommendation='reject'))
    final_report = report('accept')
    shutil.copytree(fixture.op, Path(sys.argv[2]) / sys.argv[3])
    print(json.dumps({'report': final_report, 'fixture_evidence': True}))
finally:
    fixture.tearDown()
`;
    const seeded = JSON.parse(execFileSync(receipt.python, ["-c", setup, tests, project, target, scenario], { encoding: "utf8" }));
    assert.equal(seeded.fixture_evidence, true);
    const execute = async (request) => JSON.parse(await plugin.tool.scriptor_transition.execute({ request: JSON.stringify(request) }, context));
    try {
      const status = await execute({ action: "status", opDir: target });
      assert.equal(status.next_action, "finish_optimization");
      const accept = await execute(status.next_request);
      assert.equal(accept.next_action, "complete_accept");
      const done = await execute({ ...accept.next_request, report: seeded.report });
      assert.equal(done.current_stage, "done");
      assert.equal(done.outcome, scenario === "normal" ? "accepted" : "target_unmet");
      assert.equal(done.verdict, scenario === "normal" ? "PASS" : "FAIL");
      assert.equal(done.results.status, "written");
      assert.equal(done.next_action, "read_final");
      const final = JSON.parse(fs.readFileSync(path.join(project, target, "reports/final/final.json")));
      assert.equal(final.fixture_evidence, true);
      assert.equal(final.coverage.verified, 2);
      assert.deepEqual(final.cases.map((row) => row.latency_us), [2, 4]);
    } finally {
      fs.rmSync(path.join(project, target), { recursive: true, force: true });
    }
  });
}
