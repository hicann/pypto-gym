import { expect, mock, test } from "bun:test";
import { $ } from "bun";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const optionalSchema = () => ({ optional: optionalSchema });
const tool = Object.assign(
  <T>(definition: T): T => definition,
  {
    schema: {
      number: optionalSchema,
      string: optionalSchema,
    },
  },
);

mock.module("@opencode-ai/plugin", () => ({ tool }));

const { PyptoStateTransitionPlugin } = await import("../pypto-state-transition");

type StateTransitionTool = {
  execute: (
    args: Record<string, unknown>,
    context: Record<string, unknown>,
  ) => Promise<string>;
};

async function makeTool(directory: string, worktree = directory): Promise<StateTransitionTool> {
  const plugin = await PyptoStateTransitionPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory,
    worktree,
    project: {},
  } as never);
  return plugin.tool?.state_transition as unknown as StateTransitionTool;
}

test("init creates a missing operator directory under custom", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    const statePath = join(worktree, "custom", "smoke_op", ".orchestrator_state.json");
    expect(existsSync(statePath)).toBeTrue();
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    expect(state.operator_name).toBe("smoke_op");
    expect(state.stage_status["1"]).toBe("in_progress");
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init rejects an output directory outside custom before creating it", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await expect(
      transition.execute(
        {
          action: "init",
          stage: 1,
          max_stage: 7,
          opDir: "other/smoke_op",
        },
        { agent: "build" },
      ),
    ).rejects.toThrow("may only update an operator directory");
    expect(existsSync(join(worktree, "other", "smoke_op"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init resolves custom under the active directory rather than a broader worktree", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-worktree-"));
  const directory = join(worktree, "project");
  mkdirSync(directory);
  try {
    const transition = await makeTool(directory, worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    expect(existsSync(join(directory, "custom", "smoke_op", ".orchestrator_state.json"))).toBeTrue();
    expect(existsSync(join(worktree, "custom", "smoke_op", ".orchestrator_state.json"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("design output failure blocks progress and retry checks both outputs again", async () => {
  const directory = mkdtempSync(join(tmpdir(), "pypto-design-handoff-"));
  let complete = false;
  const calls: string[] = [];
  const shell = (strings: TemplateStringsArray, ...values: unknown[]) => {
    const command = strings.reduce((s, part, i) => s + part + String(values[i] ?? ""), "");
    calls.push(command);
    const result = {
      cwd: () => result,
      quiet: () => result,
      nothrow: async () => ({
        exitCode: command.includes("validate_artifacts.py") && !complete ? 1 : 0,
        stdout: Buffer.from(command.includes("validate_artifacts.py") ? "interface check" : '{}'),
        stderr: Buffer.from(""),
      }),
    };
    return result;
  };
  try {
    const initial = await makeTool(directory);
    await initial.execute({action: "init", stage: 1, max_stage: 7, opDir: "custom/demo"}, {agent: "build"});
    const statePath = join(directory, "custom/demo/.orchestrator_state.json");
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    state.current_stage = 3;
    state.stage_status["1"] = state.stage_status["2"] = "completed";
    state.stage_status["3"] = "in_progress";
    writeFileSync(statePath, JSON.stringify(state));
    const plugin = await PyptoStateTransitionPlugin({ $: shell, directory, worktree: directory,
      client: {app: {log: async () => {}}}, project: {} } as never);
    const transition = plugin.tool?.state_transition as unknown as StateTransitionTool;
    await expect(transition.execute({action: "complete_stage", stage: 3, opDir: "custom/demo"}, {agent: "build"}))
      .rejects.toThrow("pypto-op-architect");
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(3);
    complete = true;
    await transition.execute({action: "complete_stage", stage: 3, opDir: "custom/demo"}, {agent: "build"});
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(4);
    await transition.execute({action: "submit_design", stage: 4, opDir: "custom/demo"}, {agent: "build"});
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(4);
    expect(calls.filter(c => c.includes("validate_artifacts.py"))).toHaveLength(3);
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
});
