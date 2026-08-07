// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatGateFinding,
  parseGateSummary,
  parseHookOutput,
} from "../lib/lint-output.ts";

test("post-edit empty output fails closed", () => {
  assert.equal(parseHookOutput("").decision, "block");
});

test("post-edit malformed or incomplete output fails closed", () => {
  assert.equal(parseHookOutput("not-json").decision, "block");
  assert.equal(parseHookOutput("{}").decision, "block");
});

test("post-edit valid decision is preserved", () => {
  const raw = JSON.stringify({
    hookSpecificOutput: {
      decision: "block",
      reason: "PL01 failed",
      additionalContext: "details",
    },
  });
  assert.deepEqual(parseHookOutput(raw), {
    additionalContext: "details",
    decision: "block",
    reason: "PL01 failed",
  });
});

test("gate output parser returns failures and counters", () => {
  const raw = JSON.stringify({
    findings: [
      { rule_id: "PL01", severity: "S0", status: "FAIL", message: "bad", file: "x.py" },
      { rule_id: "PL02", severity: "S0", status: "PASS", message: "ok", file: "x.py" },
    ],
    summary: {
      pass: 1,
      warn: 0,
      info: 0,
      fail: 1,
      skip: 0,
      has_error_fail: true,
    },
  });
  assert.deepEqual(parseGateSummary(raw), {
    warnCount: 0,
    infoCount: 0,
    failFindings: [
      { rule_id: "PL01", severity: "S0", status: "FAIL", message: "bad", file: "x.py" },
    ],
  });
});

test("gate output parser rejects empty, malformed, or incomplete output", () => {
  assert.throws(() => parseGateSummary(""), /empty output/);
  assert.throws(() => parseGateSummary("not-json"), /invalid lint JSON output/);
  assert.throws(
    () => parseGateSummary(JSON.stringify({
      summary: { pass: 0, warn: 0, info: 0, fail: 0, skip: 0, has_error_fail: false },
    })),
    /findings/,
  );
});

test("gate output parser rejects coercible or inconsistent summaries", () => {
  const finding = {
    rule_id: "PL01",
    severity: "S0",
    status: "FAIL",
    message: "bad",
    file: "x.py",
  };
  assert.throws(
    () => parseGateSummary(JSON.stringify({
      findings: [finding],
      summary: { pass: 0, warn: "0", info: 0, fail: 1, skip: 0, has_error_fail: true },
    })),
    /summary.warn/,
  );
  assert.throws(
    () => parseGateSummary(JSON.stringify({
      findings: [finding],
      summary: { pass: 0, warn: 0, info: 0, fail: 0, skip: 0, has_error_fail: false },
    })),
    /does not match/,
  );
});

test("gate output parser rejects malformed finding severity", () => {
  assert.throws(
    () => parseGateSummary(JSON.stringify({
      findings: [
        { rule_id: "PL01", severity: "", status: "FAIL", message: "bad", file: "x.py" },
      ],
      summary: { pass: 0, warn: 0, info: 0, fail: 1, skip: 0, has_error_fail: false },
    })),
    /severity/,
  );
});

test("gate output parser rejects non-canonical finding status", () => {
  assert.throws(
    () => parseGateSummary(JSON.stringify({
      findings: [
        { rule_id: "PL01", severity: "S0", status: "fail", message: "bad", file: "x.py" },
      ],
      summary: { pass: 0, warn: 0, info: 0, fail: 1, skip: 0, has_error_fail: false },
    })),
    /invalid finding status/,
  );
});

test("gate failure formatter does not repeat a file already in the message", () => {
  const base = { rule_id: "PL01", severity: "S0", status: "FAIL" };
  assert.equal(
    formatGateFinding({ ...base, message: "test_x.py: bad import", file: "test_x.py" }),
    "  [PL01][S0] test_x.py: bad import",
  );
  assert.equal(
    formatGateFinding({ ...base, message: "missing", file: "SPEC.md" }),
    "  [PL01][S0] missing (SPEC.md)",
  );
});
