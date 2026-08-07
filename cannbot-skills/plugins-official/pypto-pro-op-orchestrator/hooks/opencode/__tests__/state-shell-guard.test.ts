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

import { isStateWriteCommand } from "../lib/lint-output.ts";

const state = "custom/demo/.orchestrator_state.json";

test("allows explicit read-only state inspection", () => {
  const commands = [
    `cat ${state}`,
    `grep current_stage ${state}`,
    `rg current_stage ${state}`,
    `git diff -- ${state}`,
    `git status --short -- ${state}`,
    `git show HEAD:${state}`,
    `cat ${state} | jq .current_stage`,
    `cat ${state} 2>/dev/null`,
    `git diff -- ${state} > /tmp/state.patch`,
    `realpath ${state}`,
    `echo ${state}`,
  ];
  for (const command of commands) {
    assert.equal(isStateWriteCommand(command), false, command);
  }
});

test("blocks redirection and mutating commands targeting state", () => {
  const commands = [
    `cat payload > ${state}`,
    `grep current_stage payload > ${state}`,
    `cat payload 2>${state}`,
    `tee ${state}`,
    `rm ${state}`,
    `sed -i s/1/2/ ${state}`,
    `git diff --output=${state}`,
  ];
  for (const command of commands) {
    assert.equal(isStateWriteCommand(command), true, command);
  }
});

test("blocks mutation hidden in a command chain or substitution", () => {
  const commands = [
    `cat ${state} && rm ${state}`,
    `cat ${state}; truncate -s 0 ${state}`,
    `cat ${state} | xargs rm`,
    `cat ${state} $(rm ${state})`,
    `cat ${state}\nrm ${state}`,
  ];
  for (const command of commands) {
    assert.equal(isStateWriteCommand(command), true, command);
  }
});

test("ignores shell commands unrelated to orchestrator state", () => {
  assert.equal(isStateWriteCommand("rm custom/demo/test_demo.py"), false);
});
