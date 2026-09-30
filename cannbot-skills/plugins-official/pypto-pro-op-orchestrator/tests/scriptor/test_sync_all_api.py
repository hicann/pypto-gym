# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of the License at the root of this repository.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
"""Check emitted barriers against the current keyword-only PyPTO API."""

import ast
from pathlib import Path
import sys
import types
import unittest

SNAPSHOT = Path(__file__).resolve().parents[2] / "resources/ascriptor"
sys.path.insert(0, str(SNAPSHOT / "library"))

import ascriptor.a5 as a
from ascriptor.backends.pypto_pro.emit import PyptoGap, emit_module
from ascriptor.runtime.opexec import lower_kernel


def make_barrier_kernel(mode):
    @a.kernel(mode=mode, block_dim=2)
    def copy_with_barrier(x: a.GM[a.f32, (4, 64)], y: a.GM[a.f32, (4, 64)]):
        local = a.Tensor(a.f32, [1, 64], a.Position.UB)
        with a.auto_sync():
            with a.vec_scope():
                row = a.GetVecIdx()
                local <<= x[row:row + 1, :]
                y[row:row + 1, :] <<= local
                a.allvec_ready(0, pipe=a.Pipe.MTE3)
                a.allvec_wait(0, pipe=a.Pipe.S)
        return y

    return copy_with_barrier


@a.kernel(mode="vec", block_dim=2)
def unpaired_wait(y: a.GM[a.f32, (4, 64)]):
    a.allvec_wait(0, pipe=a.Pipe.S)
    return y


class SyncAllApiTests(unittest.TestCase):
    def test_generated_barrier_calls_match_current_api(self):
        for mode in ("vec", "mix"):
            with self.subTest(mode=mode):
                artifact = emit_module(lower_kernel(make_barrier_kernel(mode)),
                                       sync_mode="auto_mutex")
                source = artifact.files["kernel_pypto.py"].decode()
                tree = ast.parse(source)
                compile(tree, "kernel_pypto.py", "exec")
                calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute)
                         and node.func.attr == "sync_all"]
                self.assertEqual(len(calls), 1)
                participants = object()
                observed = []

                # This deliberately has no mode parameter or SyncAllMode enum.
                # Evaluating the actual emitted call rejects the former spelling.
                def current_sync_all(*, core_type):
                    observed.append(core_type)

                pl = types.SimpleNamespace(
                    system=types.SimpleNamespace(sync_all=current_sync_all),
                    SyncCoreType=types.SimpleNamespace(AIV_ONLY=participants))
                expression = ast.Expression(body=calls[0])
                eval(compile(expression, "kernel_pypto.py", "eval"), {"pl": pl})
                self.assertEqual(observed, [participants])
                self.assertIn("@pl.jit(auto_mutex=True)", source)

    def test_unpaired_wait_is_not_silently_dropped(self):
        with self.assertRaisesRegex(PyptoGap, "without an adjacent allvec_ready"):
            emit_module(lower_kernel(unpaired_wait), sync_mode="auto_mutex")


if __name__ == "__main__":
    unittest.main()
