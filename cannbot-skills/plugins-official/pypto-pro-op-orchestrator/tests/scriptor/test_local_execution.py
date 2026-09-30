# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scriptor checks execute only on the machine that owns the device."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN / "scripts/scriptor-runtime"))
sys.path.insert(0, str(PLUGIN / "resources/ascriptor/library"))

from ascriptor.runtime.board import Board, BoardError
from scriptorlib.runner import _execute


class LocalExecutionTests(unittest.TestCase):
    def test_source_board_rejects_connection_config(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "boards.json"
            config.write_text(json.dumps({"remote": {"ssh": "example", "workspace": temp},
                                          "mixed": {"local": True, "ssh": "example",
                                                    "workspace": temp}}))
            with self.assertRaisesRegex(BoardError, "local=true"):
                Board.from_config("remote", config)
            with self.assertRaisesRegex(BoardError, "unsupported remote connection"):
                Board.from_config("mixed", config)
            self.assertFalse(hasattr(Board, "ssh"))
            config.write_text(json.dumps({"here": {"local": True, "workspace": temp}}))
            board = Board.from_config("here", config)
            self.assertEqual(board._run_shell("printf local", login=False).stdout, "local")

    @staticmethod
    def board_modules(*, local):
        class FakeBoard:
            def require_local(self, what):
                if not local:
                    raise RuntimeError("device's own machine")

        board_type = types.SimpleNamespace(from_config=lambda name, path: FakeBoard())
        return {"ascriptor": types.ModuleType("ascriptor"),
                "ascriptor.runtime": types.ModuleType("ascriptor.runtime"),
                "ascriptor.runtime.board": types.SimpleNamespace(Board=board_type)}

    def test_remote_board_is_refused_before_a_process_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(sys.modules, self.board_modules(local=False)), patch("subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "device's own machine"):
                    _execute(root, [sys.executable, "driver.py"],
                             {"board": "remote", "timeout": 1})
                run.assert_not_called()

    def test_local_board_runs_in_place_without_transfer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(sys.modules, self.board_modules(local=True)), \
                 patch.object(importlib.util, "find_spec", return_value=object()), \
                 patch("subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "ok"
                run.return_value.stderr = ""
                self.assertEqual(_execute(root, [sys.executable, "driver.py"],
                    {"board": "here", "timeout": 1}), 0)
                self.assertEqual(run.call_args.kwargs["cwd"], root)
                self.assertEqual(run.call_args.args[0], [sys.executable, "driver.py"])
                self.assertIn("ok", (root / "execution.log").read_text())


if __name__ == "__main__":
    unittest.main()
