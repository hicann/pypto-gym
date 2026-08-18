# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import contextlib
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
TIMELINE = importlib.import_module("instruction_timeline")


class Stage5TimelineTest(unittest.TestCase):
    def test_instruction_timeline_uses_unique_target_and_same_lane_overlap(self) -> None:
        target = "_Ztarget"
        events = [
            {
                "name": target, "pid": 10, "tid": 61,
                "ts": "1786626512687837.751", "dur": "20.000", "ph": "X",
                "args": {"Task Type": "AI_CORE"},
            },
            {
                "name": "thread_name", "pid": 20, "tid": 0, "ph": "M",
                "args": {"name": "Group0-aiv0"},
            },
            {
                "name": "thread_name", "pid": 20, "tid": 1, "ph": "M",
                "args": {"name": "Group0-aiv1"},
            },
            {
                "name": "MTE2", "pid": 20, "tid": 0,
                "ts": "1786626512687840.001", "dur": "6.000", "ph": "X",
                "args": {"Core Type": "aiv0", "Block Id": 0},
            },
            {
                "name": "VEC", "pid": 20, "tid": 0,
                "ts": "1786626512687844.501", "dur": "5.000", "ph": "X",
                "args": {"Core Type": "aiv0", "Block Id": 0},
            },
            {
                # It overlaps in wall-clock time but is a different BIU lane,
                # so it must not be counted as single-lane pipeline overlap.
                "name": "MTE3", "pid": 20, "tid": 1,
                "ts": "1786626512687845.000", "dur": "5.000", "ph": "X",
                "args": {"Core Type": "aiv1", "Block Id": 14},
            },
            {
                "name": "FUTURE_PIPE", "pid": 20, "tid": 0,
                "ts": "1786626512687852.000", "dur": "1.000", "ph": "X",
                "args": {"Checkpoint Info": 9},
            },
        ]
        analysis = TIMELINE.analyze_instruction_timeline(events, target)
        self.assertEqual(analysis["attribution_status"], "unique_exact_target_window")
        self.assertEqual(analysis["pipeline_evidence_status"], "requires_tile_dag_correlation")
        self.assertFalse(analysis["completion_eligible"])
        overlap = analysis["same_lane_pipe_interval_overlap"]
        self.assertEqual(overlap["raw_event_pair_counts"], {"compute+movement": 1})
        self.assertEqual(overlap["pair_overlap_us"]["compute+movement"], "1.500")
        self.assertEqual(
            overlap["lanes"][0]["pairs"][0]["deduplicated_overlap_us"], "1.500"
        )
        self.assertEqual(analysis["instruction_event_counts"]["FUTURE_PIPE"], 1)

    def test_instruction_timeline_fails_on_ambiguous_or_cross_lane_only_evidence(self) -> None:
        target = {"name": "target", "pid": 1, "tid": 1, "ts": "100", "dur": 10, "ph": "X"}
        lane0 = {
            "name": "thread_name", "pid": 2, "tid": 0, "ph": "M",
            "args": {"name": "Group0-aiv0"},
        }
        lane1 = {
            "name": "thread_name", "pid": 2, "tid": 1, "ph": "M",
            "args": {"name": "Group0-aiv1"},
        }
        movement = {
            "name": "MTE2", "pid": 2, "tid": 0, "ts": "101", "dur": 2,
            "ph": "X", "args": {"Core Type": "aiv0", "Block Id": 0},
        }
        compute = {
            "name": "VEC", "pid": 2, "tid": 1, "ts": "101.5", "dur": 2,
            "ph": "X", "args": {"Core Type": "aiv1", "Block Id": 14},
        }
        with self.assertRaisesRegex(ValueError, "found 2"):
            TIMELINE.analyze_instruction_timeline(
                [target, target, lane0, movement], "target"
            )
        analysis = TIMELINE.analyze_instruction_timeline(
            [target, lane0, lane1, movement, compute], "target"
        )
        self.assertEqual(
            analysis["same_lane_pipe_interval_overlap"]["raw_event_pair_counts"], {}
        )

    def test_instruction_timeline_deduplicates_same_family_interval_union(self) -> None:
        events = [
            {"name": "target", "pid": 1, "tid": 1, "ts": "0", "dur": "10", "ph": "X"},
            {
                "name": "thread_name", "pid": 2, "tid": 0, "ph": "M",
                "args": {"name": "Group0-aiv0"},
            },
            {
                "name": "MTE2", "pid": 2, "tid": 0, "ts": "0", "dur": "10", "ph": "X",
                "args": {"Core Type": "aiv0", "Block Id": 0},
            },
            {
                "name": "MTE3", "pid": 2, "tid": 0, "ts": "2", "dur": "6", "ph": "X",
                "args": {"Core Type": "aiv0", "Block Id": 0},
            },
            {
                "name": "VEC", "pid": 2, "tid": 0, "ts": "4", "dur": "2", "ph": "X",
                "args": {"Core Type": "aiv0", "Block Id": 0},
            },
        ]
        overlap = TIMELINE.analyze_instruction_timeline(
            events, "target"
        )["same_lane_pipe_interval_overlap"]
        # Both movement records overlap VEC, so there are two raw event pairs;
        # their merged family union intersects VEC for only two microseconds.
        self.assertEqual(overlap["raw_event_pair_counts"], {"compute+movement": 2})
        self.assertEqual(overlap["pair_overlap_us"], {"compute+movement": "2"})
        self.assertEqual(overlap["lanes"][0]["pairs"], [{
            "pipe_pair": "compute+movement",
            "raw_event_pairs": 2,
            "deduplicated_overlap_us": "2",
        }])

    def test_biu_database_contract_is_checked_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "biu_perf.db"
            import sqlite3
            with contextlib.closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE BiuInstrStatus ("
                    "group_id INTEGER, core_type TEXT, block_id INTEGER, "
                    "instruction TEXT, timestamp INTEGER, duration INTEGER, "
                    "checkpoint_info INTEGER)"
                )
                connection.execute(
                    "INSERT INTO BiuInstrStatus VALUES (0, 'aiv0', 0, 'MTE2', 1, 2, NULL)"
                )
                connection.commit()
            record = TIMELINE.validate_biu_database(path)
            self.assertEqual(record["row_count"], 1)
            self.assertIn("instruction", record["columns"])

    def test_explicit_export_selects_only_the_new_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mindstudio_profiler_output"
            output.mkdir()
            automatic = output / "msprof_automatic.json"
            automatic.write_text("[]", encoding="utf-8")
            before = TIMELINE.exported_traces(Path(directory))
            explicit = output / "msprof_explicit.json"
            explicit.write_text("[]", encoding="utf-8")
            self.assertEqual(
                TIMELINE.new_exported_trace(Path(directory), before), explicit.resolve()
            )
            extra = output / "msprof_extra.json"
            extra.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed=2"):
                TIMELINE.new_exported_trace(Path(directory), before)

    def test_explicit_export_detects_same_path_change_and_rejects_appended_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "mindstudio_profiler_output"
            output.mkdir()
            trace = output / "msprof_same_second.json"
            trace.write_text("[]", encoding="utf-8")
            before = TIMELINE.exported_traces(root)

            trace.write_text('[{"name":"target"}]', encoding="utf-8")
            self.assertEqual(TIMELINE.new_exported_trace(root, before), trace.resolve())
            self.assertEqual(TIMELINE.load_exported_trace(trace), [{"name": "target"}])

            before_append = TIMELINE.exported_traces(root)
            with trace.open("a", encoding="utf-8") as handle:
                handle.write("[]")
            self.assertEqual(
                TIMELINE.new_exported_trace(root, before_append), trace.resolve()
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one exported timeline JSON"):
                TIMELINE.load_exported_trace(trace)

    def test_explicit_export_rejects_multiple_new_or_changed_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "mindstudio_profiler_output"
            output.mkdir()
            changed = output / "msprof_changed.json"
            changed.write_text("[]", encoding="utf-8")
            before = TIMELINE.exported_traces(root)
            changed.write_text("[1]", encoding="utf-8")
            (output / "msprof_new.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed=2"):
                TIMELINE.new_exported_trace(root, before)



if __name__ == "__main__":
    unittest.main()
