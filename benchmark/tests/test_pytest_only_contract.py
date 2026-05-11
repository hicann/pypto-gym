#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest-only coverage for the public benchmark test contract."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import textwrap
from pathlib import Path

import pytest

from benchmark import monitor
from benchmark.constants import BENCHMARK_AUTO_MONITOR_ENV
from benchmark.opencode_exporter import append_export_result_to_log, export_session_to_markdown
from benchmark import run_kernelbench
from benchmark.run_kernelbench import _build_cfg, _validate_pypto_repo_root, load_yaml_config
from benchmark.verifier import cheat_detector


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DOWNLOAD_SCRIPTS = [
    "benchmark/scripts/download_pypto.sh",
]
QUICK_START_SCRIPTS = [
    "benchmark/scripts/pypto_quick_start.sh",
    "benchmark/scripts/single_quick_start.sh",
]
PUBLIC_SHELL_SCRIPTS = sorted(DOWNLOAD_SCRIPTS + QUICK_START_SCRIPTS)


def _stub_pypto_repo_layout(tmp_path: Path) -> Path:
    root = tmp_path / "stub_pypto"
    root.mkdir()
    (root / ".opencode").mkdir()
    return root


def test_public_shell_scripts_have_shell_entrypoints() -> None:
    for rel_path in PUBLIC_SHELL_SCRIPTS:
        path = REPO_ROOT / rel_path
        assert path.is_file()
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert re.match(r"^#!.*\b(?:bash|sh)\b", first_line)


def test_config_loader_accepts_benchmark_local_config_path() -> None:
    cfg = load_yaml_config(Path("configs/cli_smoke.yaml"))

    assert cfg["dry_run"] is True
    assert cfg["output"]["base_dir"] == "benchmark_runs"


def test_config_loader_accepts_local_pool_yaml() -> None:
    cfg = load_yaml_config(Path("configs/local_pool.yaml"))

    assert cfg["device_mode"] == "pool"
    assert cfg["cases"] == "level1=19_ReLU"
    assert cfg["devices"] == [0]
    assert cfg["concurrency"] == 1


def test_config_loader_merges_partial_yaml_with_defaults(tmp_path) -> None:
    partial = tmp_path / "pypto.yaml"
    partial.write_text(
        textwrap.dedent(
            """\
            cases: "custom=1"
            pypto:
              timeout_sec: 60
            """
        ),
        encoding="utf-8",
    )

    cfg = load_yaml_config(partial)

    assert run_kernelbench.DEFAULT_CONFIG_PATH.name == "__default__.yaml"
    assert cfg["cases"] == "custom=1"
    assert cfg["pypto"]["timeout_sec"] == 60
    assert cfg["pypto"]["pref_round"] == 3
    assert cfg["pypto"]["agent"] == "pypto-op-orchestrator"
    assert cfg["verifier"]["mode"] == "performance"
    assert "arch" not in cfg["verifier"]
    assert cfg["monitor"]["poll_sec"] == 3
    assert cfg["device_mode"] == "normal"


def test_detect_ascend_arch_uses_torch_npu_soc_version(monkeypatch) -> None:
    class FakeNpu:
        @staticmethod
        def get_soc_version() -> str:
            return "Ascend910B4"

    class FakeTorchNpu:
        npu = FakeNpu()

    monkeypatch.setattr(
        run_kernelbench,
        "_detect_ascend_arches_from_npu_smi",
        lambda: (_ for _ in ()).throw(RuntimeError("npu-smi unavailable")),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", FakeTorchNpu)

    assert run_kernelbench.detect_ascend_arch() == "ascend910b4"


def test_parse_npu_smi_arches_detects_device_models() -> None:
    output = textwrap.dedent(
        """\
        | NPU   Name                | Health        |
        | 0     910B3               | OK            |
        | 0                         | 0000:C1:00.0  |
        | 1     910B4               | OK            |
        | 0                         | 0000:C2:00.0  |
        """
    )

    assert run_kernelbench._parse_npu_smi_arches(output) == {
        0: "ascend910b3",
        1: "ascend910b4",
    }


def test_parse_npu_smi_arches_detects_phy_id_devices() -> None:
    output = textwrap.dedent(
        """\
        | NPU    Name           | Health      |
        | Chip   Phy-ID         | Bus-Id      |
        | 0      Ascend910      | Alarm       |
        | 0      0              | 0000:9D:00.0|
        | 0      Ascend910      | Alarm       |
        | 1      1              | 0000:9F:00.0|
        | 5      Ascend910      | Alarm       |
        | 0      10             | 0000:89:00.0|
        | NPU    Chip           | Process id  |
        | 5      0              | 563946      |
        """
    )

    assert run_kernelbench._parse_npu_smi_arches(output) == {
        0: "ascend910",
        1: "ascend910",
        10: "ascend910",
    }


def test_detect_ascend_arches_maps_each_configured_device(monkeypatch) -> None:
    calls = []

    def fake_detect_from_npu_smi():
        calls.append("npu-smi")
        return {
            0: "ascend910b3",
            1: "ascend910b4",
            4: "ascend910b2",
        }

    monkeypatch.setattr(
        run_kernelbench,
        "_detect_ascend_arches_from_npu_smi",
        fake_detect_from_npu_smi,
    )

    assert run_kernelbench.detect_ascend_arches([0, 4]) == {
        0: "ascend910b3",
        4: "ascend910b2",
    }
    assert calls == ["npu-smi"]


def test_case_selector_accepts_custom_level() -> None:
    parsed = run_kernelbench.parse_cases_by_level("level1=19;custom=1:3")

    assert parsed == {
        "level1": ["19"],
        "custom": ["1", "2", "3"],
    }


def test_case_selector_requires_level_even_for_single_level() -> None:
    try:
        run_kernelbench.parse_cases_by_level("19_ReLU")
    except ValueError as exc:
        assert "level=cases" in str(exc)
    else:
        raise AssertionError("bare case selector must be rejected")


def test_builtin_kernelbench_preserves_full_upstream_case_set_and_pypto_curated_config() -> None:
    root = run_kernelbench.DEFAULT_KERNELBENCH_ROOT
    expected_counts = {
        "level1": 60,
        "level2": 35,
        "level3": 22,
        "level4": 20,
        "pto_case": 44,
    }

    for level, expected in expected_counts.items():
        cases = run_kernelbench.discover_cases(root / level)
        assert len(cases) == expected

    cfg = load_yaml_config(Path("configs/pypto.yaml"))
    assert cfg["cases"] == (
        "level1=1,3,10,18,19,24,26,33,35,36,47,51,89,94,95;"
        "level2=9,14,29,30,37,51,56,62,64,68,70,88,94,95,99;"
        "level3=43,46,50"
    )


def test_run_cfg_unifies_artifact_dirs_under_configured_root(tmp_path) -> None:
    root_dir = tmp_path / "run-root"

    cfg = _build_cfg({
        "output": {"root_dir": str(root_dir)},
        "pypto": {"repo_root": str(_stub_pypto_repo_layout(tmp_path))},
    })

    assert cfg.artifact_root_dir == root_dir
    assert cfg.run_subdir == "run-root"
    assert cfg.log_dir == root_dir / "logs"
    assert cfg.report_dir == root_dir / "report"
    assert cfg.artifact_custom_dir == root_dir / "custom"
    assert cfg.monitor_state_dir == root_dir / "state"


def test_copy_pypto_custom_artifacts_excludes_output_paths(tmp_path: Path) -> None:
    op_dir = tmp_path / "repo" / "custom" / "level1" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "Foo_impl.py").write_text("# impl\n", encoding="utf-8")
    (op_dir / "output").mkdir()
    (op_dir / "output" / "large.bin").write_text("x", encoding="utf-8")
    (op_dir / "output_20260507").mkdir()
    (op_dir / "output_20260507" / "large.bin").write_text("x", encoding="utf-8")
    (op_dir / "nested").mkdir()
    (op_dir / "nested" / "output_cache").mkdir()
    (op_dir / "nested" / "output_cache" / "large.bin").write_text("x", encoding="utf-8")
    (op_dir / "nested" / "keep.txt").write_text("keep\n", encoding="utf-8")

    copied = run_kernelbench._copy_pypto_custom_artifacts(
        op_dir,
        tmp_path / "run" / "custom" / "level1" / "Foo",
    )

    assert copied == tmp_path / "run" / "custom" / "level1" / "Foo"
    assert (copied / "Foo_impl.py").is_file()
    assert (copied / "nested" / "keep.txt").is_file()
    assert not (copied / "output").exists()
    assert not (copied / "output_20260507").exists()
    assert not (copied / "nested" / "output_cache").exists()


def test_run_cfg_defaults_pypto_repo_to_benchmark_cache(monkeypatch) -> None:
    monkeypatch.setattr(run_kernelbench, "_validate_pypto_repo_root", lambda *_a, **_k: None)

    cfg = _build_cfg({})

    assert cfg.pypto_repo_root == run_kernelbench.DEFAULT_PYPTO_REPO_ROOT.resolve()
    assert cfg.pypto_pref_round == 3


def test_build_cfg_accepts_pypto_pref_round(tmp_path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    cfg = _build_cfg({
        "output": {"root_dir": str(tmp_path / "run")},
        "pypto": {"repo_root": str(stub), "pref_round": 6},
    })

    assert cfg.pypto_pref_round == 6


def test_build_cfg_rejects_negative_pypto_pref_round(tmp_path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    with pytest.raises(SystemExit) as ei:
        _build_cfg({
            "output": {"root_dir": str(tmp_path / "run")},
            "pypto": {"repo_root": str(stub), "pref_round": -1},
        })

    assert "pref_round" in str(ei.value)


def test_run_cfg_uses_base_dir_with_random_session(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(run_kernelbench.BENCHMARK_ARTIFACT_ROOT_ENV, raising=False)
    base_dir = tmp_path / "runs"

    cfg = _build_cfg({
        "output": {"base_dir": str(base_dir)},
        "pypto": {"repo_root": str(_stub_pypto_repo_layout(tmp_path))},
    })

    assert re.fullmatch(r"Task_[0-9a-f]{32}", cfg.run_subdir)
    assert cfg.artifact_root_dir == base_dir / cfg.run_subdir
    assert cfg.log_dir.parent == cfg.artifact_root_dir
    assert cfg.report_dir.parent == cfg.artifact_root_dir
    assert cfg.artifact_custom_dir.parent == cfg.artifact_root_dir
    assert cfg.monitor_state_dir.parent == cfg.artifact_root_dir


def test_validate_pypto_repo_root_default_missing_prompts_download(tmp_path: Path) -> None:
    missing = tmp_path / "default_missing"
    with pytest.raises(SystemExit) as ei:
        _validate_pypto_repo_root(missing, is_default=True)
    assert "download_pypto.sh" in str(ei.value)


def test_validate_pypto_repo_root_custom_missing_plain_message(tmp_path: Path) -> None:
    missing = tmp_path / "custom_missing"
    with pytest.raises(SystemExit) as ei:
        _validate_pypto_repo_root(missing, is_default=False)
    msg = str(ei.value)
    assert str(missing) in msg or "不存在" in msg
    assert "download_pypto.sh" not in msg


def test_validate_pypto_repo_root_custom_no_opencode_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus_tree"
    bogus.mkdir()
    with pytest.raises(SystemExit) as ei:
        _validate_pypto_repo_root(bogus, is_default=False)
    assert ".opencode" in str(ei.value)


def test_validate_pypto_repo_root_passes_stub_layout(tmp_path: Path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    _validate_pypto_repo_root(stub, is_default=False)


def test_build_cfg_device_mode_defaults_to_normal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_kernelbench, "_validate_pypto_repo_root", lambda *_a, **_k: None)

    cfg = _build_cfg({})

    assert cfg.device_mode == "normal"


def test_build_cfg_accepts_pool_device_mode_normalized(tmp_path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    cfg = _build_cfg({
        "device_mode": "PoOl",
        "output": {"root_dir": str(tmp_path / "run")},
        "pypto": {"repo_root": str(stub)},
    })

    assert cfg.device_mode == "pool"


def test_build_cfg_device_mode_null_is_normal(tmp_path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    cfg = _build_cfg({
        "device_mode": None,
        "output": {"root_dir": str(tmp_path / "run")},
        "pypto": {"repo_root": str(stub)},
    })

    assert cfg.device_mode == "normal"


def test_build_cfg_rejects_invalid_device_mode(tmp_path) -> None:
    stub = _stub_pypto_repo_layout(tmp_path)
    with pytest.raises(SystemExit) as ei:
        _build_cfg({
            "device_mode": "shared",
            "output": {"root_dir": str(tmp_path / "run")},
            "pypto": {"repo_root": str(stub)},
        })

    msg = str(ei.value)
    assert "device_mode" in msg
    assert "shared" in msg


def test_build_cfg_rejects_custom_repo_without_dot_opencode(tmp_path: Path) -> None:
    bad = tmp_path / "partial"
    bad.mkdir()
    with pytest.raises(SystemExit):
        _build_cfg({"output": {}, "pypto": {"repo_root": str(bad)}})


def test_benchmark_fixture_assets_live_under_tests() -> None:
    fixture_relpath = FIXTURES_DIR.relative_to(REPO_ROOT)

    assert FIXTURES_DIR.is_dir()
    assert fixture_relpath.parts[:3] == ("benchmark", "tests", "fixtures")


def test_unified_cli_smoke_uses_expected_contract(tmp_path) -> None:
    state_dir = tmp_path / "monitor_state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps({
            "main_status": "已完成",
            "main_exit_code": 0,
            "started_at": "2026-04-25 04:02:35",
            "updated_at": "2026-04-25 04:04:20",
            "operators": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    commands = [
        [
            sys.executable,
            "-m",
            "benchmark",
            "run",
            "--config",
            "configs/cli_smoke.yaml",
        ],
        [
            sys.executable,
            "-m",
            "benchmark",
            "monitor",
            str(state_dir),
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr


def test_resolve_auto_monitor_env_overrides_cli(monkeypatch) -> None:
    from benchmark import __main__ as benchmark_cli

    monkeypatch.delenv(BENCHMARK_AUTO_MONITOR_ENV, raising=False)
    assert benchmark_cli._resolve_auto_monitor(False) is True
    assert benchmark_cli._resolve_auto_monitor(True) is False

    monkeypatch.setenv(BENCHMARK_AUTO_MONITOR_ENV, "1")
    assert benchmark_cli._resolve_auto_monitor(True) is True
    assert benchmark_cli._resolve_auto_monitor(False) is True

    monkeypatch.setenv(BENCHMARK_AUTO_MONITOR_ENV, "0")
    assert benchmark_cli._resolve_auto_monitor(True) is False
    assert benchmark_cli._resolve_auto_monitor(False) is False


def test_handle_run_no_auto_monitor_skips_tui_and_prints_command(
    monkeypatch, tmp_path, capsys,
) -> None:
    from benchmark import __main__ as benchmark_cli

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    monitor_calls: list[Path] = []

    def fake_monitor_main(sd: Path) -> int:
        monitor_calls.append(sd)
        return 0

    monkeypatch.setattr(benchmark_cli.run_kernelbench, "pre_resolve_state_dir", lambda _p: state_dir)
    monkeypatch.setattr(benchmark_cli.run_kernelbench, "preflight_background_run", lambda _p: None)
    monkeypatch.setattr(benchmark_cli.os, "fork", lambda: 12345)
    monkeypatch.setattr(benchmark_cli, "_wait_for_state_json", lambda *a, **k: True)
    monkeypatch.setattr(benchmark_cli.monitor, "main", fake_monitor_main)

    rc = benchmark_cli._handle_run(
        REPO_ROOT / "benchmark/configs/cli_smoke.yaml",
        foreground=False,
        auto_monitor=False,
    )
    assert rc == 0
    assert monitor_calls == []

    out = capsys.readouterr().out
    assert "monitor_state_dir:" in out
    assert "monitor_command:" in out
    assert str(state_dir) in out
    assert "pid=12345" in out


def test_unified_cli_monitor_requires_state_dir() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark",
            "monitor",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode != 0


def test_run_from_config_handles_keyboard_interrupt_cleanly(monkeypatch, tmp_path) -> None:
    bench_dir = tmp_path / "KernelBench"
    level_dir = bench_dir / "level1"
    level_dir.mkdir(parents=True)
    (level_dir / "1_Foo.py").write_text("class Model: pass\n", encoding="utf-8")
    stub_repo = _stub_pypto_repo_layout(tmp_path)

    async def fake_run_batch(case_paths, devices, concurrency, cfg):
        raise KeyboardInterrupt()

    registered_signals = {}
    monitor_worker = object()
    finished = {}
    monkeypatch.setattr(
        run_kernelbench,
        "load_yaml_config",
        lambda _path: {
            "bench_dir": str(bench_dir),
            "cases": "level1=1",
            "devices": [0],
            "concurrency": 1,
            "output": {"root_dir": str(tmp_path / "run")},
            "pypto": {"repo_root": str(stub_repo)},
        },
    )
    monkeypatch.setattr(run_kernelbench, "run_batch", fake_run_batch)
    monkeypatch.setattr(run_kernelbench, "detect_ascend_arches", lambda device_ids: {
        device_id: "ascend910b4" for device_id in device_ids
    })
    monkeypatch.setattr(
        run_kernelbench.signal,
        "signal",
        lambda signum, handler: registered_signals.setdefault(signum, handler),
    )
    monkeypatch.setattr(
        run_kernelbench,
        "_start_monitor_state_worker",
        lambda case_paths, cfg: monitor_worker,
    )
    monkeypatch.setattr(
        run_kernelbench,
        "_finish_monitor_state_worker",
        lambda worker, cfg, exit_code: finished.update(worker=worker, exit_code=exit_code),
    )

    assert run_kernelbench.run_from_config(tmp_path / "config.yaml") == 130
    assert registered_signals[signal.SIGINT] is run_kernelbench._signal_handler
    assert finished == {"worker": monitor_worker, "exit_code": 130}


def test_run_batch_waits_for_slot_before_starting_case(monkeypatch, tmp_path) -> None:
    case_a = tmp_path / "level1" / "1_A.py"
    case_b = tmp_path / "level1" / "2_B.py"
    case_a.parent.mkdir(parents=True)
    case_a.write_text("# a\n", encoding="utf-8")
    case_b.write_text("# b\n", encoding="utf-8")
    cfg = _build_cfg({
        "output": {"root_dir": str(tmp_path / "run")},
        "pypto": {"repo_root": str(_stub_pypto_repo_layout(tmp_path))},
    })
    started = []

    async def scenario() -> None:
        release_first = asyncio.Event()

        async def fake_run_one_case(case_path, device_id, cfg, semaphore, *, device_pool=None):
            started.append(case_path.stem)
            if case_path == case_a:
                await release_first.wait()
            return run_kernelbench.CaseRunRecord(
                op_name=case_path.stem,
                case_id=case_path.stem,
                source_file=str(case_path),
                overall_status="success",
            )

        monkeypatch.setattr(run_kernelbench, "run_one_case", fake_run_one_case)
        task = asyncio.create_task(run_kernelbench.run_batch([case_a, case_b], [0], 1, cfg))
        for _ in range(10):
            await asyncio.sleep(0)
            if started:
                break
        assert started == ["1_A"]

        release_first.set()
        await task
        assert started == ["1_A", "2_B"]

    asyncio.run(scenario())


def test_signal_handler_sets_stop_event_before_raising() -> None:
    old_stop_event = run_kernelbench._stop_event
    stop_event = threading.Event()
    run_kernelbench._stop_event = stop_event
    try:
        raised = False
        try:
            run_kernelbench._signal_handler(signal.SIGINT, None)
        except KeyboardInterrupt:
            raised = True
        assert raised
        assert stop_event.is_set()
    finally:
        run_kernelbench._stop_event = old_stop_event


def test_run_from_config_cleans_registered_process_groups_on_exit(monkeypatch, tmp_path) -> None:
    cleaned = {"called": False}
    stop_event_seen = {"set": False}

    monkeypatch.setattr(
        run_kernelbench,
        "load_yaml_config",
        lambda _path: {
            "dry_run": True,
            "output": {"root_dir": str(tmp_path / "run")},
            "pypto": {"repo_root": str(_stub_pypto_repo_layout(tmp_path))},
        },
    )
    monkeypatch.setattr(
        run_kernelbench,
        "cleanup_registered_process_groups",
        lambda: cleaned.update(called=True),
    )

    assert run_kernelbench.run_from_config(tmp_path / "config.yaml") == 0
    assert cleaned["called"]
    assert run_kernelbench._stop_event is not None
    stop_event_seen["set"] = run_kernelbench._stop_event.is_set()
    assert stop_event_seen["set"]


def test_interruptible_runner_drains_keyboard_interrupt_task(caplog) -> None:
    async def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        raised = False
        try:
            run_kernelbench._run_interruptible(raise_keyboard_interrupt())
        except KeyboardInterrupt:
            raised = True

    assert raised
    assert "Task exception was never retrieved" not in caplog.text


def test_cheat_detector_fixture_verdicts_are_pytest_covered() -> None:
    expected = {
        "clean": "pass",
        "no_pypto": "cheat",
        "no_jit": "cheat",
        "suspicious": "suspicious",
    }

    for case, verdict in expected.items():
        report = cheat_detector.detect_cheats(FIXTURES_DIR / case, "relu")
        assert report.verdict == verdict, report.to_dict()


def test_opencode_exporter_handles_invalid_utf8_and_sqlite(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    fake_opencode = tmp_path / "fake_opencode.py"
    fake_opencode.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys

            if sys.argv[1:3] == ["export", "ses_invalidutf8"]:
                payload = (
                    b'{"info":{"id":"ses_invalidutf8","title":"invalid utf8"},'
                    b'"messages":[{"info":{"role":"assistant","time":{}},'
                    b'"parts":[{"type":"text","text":"bad byte: \\xe2 end"}]}]}'
                )
                sys.stdout.buffer.write(payload)
                raise SystemExit(0)

            if sys.argv[1:3] == ["export", "ses_truncatedutf8"]:
                payload = (
                    b'{"info":{"id":"ses_truncatedutf8","title":"truncated utf8"},'
                    b'"messages":[{"info":{"role":"assistant","time":{}},'
                    b'"parts":[{"type":"text","text":"cut byte: \\xe2'
                )
                sys.stdout.buffer.write(payload)
                raise SystemExit(0)

            raise SystemExit(2)
            """
        ),
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)

    exported = export_session_to_markdown(
        session_id="ses_invalidutf8",
        output_file=tmp_path / "invalid_utf8_session.md",
        opencode_bin=str(fake_opencode),
        cwd=REPO_ROOT,
    )
    assert exported.ok, exported.to_dict()
    assert "bad byte:" in exported.markdown_file.read_text(encoding="utf-8")
    append_export_result_to_log(tmp_path / "verifier.log", exported, label="verifier")
    assert (tmp_path / "verifier_session_export.json").exists()

    truncated = export_session_to_markdown(
        session_id="ses_truncatedutf8",
        output_file=tmp_path / "truncated_utf8_session.md",
        opencode_bin=str(fake_opencode),
        cwd=REPO_ROOT,
    )
    assert truncated.status == "error"
    assert "JSON" in truncated.message
    append_export_result_to_log(tmp_path / "pypto_run.log", truncated, label="pypto")
    assert "JSON" in (tmp_path / "pypto_session_export.log").read_text(encoding="utf-8")

    db_path = tmp_path / "opencode.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table session ("
            "id text primary key, title text, directory text, version text, "
            "time_created integer, time_updated integer)"
        )
        conn.execute(
            "create table message ("
            "id text primary key, session_id text, time_created integer, "
            "time_updated integer, data text)"
        )
        conn.execute(
            "create table part ("
            "id text primary key, message_id text, session_id text, "
            "time_created integer, time_updated integer, data text)"
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?)",
            ("ses_sqlitestorage", "sqlite transcript", str(REPO_ROOT), "test", 1, 2),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_sqlite",
                "ses_sqlitestorage",
                1,
                2,
                json.dumps({"role": "assistant", "time": {"created": 1, "completed": 2}}),
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_sqlite",
                "msg_sqlite",
                "ses_sqlitestorage",
                1,
                2,
                json.dumps({"type": "text", "text": "sqlite source"}),
            ),
        )

    monkeypatch.setenv("OPENCODE_DB", str(db_path))
    sqlite_result = export_session_to_markdown(
        session_id="ses_sqlitestorage",
        output_file=tmp_path / "sqlite_session.md",
        opencode_bin="/missing/opencode",
        cwd=REPO_ROOT,
    )
    assert sqlite_result.ok, sqlite_result.to_dict()
    assert "sqlite storage" in sqlite_result.message
    assert "sqlite source" in sqlite_result.markdown_file.read_text(encoding="utf-8")


def test_opencode_exporter_includes_sqlite_child_sessions(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "opencode.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table session ("
            "id text primary key, project_id text, parent_id text, title text, "
            "directory text, version text, time_created integer, time_updated integer)"
        )
        conn.execute(
            "create table message ("
            "id text primary key, session_id text, time_created integer, "
            "time_updated integer, data text)"
        )
        conn.execute(
            "create table part ("
            "id text primary key, message_id text, session_id text, "
            "time_created integer, time_updated integer, data text)"
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_parent",
                "proj",
                None,
                "parent transcript",
                str(REPO_ROOT),
                "test",
                1,
                4,
            ),
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_child",
                "proj",
                "ses_parent",
                "child transcript (@explore subagent)",
                str(REPO_ROOT),
                "test",
                2,
                5,
            ),
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_child_late",
                "proj",
                "ses_parent",
                "later child transcript (@worker subagent)",
                str(REPO_ROOT),
                "test",
                3,
                7,
            ),
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ses_grandchild_latest",
                "proj",
                "ses_child_late",
                "nested child transcript (@debugger subagent)",
                str(REPO_ROOT),
                "test",
                4,
                9,
            ),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_parent",
                "ses_parent",
                1,
                1,
                json.dumps({"role": "assistant", "time": {"created": 1, "completed": 1}}),
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_parent",
                "msg_parent",
                "ses_parent",
                1,
                1,
                json.dumps({"type": "text", "text": "parent-only summary"}),
            ),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_child",
                "ses_child",
                2,
                3,
                json.dumps({"role": "assistant", "time": {"created": 2, "completed": 3}}),
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_child",
                "msg_child",
                "ses_child",
                2,
                3,
                json.dumps({
                    "type": "tool",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "output": "child raw tool transcript marker",
                    },
                }),
            ),
        )

    monkeypatch.setenv("OPENCODE_DB", str(db_path))
    result = export_session_to_markdown(
        session_id="ses_parent",
        output_file=tmp_path / "parent_with_child.md",
        opencode_bin="/missing/opencode",
        cwd=REPO_ROOT,
    )

    assert result.ok, result.to_dict()
    text = result.markdown_file.read_text(encoding="utf-8")
    assert "# Subagent Sessions" in text
    assert "ses_child" in text
    assert "ses_child_late" in text
    assert "ses_grandchild_latest" in text
    assert "child raw tool transcript marker" in text
    assert result.session_updated_at_ms == 4
    assert result.tree_updated_at_ms == 9

    split_result = export_session_to_markdown(
        session_id="ses_parent",
        output_file=tmp_path / "unused_when_output_dir_is_set.md",
        output_dir=tmp_path / "split_export",
        opencode_bin="/missing/opencode",
        cwd=REPO_ROOT,
    )
    assert split_result.ok, split_result.to_dict()
    assert split_result.markdown_file == (tmp_path / "split_export" / "root_full.md").resolve()
    assert split_result.json_file == (tmp_path / "split_export" / "root_full.json").resolve()
    assert split_result.session_tree_file == (tmp_path / "split_export" / "session_tree.tsv").resolve()
    assert split_result.nodes_dir == (tmp_path / "split_export" / "nodes").resolve()
    assert split_result.node_session_count == 4
    assert split_result.session_tree_file.exists()
    node_files = sorted(split_result.nodes_dir.glob("*.md"))
    assert len(node_files) == 4
    assert any("ses_grandchild_latest" in path.name for path in node_files)


def test_monitor_state_dir_and_dashboard_width(tmp_path) -> None:
    state_dir = tmp_path / "monitor_state"
    monitor.configure_state_dir(state_dir)
    monitor.write_state({
        "main_status": "已完成",
        "started_at": "—",
        "updated_at": "—",
        "operators": [],
    })
    assert (state_dir / "state.json").exists()

    state = {
        "main_pid": 734465,
        "main_status": "正在进行",
        "main_exit_code": None,
        "started_at": "2026-04-25 04:02:35",
        "updated_at": "2026-04-25 04:04:20",
        "operators": [
            {
                "op_name": "Argmax_over_a_dimension",
                "phase": "verifier",
                "started_at": "2026-04-25T04:02:44",
                "ended_at": None,
                "duration_sec": 0.0,
                "dev_status": "Verifier验证中",
                "phases": {
                    "pypto": {
                        "pid": None,
                        "started_at": "2026-04-25T04:02:44",
                        "ended_at": "2026-04-25T04:03:44",
                        "duration_sec": 60.0,
                    },
                    "verifier": {
                        "pid": None,
                        "started_at": "2026-04-25T04:03:44",
                        "duration_sec": 36.0,
                        "status": "running",
                    },
                },
            },
            {
                "op_name": "CosineSimilarityLoss",
                "phase": "prepare",
                "started_at": "2026-04-25T04:03:42",
                "ended_at": None,
                "duration_sec": 0.0,
                "dev_status": "准备中",
                "phases": {"pypto": {"pid": None}, "verifier": {"pid": None}},
            },
            {
                "op_name": "TripletMarginLoss",
                "phase": "pypto",
                "started_at": "2026-04-25T04:03:26",
                "ended_at": None,
                "duration_sec": 88.8,
                "dev_status": "PyPTO生成中",
                "phases": {
                    "pypto": {
                        "pid": 123456,
                        "started_at": "2026-04-25T04:03:26",
                        "duration_sec": 88.8,
                        "status": "running",
                    },
                    "verifier": {"pid": None},
                },
            },
        ],
    }
    dashboard = monitor._render_dashboard(state)
    assert "PyPTO PID" not in dashboard
    assert "Verify PID" not in dashboard
    assert "PyPTO Start" in dashboard
    assert "Verify Elap/End" in dashboard
    table_lines = [
        line for line in dashboard.splitlines()
        if line.startswith("  ") and (
            "Arg" in line or "Cosine" in line or "Triplet" in line or "Operator" in line
        )
    ]
    widths = {monitor.wcswidth(line) for line in table_lines}
    assert len(widths) == 1, "\n".join(table_lines)
