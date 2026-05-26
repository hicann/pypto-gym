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
"""verify / profile 脚本字符串生成器.

桥接层只有 ascend+torch+pypto 一种组合, 直接以 f-string 拼接生成 3 种脚本,
全部在 ``verify_dir`` 内由 LocalWorker 子进程执行:
    - ``verify_<op>.py``                — 精度对比 (framework Model vs ModelNew),
                                          stdout 输出 ``VERIFY_PASSED`` / ``VERIFY_FAILED``.
    - ``profile_<op>_base.py``          — base 性能 (PyTorch 原生 Model 在 NPU 上 device-time 计时),
                                          stdout 输出 ``BASE_TIME_US: <num>``.
    - ``profile_<op>_generation.py``    — generation 性能 (PyPTO ModelNew 走 swimlane),
                                          stdout 输出 ``PROFILE_RESULT_GEN_US: <num>``.

辅助源文件 (由 ``KernelVerifier.gen_verify_project`` 写入同目录, 此处不重复生成):
    - ``<op>_torch.py``                 — 原始 KernelBench task_desc, 含 ``Model`` /
                                          ``get_inputs`` / ``get_init_inputs``.
    - ``<op>_pypto_impl.py``            — PyPTO 产物, 含 ``ModelNew`` 或 wrapper.
"""

from __future__ import annotations

from . import pypto_adapter


_FRAMEWORK_LOADER_TEMPLATE = '''\
import importlib.util as _ilu
_fw_spec = _ilu.spec_from_file_location("{framework_module_name}", os.path.join(os.path.dirname(__file__), "{framework_filename}"))
framework_module = _ilu.module_from_spec(_fw_spec)
_fw_spec.loader.exec_module(framework_module)
'''


_NPU_DEVICE_SETUP = '''\
import torch
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

torch.manual_seed(42)
if _HAS_NPU:
    torch.npu.set_device({device_id})
    torch.npu.manual_seed_all(42)
    device = torch.device("npu:{device_id}")
else:
    device = torch.device("cpu")
    print("[WARN] torch_npu unavailable, falling back to CPU; results meaningless on real bench")
'''


_NPU_SYNC_SAFE = '''\
def _npu_sync():
    if _HAS_NPU:
        try:
            torch.npu.synchronize()
        except Exception:
            pass
'''


_INPUT_TO_DEVICE = '''\
def _to_device(xs, dev):
    if isinstance(xs, (list, tuple)):
        return type(xs)(_to_device(x, dev) for x in xs)
    if isinstance(xs, torch.Tensor):
        return xs.to(dev)
    return xs
'''


_OUTPUT_FLATTEN_AND_COMPARE = '''\
def _flatten(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(_flatten(e))
        return out
    if isinstance(x, dict):
        out = []
        for k in sorted(x.keys()):
            out.extend(_flatten(x[k]))
        return out
    return [x]


def _compare_outputs(ref_out, impl_out, rtol, atol):
    ref_flat = _flatten(ref_out)
    out_flat = _flatten(impl_out)
    if len(ref_flat) != len(out_flat):
        return False, [f"output count mismatch ref={len(ref_flat)}, impl={len(out_flat)}"]
    msgs = []
    for i, (r, o) in enumerate(zip(ref_flat, out_flat)):
        try:
            if isinstance(r, torch.Tensor):
                r_cpu = r.detach().to("cpu").float()
                if isinstance(o, torch.Tensor):
                    o_cpu = o.detach().to("cpu").float()
                else:
                    o_cpu = torch.as_tensor(o).float()
                torch.testing.assert_close(o_cpu, r_cpu, rtol=rtol, atol=atol)
            else:
                if r != o:
                    raise AssertionError(f"non-tensor mismatch: {r!r} vs {o!r}")
        except Exception as e:
            msgs.append(f"[output #{i}] {e}")
    return (len(msgs) == 0), msgs
'''


_WEIGHT_SYNC = '''\
def _sync_weights(src_model, dst_model):
    """将 src_model 的权重复制到 dst_model，确保 verifier 比较时使用相同参数."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    src_keys = list(src_state.keys())
    dst_keys = list(dst_state.keys())
    missing = [key for key in src_keys if key not in dst_state]
    unexpected = [key for key in dst_keys if key not in src_state]
    if src_keys != dst_keys:
        print("VERIFY_FAILED:")
        print(
            "[state_dict] ModelNew state_dict keys must match task_desc.Model; "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected={src_keys}, actual={dst_keys}"
        )
        sys.exit(1)
    for key in src_keys:
        if tuple(src_state[key].shape) != tuple(dst_state[key].shape):
            print("VERIFY_FAILED:")
            print(
                f"[state_dict] shape mismatch for {key}: "
                f"expected {tuple(src_state[key].shape)}, got {tuple(dst_state[key].shape)}"
            )
            sys.exit(1)
        if src_state[key].dtype != dst_state[key].dtype:
            print("VERIFY_FAILED:")
            print(
                f"[state_dict] dtype mismatch for {key}: "
                f"expected {src_state[key].dtype}, got {dst_state[key].dtype}"
            )
            sys.exit(1)
    dst_model.load_state_dict(src_state, strict=True)
'''


def _framework_loader(op_name: str, framework_filename: str) -> str:
    safe_module = "framework_" + op_name.replace("-", "_")
    return _FRAMEWORK_LOADER_TEMPLATE.format(
        framework_module_name=safe_module,
        framework_filename=framework_filename,
    )


def build_verify_script(
    op_name: str,
    framework_filename: str,
    device_id: int,
    pypto_run_mode: int = 0,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> str:
    """生成 ``verify_<op>.py`` 内容."""
    pypto_imports = pypto_adapter.get_pypto_imports()
    runtime_overrides = pypto_adapter.get_runtime_env_overrides(
        pypto_run_mode=pypto_run_mode,
        pypto_runtime_debug_mode=0,  # verify 路径不开 swimlane.
    )
    fw_loader = _framework_loader(op_name, framework_filename)
    modelnew_loader = pypto_adapter.get_modelnew_loader(op_name)
    npu_setup = _NPU_DEVICE_SETUP.format(device_id=device_id)

    return f'''\
#!/usr/bin/env python3
# Auto-generated by benchmark.verifier.script_builder
# verify {op_name} (framework Model vs ModelNew, NPU device={device_id})
import sys
import os
os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"

{pypto_imports}
{runtime_overrides}
{npu_setup}
{_NPU_SYNC_SAFE}
{_INPUT_TO_DEVICE}
{_OUTPUT_FLATTEN_AND_COMPARE}
{_WEIGHT_SYNC}

# === load reference Model from {framework_filename} ===
{fw_loader}

init_inputs = framework_module.get_init_inputs()
raw_inputs = framework_module.get_inputs()
inputs = _to_device(raw_inputs, device)
framework_model = framework_module.Model(*init_inputs).to(device)

# === load PyPTO ModelNew (must be after device setup; jit not patched in verify path) ===
{modelnew_loader}
impl_model = ModelNew(*init_inputs).to(device)

# === sync weights to ensure fair comparison ===
_sync_weights(framework_model, impl_model)

# === run both ===
with torch.no_grad():
    framework_output = framework_model(*inputs)
    _npu_sync()
    impl_output = impl_model(*inputs)
    _npu_sync()

# === compare ===
ok, mismatches = _compare_outputs(framework_output, impl_output, rtol={rtol}, atol={atol})
if not ok:
    print("VERIFY_FAILED:")
    for m in mismatches:
        print(m)
    sys.exit(1)

print("VERIFY_PASSED")
sys.exit(0)
'''


def build_profile_base_script(
    op_name: str,
    framework_filename: str,
    device_id: int,
    warmup_times: int,
    run_times: int,
) -> str:
    """生成 ``profile_<op>_base.py`` — PyTorch 原版在 NPU 上的 device-time 计时.

    对 Ascend/NPU 强制使用 ``torch_npu.profiler`` 读取设备侧 kernel 时间.
    如果 profiler 不可用或无法解析结果, 直接返回 ``inf`` 并由上层判失败.
    """
    fw_loader = _framework_loader(op_name, framework_filename)
    npu_setup = _NPU_DEVICE_SETUP.format(device_id=device_id)

    return f'''\
#!/usr/bin/env python3
# Auto-generated by benchmark.verifier.script_builder
# profile base for {op_name} (PyTorch Model on NPU, device={device_id})
import sys
import os
import time
import glob
import shutil
os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"

{npu_setup}
{_NPU_SYNC_SAFE}
{_INPUT_TO_DEVICE}

# === load reference Model from {framework_filename} ===
{fw_loader}

init_inputs = framework_module.get_init_inputs()
raw_inputs = framework_module.get_inputs()
inputs = _to_device(raw_inputs, device)
framework_model = framework_module.Model(*init_inputs).to(device)

def _collect_npu_profile_time_us(profile_path, warmup_times, run_times):
    csv_files = sorted(glob.glob(os.path.join(profile_path, "**", "op_summary_*.csv"), recursive=True))
    if not csv_files:
        return float("inf"), "missing_op_summary_csv"

    try:
        import csv
        rows = []
        with open(csv_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return float("inf"), f"csv_read_failed: {{e}}"

    if not rows:
        return float("inf"), "empty_op_summary_csv"

    name_key = None
    dur_key = None
    sample = rows[0]
    for k in sample:
        lk = k.strip().lower()
        if lk in ("op name", "op_name"):
            name_key = k
        if lk in ("task duration(us)", "task_duration(us)", "task duration (us)", "avg time(us)", "avg_time_us"):
            dur_key = k
    if dur_key is None:
        for k in sample:
            if "duration" in k.strip().lower() and "us" in k.strip().lower():
                dur_key = k
                break
    if dur_key is None:
        return float("inf"), f"duration_column_not_found: {{list(sample.keys())}}"

    filtered = rows
    if name_key is not None:
        filtered = [
            r for r in rows
            if "aclnnIsClose_IsCloseAiCpu_IsClose" not in str(r.get(name_key, ""))
            and "aclnnAll_ReduceAll_ReduceAll" not in str(r.get(name_key, ""))
        ]

    durations = []
    for r in filtered:
        try:
            durations.append(float(r.get(dur_key, "") or 0))
        except Exception:
            continue

    if not durations:
        return float("inf"), "no_valid_duration_rows"

    if warmup_times > 0 and len(durations) > warmup_times:
        durations = durations[warmup_times:]

    if run_times > 0 and len(durations) > run_times:
        durations = durations[:run_times]

    if not durations:
        return float("inf"), "durations_empty_after_trim"

    return min(durations), ""


def _profile_base_device_time_us(fn, warmup_times, run_times):
    if not _HAS_NPU:
        return float("inf"), "torch_npu_unavailable"
    if "torch_npu" not in sys.modules:
        return float("inf"), "torch_npu_module_missing"
    try:
        import torch_npu
    except Exception as e:
        return float("inf"), f"torch_npu_import_failed: {{e}}"

    try:
        exp_cfg = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )
    except Exception as e:
        return float("inf"), f"experimental_config_failed: {{e}}"

    fn()
    _npu_sync()

    timestamp = int(time.time() * 1000)
    profile_path = os.path.abspath(f"prof_base_output_{{timestamp}}")
    skip_first = 1 + warmup_times
    wait = 0
    warmup_prof = 0
    active = max(run_times, 1)
    repeat = 1
    total = skip_first + (wait + warmup_prof + active) * repeat

    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=wait, warmup=warmup_prof, active=active, repeat=repeat, skip_first=skip_first
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=exp_cfg,
        ) as prof:
            for _ in range(total):
                fn()
                prof.step()
                _npu_sync()

        base_time_us, reason = _collect_npu_profile_time_us(profile_path, warmup_times, run_times)
        if base_time_us == float("inf"):
            return float("inf"), (reason or "profile_result_invalid")
        return base_time_us, ""
    except Exception as e:
        return float("inf"), f"profile_failed: {{e}}"
    finally:
        try:
            if os.path.exists(profile_path):
                shutil.rmtree(profile_path)
        except Exception:
            pass


with torch.no_grad():
    def _base_benchmark_fn():
        return framework_model(*inputs)

    base_time_us, device_profile_reason = _profile_base_device_time_us(
        _base_benchmark_fn, {warmup_times}, {run_times}
    )

    if base_time_us == float("inf"):
        print(f"[ERROR] base device-time profile failed: {{device_profile_reason}}")
    else:
        print(f"[INFO] base device-time profile (us): min={{base_time_us:.2f}}")

print(f"BASE_TIME_US: {{base_time_us:.6f}}")
sys.exit(0)
'''


def build_profile_generation_script(
    op_name: str,
    framework_filename: str,
    device_id: int,
    pypto_run_mode: int = 0,
) -> str:
    """生成 ``profile_<op>_generation.py`` — PyPTO ModelNew 走 swimlane 路径.

    ``warmup_times`` / ``run_times`` 在此**不消费**: PyPTO autotune 会在第一次
    forward 内部自行收敛, swimlane 也是单次 forward 抓全部 jit kernel trace,
    无需重复采样.
    """
    pypto_imports = pypto_adapter.get_pypto_imports()
    runtime_overrides = pypto_adapter.get_runtime_env_overrides(
        pypto_run_mode=pypto_run_mode,
        pypto_runtime_debug_mode=1,  # 必须开 swimlane + monkey-patch jit.
    )
    fw_loader = _framework_loader(op_name, framework_filename)
    modelnew_loader = pypto_adapter.get_modelnew_loader(op_name)
    swimlane_output_setup = pypto_adapter.get_swimlane_output_setup()
    swimlane_body = pypto_adapter.get_swimlane_benchmark_body()
    npu_setup = _NPU_DEVICE_SETUP.format(device_id=device_id)

    return f'''\
#!/usr/bin/env python3
# Auto-generated by benchmark.verifier.script_builder
# profile generation for {op_name} (PyPTO ModelNew swimlane, device={device_id})
import sys
import os
os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"

{pypto_imports}
{runtime_overrides}
{npu_setup}
{_NPU_SYNC_SAFE}
{_INPUT_TO_DEVICE}

{swimlane_output_setup}

# === load Model module (only need get_inputs / get_init_inputs, not Model itself) ===
{fw_loader}

init_inputs = framework_module.get_init_inputs()
raw_inputs = framework_module.get_inputs()
inputs = _to_device(raw_inputs, device)

# === load PyPTO ModelNew AFTER jit monkey-patch above ===
{modelnew_loader}
impl_model = ModelNew(*init_inputs).to(device)

# === swimlane benchmark (provides execution_time_us + PROFILE_RESULT_GEN_US:) ===
{swimlane_body}

sys.exit(0)
'''
