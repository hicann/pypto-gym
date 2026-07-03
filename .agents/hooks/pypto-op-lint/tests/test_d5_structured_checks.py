# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def _write_module_interfaces(op_dir: Path):
    write_file(op_dir / "eval" / "module_interfaces.yaml", """
primary_inputs:
  - name: x
    shape: [N, M]
    dtype: bf16
  - name: y
    shape: [N, M]
    dtype: bf16
modules:
  - id: 1
    inputs:
      - {name: x, source: primary}
      - {name: y, source: primary}
    outputs:
      - {name: out_a}
      - {name: out_b}
final_outputs:
  - {name: out_a}
  - {name: out_b}
""")


def test_ol30_uses_supported_dtypes_from_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL30")
    assert finding.status == "PASS"


def test_ol31_checks_dynamic_axes_from_design_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL31")
    assert finding.status == "PASS"


def test_ol32_checks_tolerance_from_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL32")
    assert finding.status in {"PASS", "WARN"}


def test_ol34_recognizes_shapes_from_run_and_check(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL34")
    assert finding.status == "PASS"


def test_ol34_named_dim_fail_includes_format_hint(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
p0_shapes:
  - {B: 4, C: 64, H: 56, W: 56}
tolerance: {'atol': 0.001, 'rtol': 0.001}
---
# SPEC
"""
    write_file(op_dir / "SPEC.md", spec)
    finding = run_rule(mod, op_dir, "OL34")
    assert finding.status == "FAIL"
    assert "期望格式" in finding.message
    assert "[[1024, 128]" in finding.message


def test_ol31_focuses_on_primary_kernel_called_by_wrapper(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")

    impl = """import pypto
_N = pypto.DYNAMIC
_M = pypto.STATIC

@pypto.frontend.jit
def marker_kernel(x: pypto.Tensor([_N, _M], pypto.DT_BF16), y: pypto.Tensor([_N, _M], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(1, 1)
    y[:] = x

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL31")
    assert finding.status == "FAIL"


def test_ol31_fail_on_module_file_when_design_declares_dynamic_axes(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 impl 未声明 DYNAMIC 的违规，
    不需要等到 Stage 6 集成。
    """
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 删除 integrated impl，模拟仅有 module impl 的开发中状态
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
                        y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    demo_module1_kernel(x, y)
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)

    finding = run_rule(mod, op_dir, "OL31")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol41_detects_lint_output_pollution(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")

    polluted = (op_dir / "demo_impl.py").read_text(encoding="utf-8") + "\n[pypto-op-lint] 以下规则违规\n"
    write_file(op_dir / "demo_impl.py", polluted)

    finding = run_rule(mod, op_dir, "OL41")
    assert finding.status == "FAIL"


def test_ol41_detects_module_impl_lint_output_pollution(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    polluted = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x, y):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    demo_module1_kernel(x, y)

[pypto-op-lint] blocking_rules: OL01
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", polluted)

    finding = run_rule(mod, op_dir, "OL41")
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message


def test_ol50_fails_on_explicit_non_primary_wrapper_arg(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a[:] = x
    out_b[:] = y

def demo_wrapper(x, y, runtime_options=None):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL50")
    assert finding.status == "FAIL"
    assert "runtime_options" in finding.message
    assert "primary_inputs" in finding.message


def test_ol50_allows_debug_kwargs_without_public_abi_drift(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a[:] = x
    out_b[:] = y

def demo_wrapper(x, y, **kwargs):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL50")
    assert finding.status == "PASS"


def test_ol50_provides_guidance_for_standalone_module_suffix(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    pass

def demo_module15_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "modules" / "demo_module15_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL50")
    assert finding.status == "FAIL"
    assert "累积命名规则" in finding.message
    assert "module123" in finding.message or "module12" in finding.message
    assert "修正方式" in finding.message
    assert "standalone" in finding.message


def test_ol51_bounts_only_real_assemble_targets(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    tile_result = x
    offsets = [0, 0]
    pypto.assemble(tile_result, offsets, out_a)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "FAIL"
    assert "只检测到 1 个写回点" in finding.message


def test_ol51_does_not_treat_two_arg_assemble_offsets_as_target(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    tile_result = x
    offsets = [0, 0]
    pypto.assemble(tile_result, offsets)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "FAIL"
    assert "只检测到 0 个写回点" in finding.message


def test_ol51_bounts_tensor_method_move_writeback(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    # Source values are wrapped in compute ops so the writes are non-trivial
    # (OL51.b) while still exercising the `.move()` / `[:] =` recognition
    # path (OL51.a writeback API coverage).
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a.move(pypto.exp(x))
    out_b[:] = pypto.add(x, y)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


def test_ol51_recognizes_index_add__as_writeback(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    index = pypto.zeros([32], pypto.DT_INT32)
    pypto.index_add_(out_a, 0, index, x)
    pypto.index_add_(out_b, 0, index, y)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


def test_ol51_recognizes_scatter__as_writeback(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    index = pypto.zeros([32, 32], pypto.DT_INT64)
    pypto.scatter_(out_a, 0, index, x)
    pypto.scatter_(out_b, 0, index, y)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


def test_ol51_recognizes_axpy__as_writeback(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    pypto.axpy_(out_a, x, alpha=1.0)
    pypto.axpy_(out_b, y, alpha=1.0)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


def test_ol51_recognizes_tensor_method_index_add__as_writeback(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    index = pypto.zeros([32], pypto.DT_INT32)
    out_a.index_add_(0, index, x)
    out_b.index_add_(0, index, y)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


# ───────────────────────────────────────────────────────────────────────────
# OL51.b — 非平凡写入层 (覆盖 ds_v4 / Issue #2083 placeholder 模式)
# ───────────────────────────────────────────────────────────────────────────


def test_ol51_b_fail_on_direct_input_copy(tmp_path: Path):
    """ds_v4_placeholder 模式: `out_a[:] = x` 直接复制输入 → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a[:] = x
    out_b[:] = y

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "FAIL"
    assert "trivial" in finding.message.lower() or "OL51.b" in finding.message


def test_ol51_b_fail_on_direct_zeros_write(tmp_path: Path):
    """`out_a[:] = pypto.zeros(...)` 等未初始化值 → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a[:] = pypto.zeros([1, 128], pypto.DT_BF16)
    out_b[:] = pypto.tensor([1, 128], pypto.DT_BF16)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "FAIL"


def test_ol51_b_fail_on_indirect_zeros_via_local_var(tmp_path: Path):
    """`z = pypto.zeros(...); pypto.assemble(z, ..., out)` 间接零值经回溯 → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    z = pypto.zeros([1, 128], pypto.DT_BF16)
    pypto.assemble(z, [0, 0], out_a)
    pypto.assemble(z, [0, 0], out_b)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "FAIL"


def test_ol51_b_pass_when_local_var_subsequently_modified(tmp_path: Path):
    """Task_0527 acc 模式: subscript 修改使 acc 视为非平凡 → PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    acc_a = pypto.tensor([1, 128], pypto.DT_BF16)
    acc_b = pypto.tensor([1, 128], pypto.DT_BF16)
    acc_a[:] = pypto.exp(x)
    acc_b[:] = pypto.add(x, y)
    pypto.assemble(acc_a, [0, 0], out_a)
    pypto.assemble(acc_b, [0, 0], out_b)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


def test_ol51_b_pass_when_writes_are_real_compute(tmp_path: Path):
    """直接写 compute op 的返回值 → 非平凡 → PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
                out_b: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    out_a[:] = pypto.exp(x)
    pypto.assemble(pypto.add(x, y), [0, 0], out_b)

def demo_wrapper(x, y):
    demo_kernel(x, y, y, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL51")
    assert finding.status == "PASS"


# ───────────────────────────────────────────────────────────────────────────
# OL62 — impl 内 torch 仅限 layout/alloc/cast; 数值计算必须在 @jit 图内
# ───────────────────────────────────────────────────────────────────────────


def test_ol62_fail_on_host_torch_arithmetic(tmp_path: Path):
    """host wrapper 用 torch.matmul 做主计算 (dummy-JIT 作弊) → FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import torch, pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1, 1], pypto.DT_FP32),
                out: pypto.Tensor([1, 1], pypto.DT_FP32)):
    pypto.assemble(pypto.matmul(x, x, pypto.DT_FP32), [0, 0], out)

def demo_wrapper(x, w):
    a = torch.matmul(x.float(), w.float())   # host torch compute = cheat
    return torch.softmax(a, dim=-1)
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL62")
    assert finding.status == "FAIL"
    assert "torch" in finding.message


def test_ol62_pass_on_clean_pypto_kernel(tmp_path: Path):
    """计算全在 JIT (pypto.*); host 只做 torch layout/alloc → PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import torch, pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1, 1], pypto.DT_FP32),
                out: pypto.Tensor([1, 1], pypto.DT_FP32)):
    pypto.assemble(pypto.matmul(x, x, pypto.DT_FP32), [0, 0], out)

def demo_wrapper(x):
    out = torch.empty_like(x).reshape(1, 1).contiguous()   # layout/alloc only
    demo_kernel(x, out)
    return out.to(torch.float32)
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL62")
    assert finding.status == "PASS"


def test_ol62_ignores_main_and_symbolic_min(tmp_path: Path):
    """`__main__` 内 torch 算术与 pypto SymbolicScalar `.min()` 不误判 → PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    _write_module_interfaces(op_dir)
    impl = """import torch, pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
                out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32)):
    M = x.shape[0]
    for m in pypto.loop(M, name="m"):
        rem = (M - m).min(1)                 # SymbolicScalar.min — not torch
        pypto.assemble(pypto.exp(x), [m, 0], out)

def demo_wrapper(x):
    out = torch.empty_like(x)
    demo_kernel(x, out)
    return out

if __name__ == "__main__":
    a = torch.randn(4, 4)
    print(torch.matmul(a, a).sum())          # self-test torch — must be ignored
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL62")
    assert finding.status == "PASS"
