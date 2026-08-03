#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro 环境 Preflight 聚合检测脚本
#
# 在 orchestrator 启动算子开发前执行一次，统一确认：
#   [1] NPU 设备层   — 复用 _npu_info.py（设备枚举 / 健康 / 可用数）
#   [2] NPU 架构层   — 复用 get_npu_arch.py（dav-* 架构探测，如 dav-3510=a5）
#   [3] 运行时层     — torch_npu / pypto_pro.language / pl.jit 是否可导入（PyPTO-Pro 专属）
#   [4] CANN 层      — ASCEND_HOME_PATH / set_env / CANN 版本（裁剪自 check_env.sh）
#
# 设计约定：
#   - 只检测、不修复。发现问题时汇总 errors/warnings 供 orchestrator 反馈用户。
#   - errors 视为阻断（exit 1）；warnings 不阻断（exit 0）。
#   - 输出人类可读摘要 + 末尾一行机器可读 JSON（PREFLIGHT_JSON: {...}）。
#
# 用法：
#   python ./env_preflight.py          # 完整检测
#   python ./env_preflight.py --json    # 仅输出 JSON
# -----------------------------------------------------------------------------

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
LOGGER = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# 输出辅助
# ─────────────────────────────────────────────────────────────

_RED = "\033[0;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_BLUE = "\033[0;34m"
_NC = "\033[0m"


class Report:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.errors = []
        self.warnings = []
        self.devices = []
        # [1] 设备层的唯一裁决：是否枚举到可用卡（供 [3]/最终裁决复用）
        self.npu_usable = False
        self.npu_arch = None
        self.torch_npu_ok = False
        self.pypto_pro_ok = False
        self.cann_version = None
        self.ascend_home_path = os.environ.get("ASCEND_HOME_PATH", "")

    def success(self, msg):
        if not self.quiet:
            LOGGER.info("%s✓ %s%s", _GREEN, msg, _NC)

    def section(self, msg):
        if not self.quiet:
            LOGGER.info("\n%s%s%s", _BLUE, msg, _NC)

    def error(self, msg):
        self.errors.append(msg)
        if not self.quiet:
            LOGGER.error("%s✗ %s%s", _RED, msg, _NC)

    def warning(self, msg):
        self.warnings.append(msg)
        if not self.quiet:
            LOGGER.warning("%s⚠ %s%s", _YELLOW, msg, _NC)

    def to_dict(self):
        return {
            "passed": not self.errors,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "devices": self.devices,
            "npu_usable": self.npu_usable,
            "npu_arch": self.npu_arch,
            "torch_npu_ok": self.torch_npu_ok,
            "pypto_pro_ok": self.pypto_pro_ok,
            "cann_version": self.cann_version,
            "ascend_home_path": self.ascend_home_path,
        }


# ─────────────────────────────────────────────────────────────
# [1] NPU 设备层 — 分层后端
#   主：npu-smi via _npu_info.py（如 Ascend910 环境，npu-smi 正常，已实测可用）
#   回退：torch_npu（部分环境如 a5/Ascend950PR，机器自带的 npu-smi 是残缺 stub，
#         info -m / -t 只回显单行、无法枚举多卡；但设备是真实 NPU 硬件，
#         torch_npu 能正常识别 device_count/name/properties）
# ─────────────────────────────────────────────────────────────

def _torch_npu_device(torch, index):
    """Collect the available torch_npu metadata for one device."""
    try:
        name = torch.npu.get_device_name(index)
    except Exception:
        name = "unknown"

    device = {
        "npu_id": index,
        "chip_name": name,
        "health": "available",
        "health_source": "torch_npu.is_available",
    }
    try:
        properties = torch.npu.get_device_properties(index)
        property_names = (
            "total_memory", "cube_core_num", "vector_core_num",
            "L2_cache_size", "uuid",
        )
        for attr in property_names:
            value = getattr(properties, attr, None)
            if value is not None:
                device[attr] = str(value)
    except Exception as error:
        logging.debug(
            "get_device_properties failed for NPU %d: %s", index, error,
        )
    return name, device


def _check_devices_via_torch_npu(rep: Report) -> bool:
    """npu-smi 枚举为空时的回退后端。

    仅采集 preflight 真正需要的字段：设备数量 + 名称 + 可用性。
    这些 API 有据：
      - torch.npu.is_available() / device_count() —— 本仓 golden 已依赖
        (custom/relu/relu_golden.py:29)
    规格字段（total_memory 等）用 getattr 防御式采集：有则记录、
    无则跳过，不写死依赖、不因缺字段报错。

    返回 True 表示回退后端成功枚举到设备。
    """
    try:
        import torch  # noqa
        import torch_npu  # noqa
    except ImportError:
        return False
    except Exception as error:
        rep.warning(f"torch_npu 回退后端初始化失败: {error}")
        return False

    try:
        if not torch.npu.is_available():
            return False
        count = torch.npu.device_count()
    except Exception as e:
        rep.warning(f"torch_npu 设备查询异常: {e}")
        return False

    if not count or count <= 0:
        return False

    rep.npu_usable = True  # [1] 层裁决：torch_npu 后端确认有可用卡
    rep.warning("npu-smi 枚举为空（疑似 npu-smi 残缺 stub 环境，如 a5），"
                "回退 torch_npu 后端枚举设备")
    rep.success(f"检测到 {count} 个 NPU 设备 (via torch_npu): "
                f"{list(range(count))}")

    for index in range(count):
        name, device = _torch_npu_device(torch, index)
        rep.devices.append(device)
        rep.success(
            f"  NPU {index}: {name} | 可用 (via torch_npu.is_available)"
        )

    return True


def check_devices(rep: Report):
    rep.section("[1/4] NPU 设备检测 (主: npu-smi / 回退: torch_npu)...")

    npu_ids = None
    collector = None
    try:
        sys.path.insert(0, _SCRIPT_DIR)
        from _npu_info import NpuInfoCollector  # type: ignore  # noqa: PLC0415, PLC2701
        collector = NpuInfoCollector()
        npu_ids = collector.get_npu_ids()
    except Exception as e:
        # _npu_info 导入或枚举异常，不立即失败，尝试 torch_npu 回退
        rep.warning(f"npu-smi 后端不可用: {e}，尝试 torch_npu 回退")

    # 主路径成功枚举到设备
    if npu_ids:
        rep.npu_usable = True  # [1] 层裁决：有可用卡
        rep.success(f"检测到 {len(npu_ids)} 个 NPU 设备 (via npu-smi): {npu_ids}")
        for npu_id in npu_ids:
            try:
                info = collector.get_all_info(npu_id)
            except Exception as e:
                rep.warning(f"NPU {npu_id} 信息查询失败: {e}")
                rep.devices.append({"npu_id": npu_id, "health": "unknown",
                                    "error": str(e)})
                continue
            health = info.get("health") or "unknown"
            chip = info.get("chip_name") or "unknown"
            rep.devices.append({
                "npu_id": npu_id,
                "chip_name": chip,
                "health": health,
            })
            # 健康值判定依据（仅取 cannbot 文档明确出现过的值）：
            #   npu-smi: `Health : OK`      (references/npu_commands.md:69)
            #   asys:    `Healthy | 可用`   (references/asys_commands.md:22)
            if str(health).lower() in ("ok", "healthy"):
                rep.success(f"  NPU {npu_id}: {chip} | 健康={health}")
            else:
                rep.warning(f"  NPU {npu_id}: {chip} | 健康={health} (非 OK)")

        # npu-smi 解析告警透传
        for w in collector.get_all_warnings():
            rep.warning(f"npu-smi 解析告警: {w}")
        return

    # 主路径枚举为空（真实无卡 或 npu-smi 残缺 stub 环境如 a5）→ 回退 torch_npu
    if _check_devices_via_torch_npu(rep):
        return

    # 两条路径都拿不到设备
    rep.error("未检测到任何 NPU 设备 "
              "(npu-smi 枚举为空，torch_npu 回退也不可用)")


# ─────────────────────────────────────────────────────────────
# [2] NPU 架构层 — 复用 get_npu_arch.py
# ─────────────────────────────────────────────────────────────

def check_arch(rep: Report):
    rep.section("[2/4] NPU 架构探测 (libascend_hal.so via get_npu_arch.py)...")
    arch_script = os.path.join(_SCRIPT_DIR, "get_npu_arch.py")
    if not os.path.isfile(arch_script):
        rep.warning("get_npu_arch.py 不存在，跳过架构探测")
        return
    try:
        out = subprocess.run(
            [sys.executable, arch_script],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        rep.warning("架构探测超时 (30s)")
        return
    except Exception as e:
        rep.warning(f"架构探测执行失败: {e}")
        return

    # get_npu_arch.py 通过 logging 输出，默认写 stderr；成功时 exit=0。
    # 因此以 returncode 为准，结果优先取 stdout，回退 stderr。
    if out.returncode == 0:
        combined = out.stdout.strip() or out.stderr.strip()
        if combined:
            arch = combined.splitlines()[-1].strip()
            rep.npu_arch = arch
            rep.success(f"NPU 架构 = {arch}")
        else:
            rep.warning("架构探测无输出")
    else:
        msg = (out.stderr or out.stdout).strip().splitlines()
        detail = msg[-1] if msg else "无输出"
        rep.warning(f"架构探测未成功: {detail}")


# ─────────────────────────────────────────────────────────────
# [3] 运行时层 — torch_npu / pypto_pro.language / pl.jit（PyPTO-Pro 专属）
# ─────────────────────────────────────────────────────────────

def check_runtime(rep: Report):
    rep.section("[3/4] 运行时依赖检测 (torch_npu / pypto_pro)...")

    # torch
    try:
        import torch  # noqa
        rep.success(f"torch {torch.__version__}")
    except Exception as e:
        rep.error(f"import torch 失败: {e}")
        return

    # torch_npu —— [3] 层只裁决"库能否 import"，不重复裁决"设备可用性"
    #   设备可用性由 [1] 层（双后端）统一裁决 → rep.npu_usable。
    #   捕获所有异常（不只 ImportError）：区分未安装 vs 已装但初始化失败，
    #   避免非 ImportError 异常（如 RuntimeError）直接崩脚本。
    try:
        import torch_npu  # noqa
    except ImportError as e:
        rep.error(f"torch_npu 未安装: {e}")
    except Exception as e:
        rep.error(f"torch_npu 已安装但初始化失败: {e}")
    else:
        # import 成功 = [3] 层通过。is_available() 仅作辅助信号，不阻断。
        try:
            avail = torch.npu.is_available()
            count = torch.npu.device_count()
        except Exception as e:
            avail, count = None, None
            rep.warning(f"torch.npu 状态查询异常（不阻断）: {e}")
        rep.torch_npu_ok = True
        if avail and count and count > 0:
            rep.success(f"torch_npu 可用 (device_count={count})")
        elif rep.npu_usable:
            # [1] 层已确认有卡，但此处 is_available() 说不可用 → 以 [1] 为准，仅告警
            rep.warning(
                f"torch_npu 已导入且 [1] 层确认有卡，但 is_available={avail} "
                f"(可能 ASCEND_RT_VISIBLE_DEVICES 屏蔽或初始化时序)；以 [1] 层为准，不阻断"
            )
        else:
            rep.warning(
                f"torch_npu 已导入但 is_available={avail}, device_count={count}"
                f"（[1] 层亦未确认可用卡，详见 [1] 层结论）"
            )

    # pypto_pro.language — 对齐本仓算子代码真实入口
    #   证据：custom/*/test_*.py 均为 `import pypto_pro.language as pl` + `@pl.jit(...)`
    #        （如 custom/softmax/test_softmax.py:30, custom/relu/test_relu.py:20）
    #   本仓无任何单独 `import pypto_pro` 顶层用法，故不检测顶层包。
    try:
        import pypto_pro.language as pl  # type: ignore  # noqa: PLC0415
        rep.pypto_pro_ok = True
        rep.success("pypto_pro.language 可导入")
        # pl.jit 可用性（算子实现依赖 @pl.jit 装饰器）
        if hasattr(pl, "jit"):
            rep.success("pl.jit 可用")
        else:
            rep.error("pypto_pro.language 无 jit 属性，无法编译 PyPTO-Pro 算子")
    except ImportError as e:
        rep.error(f"import pypto_pro.language 失败: {e}")
    except Exception as e:
        rep.error(f"pypto_pro.language 已安装但初始化失败: {e}")


# ─────────────────────────────────────────────────────────────
# [4] CANN 层 — 裁剪自 check_env.sh（只留 toolkit/set_env/版本）
# ─────────────────────────────────────────────────────────────

def _resolve_toolkit_path(base):
    if not base or not os.path.isdir(base):
        return None
    if os.path.isdir(os.path.join(base, "compiler")):
        return base
    tk = os.path.join(base, "ascend-toolkit")
    if os.path.isdir(tk):
        cands = []
        for d in os.listdir(tk):
            if d != "latest" and os.path.isdir(os.path.join(tk, d, "compiler")):
                cands.append(d)
        cands.sort(reverse=True)
        if cands:
            return os.path.join(tk, cands[0])
        latest = os.path.join(tk, "latest")
        if os.path.islink(latest):
            real = os.path.realpath(latest)
            if os.path.isdir(os.path.join(real, "compiler")):
                return real
    for d in sorted(os.listdir(base), reverse=True):
        if d.startswith("cann-") and os.path.isdir(os.path.join(base, d, "compiler")):
            return os.path.join(base, d)
    return None


def _toolkit_from_environment(opp_path):
    """Resolve the active toolkit from OPP and standard CANN variables."""
    if opp_path.endswith("/opp"):
        candidate = opp_path[:-4]
        if os.path.isdir(os.path.join(candidate, "compiler")):
            return candidate

    for variable in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME", "ASCEND_HOME_PATH"):
        toolkit = _resolve_toolkit_path(os.environ.get(variable, ""))
        if toolkit:
            return toolkit
    return None


def _read_cann_version(toolkit):
    """Read version.info, falling back to the toolkit directory name."""
    fallback = os.path.basename(toolkit).replace("cann-", "")
    version_file = os.path.join(toolkit, "compiler", "version.info")
    if not os.path.isfile(version_file):
        return fallback

    try:
        version = _version_from_file(version_file)
        return version or fallback
    except Exception as error:
        logging.debug("Failed to read CANN version.info: %s", error)
    return fallback


def _version_from_file(version_file):
    with open(version_file) as file_handle:
        for line in file_handle:
            if line.startswith("Version="):
                return line.split("=", 1)[1].strip()
    return ""


def check_cann(rep: Report):
    rep.section("[4/4] CANN Toolkit 检测...")
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    toolkit = _toolkit_from_environment(opp)

    if not toolkit:
        rep.error("无法定位 CANN Toolkit 目录 "
                  "(检查 ASCEND_HOME_PATH / 是否 source set_env.sh)")
        return

    rep.success(f"CANN Toolkit = {toolkit}")

    version = _read_cann_version(toolkit)
    rep.cann_version = version
    rep.success(f"CANN 版本 = {version}")

    # ASCEND_OPP_PATH（运行时依赖）
    if not opp:
        rep.warning("ASCEND_OPP_PATH 未设置 (运行算子时必需)")
    elif not os.path.isdir(opp):
        rep.warning(f"ASCEND_OPP_PATH 指向不存在的目录: {opp}")
    else:
        rep.success(f"ASCEND_OPP_PATH = {opp}")


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PyPTO-Pro 环境 Preflight")
    parser.add_argument("--json", action="store_true",
                        help="仅输出 JSON（不打印人类可读摘要）")
    args = parser.parse_args()

    rep = Report(quiet=args.json)

    if not args.json:
        LOGGER.info("=" * 64)
        LOGGER.info("PyPTO-Pro 环境 Preflight")
        LOGGER.info("=" * 64)

    check_devices(rep)
    check_arch(rep)
    check_runtime(rep)
    check_cann(rep)

    # 最终裁决复核：事实源优先。
    # 若 [1] 层实际枚举到可用卡（rep.npu_usable / rep.devices 非空），
    # 则不应存在任何"设备不可用"类 error —— 设备类阻断只允许在
    # 双后端都枚举不到卡时产生。这里做一致性自检，防止未来改动引入
    # "有卡却因二次布尔判定翻盘判失败"的误报。
    if rep.npu_usable and rep.devices:
        leaked = [e for e in rep.errors
                  if ("NPU 不可用" in e or "未检测到任何 NPU" in e)]
        if leaked:
            rep.warning(
                "一致性自检：[1] 层已确认可用卡，但 errors 中存在设备不可用类"
                f"条目 {leaked}，判定为误报信号，请检查检测逻辑"
            )

    if not args.json:
        LOGGER.info("\n" + "=" * 64)
        LOGGER.info("Preflight 结果")
        LOGGER.info("=" * 64)
        if rep.errors:
            LOGGER.error("%s✗ 环境检测未通过：%d 个错误, %d 个警告%s",
                         _RED, len(rep.errors), len(rep.warnings), _NC)
            for e in rep.errors:
                LOGGER.error("  %s✗%s %s", _RED, _NC, e)
        elif rep.warnings:
            LOGGER.warning("%s⚠ 环境检测通过（含 %d 个警告）%s",
                           _YELLOW, len(rep.warnings), _NC)
        else:
            LOGGER.info("%s✓ 环境检测全部通过%s", _GREEN, _NC)

    payload = json.dumps(rep.to_dict(), ensure_ascii=False)
    if args.json:
        LOGGER.info(payload)
    else:
        # 默认模式保留人类可读摘要后的机器可读标记行。
        LOGGER.info("PREFLIGHT_JSON: " + payload)

    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
