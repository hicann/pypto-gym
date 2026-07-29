#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO 项目 CI 场景构建控制总入口

本文件提供 PyPTO 项目 CI 场景的统一构建入口, 支持多种构建模式和配置选项.

主要功能:
    - 支持 whl 包的常规编译和可编辑模式编译
    - 支持 UTest/STest/Examples 等测试用例的执行
    - 支持构建超时控制和超时后自动清理子进程

使用方式:
    通过命令行参数配置构建选项, 执行脚本即可触发构建流程:

        python build_ci.py [选项]

    常用选项:
        -f/--frontend: 指定前端类型 (python3/cpp)
        -b/--backend: 指定后端类型 (npu/cost_model)
        -t/--targets: 指定编译目标
        -j/--job_num: 指定编译并行度
        --build_type: 指定构建类型 (Debug/Release/MinSizeRel/RelWithDebInfo)
        -c/--clean: 清理构建目录和安装目录

示例:
    # 使用默认配置构建
    python build_ci.py

    # 指定前端和后端类型构建
    python build_ci.py -f python3 -b npu

    # 启用测试并指定并行度
    python build_ci.py -u -s -j 8

    # 清理并重新构建
    python build_ci.py -c --build_type Debug
"""
import abc
import argparse
import dataclasses
import logging
import math
import multiprocessing
import os
import re
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from importlib import metadata
from packaging import requirements



@dataclasses.dataclass
class FeatureParam():
    """特性控制相关参数

    管理构建过程中的特性选项, 包括前端类型, 后端类型和 whl 包编译模式.
    """
    whl_name: str = "pypto"
    frontend_type: Optional[str] = None  # 前端类型, 支持 python3, cpp
    backend_type: Optional[str] = None  # 后端类型, 支持 npu, cost_model
    whl_plat_name: Optional[str] = None  # python3 whl 包 plat-name
    whl_isolation: bool = False  # 以 isolation 模式编译 whl 包

    def __init__(self, args):
        """初始化 FeatureParam 实例

        从命令行参数中解析前端类型, 后端类型和 whl 包编译模式.
        如果后端类型为 npu 但未设置 ASCEND_HOME_PATH 环境变量, 则自动回退到 cost_model 后端.

        :param args: 命令行参数解析结果
        """
        self.frontend_type = "python3" if args.frontend is None else args.frontend
        self.backend_type = "npu" if args.backend is None else args.backend
        if not os.environ.get("ASCEND_HOME_PATH") and self.backend_type in ["npu"]:
            logging.warning("Environment variable ASCEND_HOME_PATH is unset/empty, falling back to cost_model backend.")
            self.backend_type = "cost_model"
        self.whl_plat_name = f"{args.plat_name}_{BuildCtrl.get_system_processor()}" if args.plat_name else ""
        self.whl_isolation = args.isolation
        self.multi_py3_cfg = None

    def __str__(self) -> str:
        """返回特性参数的字符串表示

        :return: 格式化的特性参数字符串
        :rtype: str
        """
        desc = ""
        desc += f"\nFeature"
        desc += f"\n    Frontend                : {self.frontend_type}"
        if self.frontend_type_python3:
            if self.whl_plat_name:
                desc += f"\n    PlatName                : {self.whl_plat_name}"
            desc += f"\n    Isolation               : {self.whl_isolation}"
        desc += f"\n    Backend                 : {self.backend_type}"
        return desc

    @property
    def frontend_type_python3(self) -> bool:
        """判断前端类型是否为 Python3

        :return: 如果前端类型为 "python" 或 "python3", 返回 True
        :rtype: bool
        """
        return self.frontend_type in ["python", "python3"]
    
    @property
    def multi_py3_min_minor(self) -> Optional[int]:
        minor_list = [int(c.minor) for c in (self.multi_py3_cfg or [])]
        if minor_list:
            return min(minor_list)
        else:
            return None

    @property
    def multi_py3_exe_cfg(self) -> str:
        if not self.multi_py3_cfg:
            return ""
        cfg_list = [f"{cfg.exe}" for cfg in self.multi_py3_cfg]
        return ":".join(cfg_list)

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册特性相关的命令行参数

        向参数解析器注册前端类型, 后端类型, whl 编译模式等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("-f", "--frontend", nargs="?", type=str, default="python3",
                            choices=["python3", "cpp"],
                            help="frontend, such as python3/cpp etc.")
        parser.add_argument("--plat_name", nargs="?", type=str, default="",
                            choices=["manylinux2014", "manylinux_2_24", "manylinux_2_28"],
                            help="whl plat_name, such as manylinux2014/manylinux_2_24/manylinux_2_28 etc.")
        parser.add_argument("-b", "--backend", nargs="?", type=str, default="npu",
                            choices=["npu", "cost_model"],
                            help="backend, such as npu/cost_model etc.")
        parser.add_argument("--no_isolation", action="store_false", default=True, dest="isolation",
                            help="Disable building the project(whl) in an isolated virtual environment. "
                                 "Build dependencies must be installed separately when this option is used.")


@dataclasses.dataclass
class TestsExecuteParam():
    """测试执行相关参数

    管理测试执行的配置选项, 包括自动执行, 并行执行, 超时控制和耗时缓存等.
    """
    changed_file: Optional[Path] = None  # 修改文件路径
    auto_execute: bool = False  # 用例自动执行
    auto_execute_parallel: bool = False  # 用例并行执行
    case_execute_timeout: Optional[int] = None  # 用例执行时, 单个用例超时时长
    case_execute_cpu_rank_size: Optional[int] = None  # 用例并行执行时, CPU 亲和性 Rank Size
    dump_case_duration_json: Optional[Path] = None  # 用例耗时缓存文件路径
    dump_case_duration_max_num: Optional[int] = None  # 用例耗时缓存最大数量
    dump_case_duration_min_secends: Optional[int] = None  # 用例耗时缓存最小秒数

    def __init__(self, args):
        """初始化 TestsExecuteParam 实例

        从命令行参数中解析测试执行相关的配置选项.

        :param args: 命令行参数解析结果
        """
        self.changed_file = None if not args.changed_files else Path(args.changed_files).resolve()
        self.auto_execute = args.disable_auto_execute
        self.auto_execute_parallel = self.auto_execute and self.ci_model
        timeout = args.case_execute_timeout
        self.case_execute_timeout = timeout if timeout and timeout > 0 else None  # 单个用例执行超时时长
        self.case_execute_cpu_rank_size = args.cpu_rank_size
        duration_json = args.dump_case_duration_json
        self.dump_case_duration_json = Path(duration_json).resolve() if duration_json else None
        self.dump_case_duration_max_num = args.dump_case_duration_max_num
        self.dump_case_duration_min_secends = args.dump_case_duration_min_secends

    def __str__(self) -> str:
        """返回测试执行参数的字符串表示

        :return: 格式化的测试执行参数字符串
        :rtype: str
        """
        desc = f"\n    Execute"
        desc += f"\n               Changed File : {self.changed_file}"
        desc += f"\n                       Auto : {self.auto_execute}"
        desc += f"\n                   Parallel : {self.auto_execute_parallel}"
        desc += f"\n                CaseTimeout : {self.case_execute_timeout}"
        desc += f"\n        CaseDuration"
        desc += f"\n                       Json : {self.dump_case_duration_json}"
        desc += f"\n                     MaxNum : {self.dump_case_duration_max_num}"
        desc += f"\n                     MinSec : {self.dump_case_duration_min_secends}"
        return desc

    @property
    def ci_model(self) -> bool:
        """判断是否为 CI 模式

        :return: 如果指定了修改文件, 则返回 True (表示 CI 模式)
        :rtype: bool
        """
        return True if self.changed_file else False

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册测试执行相关的命令行参数

        向参数解析器注册增量测试, 自动执行, 超时控制等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("--changed_files", nargs="?", type=Path, default=None,
                            help="Specify the file of files changed, "
                                 "so that the corresponding test cases can be triggered incrementally.")
        parser.add_argument("--disable_auto_execute", action="store_false", default=True,
                            help="Disable auto execute STest/Utest with build.")
        parser.add_argument("--case_execute_timeout", nargs="?", type=int, default=None,
                            help="Case execute timeout.")
        parser.add_argument("--cpu_rank_size", nargs="?", type=int, default=None,
                            help="Specify the rank size for CPU affinity grouping.")
        parser.add_argument("--dump_case_duration_json", nargs="?", type=Path, default=None,
                            help="Specify the path to the case duration json cache file.")
        parser.add_argument("--dump_case_duration_max_num", nargs="?", type=int, default=None,
                            help="Maximum number of cases to dump to duration json cache.")
        parser.add_argument("--dump_case_duration_min_secends", nargs="?", type=int, default=None,
                            help="Minimum duration (in seconds) for cases to dump to duration json cache.")


@dataclasses.dataclass
class TestsGoldenParam():
    """Golden 测试相关参数

    管理系统测试 (STest) 的 Golden 标准数据相关配置.
    """
    clean: bool = False  # 清理 Golden 标记
    path: Optional[Path] = None  # 指定 Golden 路径

    def __init__(self, args):
        """初始化 TestsGoldenParam 实例

        从命令行参数中解析 Golden 测试相关配置.

        :param args: 命令行参数解析结果
        """
        self.clean = args.golden_clean
        if args.golden_path:
            # 传参且指定具体路径时, 使用指定路径, 否则具体缺省路径由 CMake 侧决定
            self.path = Path(args.golden_path).resolve()

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册 Golden 测试相关的命令行参数

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        parser.add_argument("--golden_path", "--stest_golden_path", nargs="?", type=str, default="",
                            help="Specific Tests golden path.", dest="golden_path")
        parser.add_argument("--golden_clean", "--golden_path_clean", "--stest_golden_path_clean",
                            action="store_true", default=False,
                            help="Clean Tests golden.", dest="golden_clean")


@dataclasses.dataclass
class BuildParam():
    """构建相关参数

    管理构建过程的配置选项, 包括 CMake 配置参数和构建执行参数.
    """
    # Configure
    generator: Optional[str] = None  # Generator
    build_type: Optional[str] = None  # 构建类型
    asan: bool = False  # 使能 AddressSanitizer
    ubsan: bool = False  # 使能 UndefinedBehaviorSanitizer
    gcov: bool = False  # 使能 GNU Coverage
    gcov_incr: bool = False  # 使能增量覆盖率 GCov 计算
    compile_dependency_check: bool = False  # 使能编译依赖关系检查
    # Build
    targets: Optional[List[str]] = None  # 编译目标
    job_num: Optional[int] = None  # 编译阶段使用核数

    def __init__(self, args):
        """初始化 BuildParam 实例

        从命令行参数中解析构建相关的配置选项.

        :param args: 命令行参数解析结果
        """
        self.targets = args.targets
        self.job_num = self._get_job_num(job_num=args.job_num, generator=args.generator)
        self.generator = self._get_generator(generator=args.generator)
        self.build_type = args.build_type
        self.asan = args.asan
        self.ubsan = args.ubsan
        self.gcov = args.gcov
        self.gcov_incr = args.gcov_increment
        self.compile_dependency_check = args.compile_dependency_check

    def __str__(self) -> str:
        """返回构建参数的字符串表示

        :return: 格式化的构建参数字符串
        :rtype: str
        """
        desc = f"\nBuild"
        desc += f"\n    CMake"
        desc += f"\n        Configure"
        desc += f"\n                  Generator : {self.generator}"
        desc += f"\n                  BuildType : {self.build_type}"
        desc += f"\n                       ASan : {self.asan}"
        desc += f"\n                      UbSan : {self.ubsan}"
        desc += f"\n                       GCov : {self.gcov}, Increment: {self.gcov_incr}"
        desc += f"\n            CompileDepCheck : {self.compile_dependency_check}"
        desc += f"\n        Build"
        desc += f"\n                    Targets : {self.targets}"
        desc += f"\n                    Job Num : {self.job_num}"
        return desc

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册构建相关的命令行参数

        向参数解析器注册构建生成器, 构建类型, Sanitizer 选项等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 未使用
        :type ext: Optional[Any]
        """
        # Configure
        parser.add_argument("--generator", nargs="?", type=str, default="",
                            help="Specify a build system generator.")
        parser.add_argument("--build_type", "--build-type", nargs="?", type=str, default="Release",
                            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
                            help="build type.")
        parser.add_argument("--asan", action="store_true", default=False,
                            help="Enable AddressSanitizer.")
        parser.add_argument("--ubsan", action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer.")
        parser.add_argument("--gcov", action="store_true", default=False,
                            help="Enable GNU Coverage Instrumentation Tool.")
        parser.add_argument("--gcov_increment", action="store_true", default=False,
                            help="Enable increment coverage calculation based on latest commit.")
        parser.add_argument("--clang", nargs="?", type=str, default="",
                            help="Specify clang install path, such as /usr/bin/clang")
        parser.add_argument("--compile_dependency_check", action="store_true", default=False,
                            help="Enable compile dependency relation check.")
        # Build
        parser.add_argument("-t", "--targets", nargs="?", type=str, action="append",
                            help="targets, specific build targets, "
                                 "If you specify more than one, all targets within the specified range are built.")
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=-1,
                            help="job num, specific job num of build.")

    @staticmethod
    def _get_job_num(job_num: Optional[int], generator: Optional[str]) -> Optional[int]:
        """获取构建并行任务数

        根据系统 CPU 核数和构建生成器类型确定合适的并行任务数. 如果使用 Ninja 生成器, 则由 Ninja 自动决定并行度.

        :param job_num: 用户指定的并行任务数
        :type job_num: Optional[int]
        :param generator: 构建生成器名称
        :type generator: Optional[str]
        :return: 最终的并行任务数, None 表示由构建工具自动决定
        :rtype: Optional[int]
        """
        def_job_num = min(int(math.ceil(float(multiprocessing.cpu_count()) * 0.9)), 128)  # 128 为缺省最大核数
        def_job_num = None if generator and generator.lower() in ["ninja", ] else def_job_num  # ninja 自身决定缺省核数
        job_num = job_num if job_num and job_num > 0 else def_job_num
        return job_num

    @staticmethod
    def _get_generator(generator: Optional[str]) -> Optional[str]:
        """获取构建生成器名称

        如果指定了生成器, 则在名称外添加引号以支持带空格的生成器名称.

        :param generator: 构建生成器名称
        :type generator: Optional[str]
        :return: 处理后的构建生成器名称
        :rtype: Optional[str]
        """
        return f"\"{generator}\"" if generator else generator

    def get_build_cmd_lst(self, cmake: Path, binary_path: Path) -> List[str]:
        """生成 CMake 构建命令列表

        根据指定的构建目标生成对应的 CMake 构建命令.

        :param cmake: CMake 可执行文件路径
        :type cmake: Path
        :param binary_path: 二进制构建目录路径
        :type binary_path: Path
        :return: CMake 构建命令列表
        :rtype: List[str]
        """
        cmd_list = []
        if self.targets:
            for t in self.targets:
                cmd = f"{cmake} --build {binary_path} --target {t}"
                cmd += f" -j {self.job_num}" if self.job_num else ""
                cmd_list.append(cmd)
        else:


            cmd = f"{cmake} --build {binary_path}"
            cmd += f" -j {self.job_num}" if self.job_num else ""
            cmd_list.append(cmd)
        return cmd_list


@dataclasses.dataclass
class TestsFilterParam():
    """测试过滤参数

    用于按条件过滤测试用例, 支持多种测试类型和过滤模式.
    """
    cmake_option: str = ""
    enable: bool = False
    filter_str: Optional[str] = None

    def __init__(self, argv: Optional[str], opt: str = ""):
        """初始化 TestsFilterParam 实例

        根据命令行参数值确定过滤选项的启用状态和过滤字符串.

        :param argv: 命令行参数值, None 表示启用默认过滤, 空字符串表示禁用, 其他值表示指定过滤字符串
        :type argv: Optional[str]
        :param opt: CMake 选项名称
        :type opt: str
        """
        self.cmake_option = opt
        if argv is None:
            self.enable, self.filter_str = True, "ON"  # 指定 对应参数, 但未指定内容
        elif argv == "":
            self.enable, self.filter_str = False, "OFF"  # 未指定 对应参数
        else:
            self.enable, self.filter_str = True, argv  # 指定 对应参数 且指定内容

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册测试过滤相关的命令行参数

        根据扩展信息生成对应的命令行参数选项.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息, 用于生成参数名称和帮助信息
        :type ext: Optional[Any]
        """
        mark = str(ext).lower()
        mark_lst = mark.split("_")
        have_char = len(mark_lst) <= 1
        mark_word = mark.replace("_", " ")
        help_str = f"Enable {mark_word} scene, specific {mark_word} filter, multiple cases are separated by ','"
        if have_char:
            mark_char = mark_lst[0][0] if have_char else None
            parser.add_argument(f"-{mark_char}", f"--{mark}", nargs="?", type=str, default="", help=help_str)
        else:
            parser.add_argument(f"--{mark}", nargs="?", type=str, default="", help=help_str)

    def get_filter_str(self, def_filter: str) -> str:
        """获取测试过滤字符串

        根据配置和默认过滤条件生成最终的过滤字符串.

        :param def_filter: 默认过滤条件
        :type def_filter: str
        :return: 过滤字符串, 如果未启用则返回空字符串
        :rtype: str
        """
        if not self.enable:
            return ""
        if self.filter_str not in ["ON"]:
            return self.filter_str
        if def_filter:
            return def_filter
        return self.filter_str


class TestsParam():
    """测试参数总控类

    聚合所有测试相关的参数配置, 包括执行参数, Golden 参数, 过滤参数等.
    """

    def __init__(self, args):
        """初始化 TestsParam 实例

        从命令行参数中解析并初始化所有测试相关的参数配置.

        :param args: 命令行参数解析结果
        """
        self.exec: TestsExecuteParam = TestsExecuteParam(args=args)
        self.golden: TestsGoldenParam = TestsGoldenParam(args=args)
        self.models: TestsFilterParam = TestsFilterParam(argv=args.models)

    def __str__(self) -> str:
        """返回测试参数的字符串表示

        :return: 格式化的测试参数字符串
        :rtype: str
        """
        if not self.enable:
            return ""
        desc = f"\nTests"
        desc += f"{self.exec}"
        if self.models.enable:
            desc += f"\n    Models"
            desc += f"\n                     Enable : {self.models.enable}"
            desc += f"\n                     Filter : {self.models.filter_str}"
        return desc

    @property
    def enable(self) -> bool:
        """判断是否启用任意测试

        :return: 如果启用了任意类型的测试, 返回 True
        :rtype: bool
        """
        return self.models.enable

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        """注册所有测试相关的命令行参数

        向参数解析器注册测试执行, Golden 测试, 过滤选项等参数.

        :param parser: ArgumentParser 参数解析器实例
        :param ext: 扩展信息 (子命令解析器)
        :type ext: Optional[Any]
        """
        TestsExecuteParam.reg_args(parser=parser)
        TestsGoldenParam.reg_args(parser=parser)
        TestsFilterParam.reg_args(parser=parser, ext="models")


class BuildCtrl():
    """构建过程控制类

    本类包含由命令行指定或解析出的控制标记/参数, 以控制构建过程执行. 是整个构建流程的入口和控制器, 负责协调整个构建过程.
    """
    _PYTHONPATH: str = "PYTHONPATH"

    def __init__(self, args):
        """初始化 BuildCtrl 实例

        从命令行参数中解析并初始化所有构建相关的配置.

        :param args: 命令行参数解析结果
        """
        self.origin_timeout: Optional[int] = args.timeout if args.timeout and args.timeout > 0 else None  # 超时时长
        self.remain_timeout: Optional[int] = self.origin_timeout
        self.src_root: Path = Path(__file__).parent.resolve()
        self.pypto_root: Path = Path(__file__).parent.parent.resolve().joinpath("pypto")
        self.build_root: Path = Path(self.pypto_root, "build")
        self.install_root: Path = Path(self.build_root.parent, "build_out")
        self.feature: FeatureParam = FeatureParam(args=args)
        self.build: BuildParam = BuildParam(args=args)
        self.tests: TestsParam = TestsParam(args=args)
        self.third_party_path: Optional[Path] = Path(args.third_party_path).resolve() if args.third_party_path else None
        self.verbose: bool = args.verbose
        # 表示 pip 版本是否支持传递 --config-setting 这种 pep 标准参数传递方式
        self.pip_dependence_desc: Dict[str, str] = {"pip": ">=22.1"}
        self.pip_support_config_setting = self.check_pip_dependencies(deps=self.pip_dependence_desc,
                                                                      raise_err=False, log_err=False)

        devs = ["0"]
        if args.device is not None:
            devs = [str(d) for d in args.device if d is not None]
        self.auto_execute_device_id = ":".join(devs)

    def __str__(self) -> str:
        """返回构建控制参数的字符串表示

        :return: 格式化的构建控制参数字符串
        :rtype: str
        """
        py3_ver = sys.version_info
        pip_ver = metadata.version("pip")
        desc = ""
        desc += f"\nEnviron"
        desc += f"\n    Python3                 : {sys.executable} ({py3_ver.major}.{py3_ver.minor}.{py3_ver.micro})"
        desc += f"\n    pip3                    : {pip_ver}"
        desc += f"\nPath"
        desc += f"\n    Source  Dir             : {self.pypto_root}"
        desc += f"\n    Build   Dir             : {self.build_root}"
        desc += f"\n    Install Dir             : {self.install_root}"
        desc += f"\n    3rd     Dir             : {self.third_party_path}"
        desc += f"\nFlag"
        desc += f"\n    Verbose                 : {self.verbose}"
        desc += f"\nOthers"
        desc += f"\n    Timeout                 : {self.origin_timeout}"
        desc += f"{self.feature}"
        desc += f"{self.build}"
        desc += f"{self.tests}"
        desc += f"\n"
        return desc

    @staticmethod
    def find_match_whl(name: str, path: Path) -> Optional[Path]:
        """
        在指定路径下, 查找对应匹配的 whl 包文件

        :param name: 包名
        :type name: str
        :param path: 指定路径
        :type path: Path
        :return: 指定路径
        :rtype: Path | None
        """
        cpp_desc = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pattern = f"{name}-*-{cpp_desc}-{cpp_desc}-*.whl"
        whl_glob = path.glob(pattern=pattern)
        whl_files = [Path(f) for f in whl_glob]
        whl_file = whl_files[0] if whl_files else None
        if whl_file:
            logging.info("Success find match %s from %s", whl_file, path)
        else:
            logging.error("Failed to find match %s whl from %s, pattern=%s", name, path, pattern)
        return whl_file

    @staticmethod
    def reg_args(parser, ext: Optional[Any] = None):
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Device ID, default 0.")
        parser.add_argument("-c", "--clean", action="store_true", default=False,
                            help="clean, clean Build-Tree and Install-Tree before build.")
        parser.add_argument("--timeout", nargs="?", type=int, default=None,
                            help="Total timeout.")
        parser.add_argument("--cann_3rd_lib_path", "--third_party_path",
                            nargs="?", type=str, default="", dest="third_party_path",
                            help="Specify 3rd Libraries Path")
        parser.add_argument("--verbose", action="store_true", default=False,
                            help="verbose, enable verbose output.")

    @classmethod
    def check_pip_dependencies(cls, deps: Dict[str, str], raise_err: bool = False, log_err: bool = True) -> bool:
        info_lst = []
        for pkg, ver in deps.items():
            info = cls._check_pip_pkg(pkg=pkg, ver=ver)
            info_lst.extend(info)
        if info_lst:
            if log_err:
                logging.error("%s", info_lst)
                install_cmd = " ".join([f'{pkg}{deps[pkg]}' for pkg in deps])
                logging.error(f"Please install the missing dependencies first [{install_cmd}]")
            if raise_err:
                raise RuntimeError("\n".join(info_lst))
            return False
        return True

    @classmethod
    def main(cls):
        ts = datetime.now(tz=timezone.utc)
        try:
            cls._main()
        except KeyboardInterrupt as e:
            logging.error("Operation cancelled by user")
            raise e
        except subprocess.TimeoutExpired as e:
            logging.error("Operation timeout, %s", e)
            raise e
        # 计算总耗时
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("Build[CI] Finish, Duration %s secs.", duration)

    @classmethod
    def _check_pip_pkg(cls, pkg: str, ver: str) -> List[str]:
        info_lst = []
        requirement_str = f"{pkg}{ver}"
        try:
            req = requirements.Requirement(requirement_str)
            try:
                installed_version = metadata.version(pkg)
                if ver and not req.specifier.contains(installed_version, prereleases=True):
                    info_lst.append(f"{pkg}: version {installed_version} not satisfy {ver}")
            except metadata.PackageNotFoundError:
                info_lst.append(f"package {pkg} has not been installed")
        except Exception as e:
            info_lst.append(f"package {pkg} check fail {e}")
        return info_lst

    @classmethod
    def _main(cls):
        """主处理流程
        """
        parser = argparse.ArgumentParser(description=f"PyPTO Build Ctrl.", epilog="Best Regards!")
        sub_parser = parser.add_subparsers()  # 子命令
        # 参数注册
        FeatureParam.reg_args(parser=parser)
        BuildParam.reg_args(parser=parser)
        TestsParam.reg_args(parser=parser, ext=sub_parser)
        BuildCtrl.reg_args(parser=parser)

        # 参数处理
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        # 流程处理
        if ctrl.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        # 区分 python3 前端和 cpp 前端
        logging.info("%s", ctrl)
        logging.info("Front-end(python3), start process")
        ctrl.py_clean()
        ctrl.py_build()
        ctrl.py_tests()
    

    @staticmethod
    def get_system_processor() -> str:
        """获取系统处理器架构名称

        通过 platform.machine() 获取当前系统的处理器架构, 并将常见的别名映射到标准名称.

        :return: 标准化的处理器架构名称, 如 x86_64 或 aarch64
        :rtype: str
        """
        machine = platform.machine().lower()
        arch_map = {  # 直接映射常见架构
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
        }
        return arch_map.get(machine, machine)


    def run_build_cmd(self, cmd: str, update_env: Optional[Dict[str, str]] = None,
                      check: bool = True, pg_desc: str = "CMake") -> Tuple[subprocess.CompletedProcess, str]:
        """执行具体 build 命令行

        因以下原因, 设置本函数, 而非调用原生 subprocess.run
            1. 支持多 target 构建, 各 target 构建时长共享公共 timeout 配置;

        本函数内支持 timeout 重计算, 仅执行成功时会进行重计算

        :param cmd: Build 命令行
        :param update_env: 环境变量(额外更新内容)
        :param check: 检查返回值
        :param pg_desc: Process Group Desc, 进程组描述
        """

        def _stop_pg(_msg: str, _p: subprocess.Popen):
            """通过 SIGINT 信号通知所有子/孙进程结束, python 并行脚本内会捕获该信号进行结算处理
            """
            _pgid = os.getpgid(_p.pid)
            logging.info("%s. Send terminate event to %s[%s]", _msg, pg_desc, _pgid)
            os.killpg(_pgid, signal.SIGINT)

        ts = datetime.now(tz=timezone.utc)
        stdout = None
        stderr = None
        env = os.environ.copy()
        env.update(update_env if update_env else {})
        with subprocess.Popen(shlex.split(cmd), env=env, text=True, encoding='utf-8',
                              start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.remain_timeout)
            except subprocess.TimeoutExpired as e:
                _stop_pg(_msg=f"Timeout({self.remain_timeout})", _p=process)
                raise e
            except KeyboardInterrupt as e:
                _stop_pg(_msg="KeyboardInterrupt", _p=process)
                raise e
            except Exception as e:
                process.kill()
                raise e
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if check and ret_code:
                raise subprocess.CalledProcessError(ret_code, process.args, output=stdout, stderr=stderr)
        # 超时时长更新
        duration = self._duration(ts=ts)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr), duration
    
    def pip_install(self, install_root: Path, dest: Optional[Path] = None,
                    update_env: Optional[Dict[str, str]] = None):
        """安装指定的 whl 包

        使用 pip 命令安装指定的 whl 包, 支持自定义安装路径和参数.

        :param whl: whl 包文件路径
        :type whl: Path
        :param dest: 安装路径, 未指定时使用默认路径
        :type dest: Optional[Path]
        :param opt: 额外安装参数
        :type opt: str
        :param update_env: 环境变量 (额外更新内容)
        :type update_env: Optional[Dict[str, str]]
        """
        root_dir = "."
        # 递归匹配 pypto 开头 whl
        wheel_list = list(Path(install_root).rglob("pypto*.whl"))

        # 只取唯一包（目录仅有一个）
        if wheel_list:
            whl_file = wheel_list[0]
            whl_path = whl_file.resolve()
            whl_name = whl_file.name
            print("找到唯一whl文件完整路径:", whl_path)
            print("whl包名:", whl_name)
        else:
            raise FileNotFoundError("未找到任何 pypto 开头的 .whl 安装包")

        cmd = f"{sys.executable} -m pip install " + f"{whl_path}" + \
            (" --no-compile --no-deps --no-cache-dir ")
        logging.info("Install %s, Cmd: %s, Timeout: %s", whl_path, cmd, self.remain_timeout)
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pip")
        logging.info("Install %s%s success, %s", whl_path, f" to {dest}" if dest else "", duration)
        
    
    def pip_uninstall(self, name: str, path: Optional[Path] = None):
        """卸载指定的 whl 包

        根据是否指定安装路径, 选择使用 pip 卸载或直接删除文件.

        :param name: 包名
        :type name: str
        :param path: 指定安装路径, 如果指定则直接删除对应路径下的文件
        :type path: Optional[Path]
        """
        
        cmd = f"{sys.executable} -m pip uninstall -v -y {name}"
        logging.info("Uninstall %s package, Cmd: %s, Timeout: %s", name, cmd, self.remain_timeout)
        _, _ = self.run_build_cmd(cmd=cmd, pg_desc="pip")
        logging.info("Uninstall %s package%s success", name, f" from {path}" if path else "")
    

    def py_clean(self):
        """清理 Python 前端构建的中间结果

        清理包括 CMake 构建目录, Python 缓存文件, 输出目录等. 仅在 clean 标记为 True 时执行额外清理.
        """
        try:
            # 切换工作目录至 pypto 源码根目录 self.pypto_root
            os.chdir(self.pypto_root)
            pkg_src = Path(self.pypto_root)
            path_lst = [
                Path(Path.cwd(), "output"),
                Path(Path.cwd(), "kernel_meta"),
                Path(self.pypto_root, "python/pypto.egg-info"),
                Path(pkg_src, "__pycache__"),
                Path(pkg_src, "op/__pycache__"),
                Path(pkg_src, "lib"),  # edit 模式
            ]
            so_glob = pkg_src.glob(pattern=f"*.so")
            so_path = [Path(p) for p in so_glob]
            path_lst.extend(so_path)
            for cache_dir in path_lst:
                if not cache_dir.exists():
                    continue
                logging.info("Clean Cache/Output Path(%s)", cache_dir)
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                else:
                    os.remove(cache_dir)
        finally:
            os.chdir(self.src_root)

    def py_build(self):
        """whl 包编译处理

        支持两种编译模式:
            1. 正式编译: 调用 build 库触发 setuptools(bdist_wheel 命令) 进而触发 CMake 完成编译
            2. pip 编译: 调用 pip install 命令触发 setuptools(editable_wheel 命令) 进而触发 CMake 完成编译
               pip 编译有两种模式:
               - 常规安装: 适用于生产环境或代码稳定后使用, 安装后对源码的修改不会反映到已安装的包中
               - 可编辑安装: 便于开发调试, 在 site-packages 中创建指向本地的链接,
                 对 Python 源码的修改会即时生效, 无需重新安装
        """
        update_env = {}
        if self.third_party_path:
            print("third party set")
            update_env.update({"PYPTO_THIRD_PARTY_PATH": self.third_party_path})
        update_env.update({"PYPTO_ALLOW_WHL_BUILD": "1"})
        try:
            # 切换工作目录至 pypto 源码根目录 self.pypto_root
            os.chdir(self.pypto_root)
            # 检查 build 包版本是否符合要求, 之所以将其放在此处检查, 是因为 pyproject.toml 中 build-system.requires 的检查功能
            # 就是 build 包实现的, 所以将其写在 pyproject.toml 中并无法提前检查
            self.check_pip_dependencies(deps={"build": ">=1.0.3"}, raise_err=True, log_err=True)
            cmd = f"{sys.executable} -m build --outdir={self.install_root}"
            cmd += f" --no-isolation" if not self.feature.whl_isolation else ""
            cmd += f" {self._get_setuptools_bdist_wheel_config_setting()}"
            logging.info("Build whl, Cmd: %s, Timeout: %s", cmd, self.remain_timeout)
            _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="build")
            logging.info("Build whl success, %s", duration)

            opt = f" --no-compile --no-deps"
            opt += f" --no-build-isolation" if not self.feature.whl_isolation else ""

            cmd_config_setting, env_config_setting = self._get_setuptools_build_ext_config_setting()
            if self.pip_support_config_setting:
                opt += f" {cmd_config_setting}" if cmd_config_setting else ""
            else:
                # pip 低版本无 --config-setting 参数, 此时以环境变量方式传入
                update_env["PYPTO_BUILD_EXT_ARGS"] = env_config_setting

            # 重装 whl 包
            dist = self._get_pip_install_dist()
            self.pip_uninstall(name=self.feature.whl_name, path=dist)
            self.pip_install(install_root=self.install_root, dest=dist, update_env=update_env)
        finally:
            os.chdir(self.src_root)

    def py_tests(self):
        """执行 Python 前端测试

        包括单元测试 (UTest) , 系统测试 (STest) , 模型测试 (Models) .
        如果未使用 pip 安装模式, 会先卸载并重新安装 whl 包.
        """
        os.chdir(self.src_root)
        if not self.tests.models.enable:
            return
        dist = None
        # 执行用例, UTest
        # 在 Python 3.12 中, pytest-xdist 通过 os.fork() 创建子进程时会产生 DeprecationWarning.
        # 使用 -W ignore::DeprecationWarning 参数来忽略该警告.
        if self.build.job_num is not None and self.build.job_num > 0:
            n_workers = str(self.build.job_num)
        else:
            n_workers = "auto"

        # 执行用例, Models/STest, 支持混合执行
        dev_lst = [int(d) for d in self.auto_execute_device_id.split(":")]
        dev_ext = " ".join(f"{d}" for d in dev_lst)
        ext_str = f"-n {len(dev_lst)}"

        # 关键：必须让 pytest 自动分配设备
        ext_str += " --dist=loadscope"

        # 执行用例Models
        self.py_tests_run_pytest(dist=dist, params=[(self.tests.models, "tests/ops/")],
                                 ext=ext_str)

    def py_tests_run_pytest(self, dist: Optional[Path], params: List[Tuple[TestsFilterParam, str]], ext: str = ""):
        """调用 pytest 执行测试用例

        支持多路径下用例混跑, 可以根据配置并行执行.

        :param dist: 二进制分发包安装路径
        :type dist: Optional[Path]
        :param params: 参数列表, 支持多路径下用例混跑, 每个元素为 (TestsFilterParam, 测试路径)
        :type params: List[Tuple[TestsFilterParam, str]]
        :param ext: 扩展命令参数
        :type ext: str
        """
        # filter 处理
        filter_str = ""
        for cur_tests, cur_filter_str in params:
            cur_filter_str = cur_tests.get_filter_str(def_filter=cur_filter_str)
            if cur_filter_str:
                filter_str += f" {cur_filter_str}"
        if not filter_str:
            return
        # 执行 pytest
        self._py_tests_run_pytest(dist=dist, filter_str=filter_str, ext=ext)

    def _py_tests_run_pytest(self, dist: Optional[Path], filter_str: str, ext: str = ""):
        if not self.tests.exec.auto_execute:
            return

        filter_str = filter_str.replace(',', ' ')


        device_id_raw = self.auto_execute_device_id

        # 空值保护
        if not device_id_raw:
            device_id_raw = "0"
        # ===================== 修复 =====================
        # 把设备列表传入环境变量，pytest 插件会自动均分
        dev_ids = self.auto_execute_device_id.replace(":", ",")
        
        update_env = os.environ.copy()
        device_id_raw = self.auto_execute_device_id
        dev_list = device_id_raw.split(":")
        update_env["ASCEND_VISIBLE_DEVICES"] = dev_ids
        update_env["PYTEST_AVAILABLE_DEVICES"] = ",".join(dev_list)

        # ===============================================
        cmd = f"{sys.executable} -m pytest {filter_str} -v --durations=0 -s --capture=no"
        cmd += f" --rootdir={self.pypto_root} {ext}"

        if self.check_pip_dependencies(deps={"pytest-xdist": ">=3.8.0"}, raise_err=False, log_err=False):
            cmd += " --no-loadscope-reorder"

        # 传入 update_env !!!
        _, duration = self.run_build_cmd(cmd=cmd, update_env=update_env, pg_desc="pytest")
        logging.info("pytest run success, %s", duration)


    def _py_tests_get_xsan_env(self) -> Dict[str, str]:
        update_env = {}
        if not (self.build.asan or self.build.ubsan):
            return update_env
        logging.warning("ASAN/UBSAN support in WHL package scenarios is experimental - use with caution.")

        py3_ver = sys.version_info
        dir_name = f"temp.linux-{self.get_system_processor()}-cpython-{py3_ver.major}{py3_ver.minor}"
        xsan_config_file = Path(self.build_root, dir_name, "_pypto_xsan_config.txt")
        if not xsan_config_file.exists():
            logging.warning("XSAN config file not found: %s", xsan_config_file)
            return update_env

        with open(xsan_config_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                update_env[k] = v
        for k, v in update_env.items():
            logging.info("%s=%s", k, v)
        return update_env


    def _get_pip_install_dist(self) -> Optional[Path]:
        # pip install -e 场景需直接安装到 site-packages 默认路径(与指定 --target 参数逻辑冲突), 其他场景安装到自定义目录
        return self.install_root

    def _get_setuptools_build_ext_config_setting(self) -> Tuple[str, str]:
        cmake_args = ""
        ascend_cann_package_path = (
            os.environ.get("ASCEND_CANN_PACKAGE_PATH")
            or os.environ.get("ASCEND_HOME_PATH")
            or (os.path.dirname(os.environ["ASCEND_OPP_PATH"]) if os.environ.get("ASCEND_OPP_PATH") else None)
        )
        if ascend_cann_package_path:
            cmake_args += f' -DASCEND_CANN_PACKAGE_PATH="{ascend_cann_package_path}"'
        
        env_setting = ""
        env_setting += f" --cmake-options=\"{cmake_args}\"" if cmake_args else ""
        env_setting += f" --backend-type={self.feature.backend_type}"
        
        multi_py3_exe_cfg = self.feature.multi_py3_exe_cfg
        if multi_py3_exe_cfg:
            env_setting += f" --multi-py3-exe={multi_py3_exe_cfg}"
        cmd_setting = ""
        if env_setting:
            cmd_setting = f" --config-setting=--build-option='build_ext {env_setting}'"
        return cmd_setting, env_setting

    def _get_setuptools_bdist_wheel_config_setting(self) -> str:
        cmd = f" bdist_wheel --plat-name={self.feature.whl_plat_name}" if self.feature.whl_plat_name else ""
        multi_py3_min_minor = self.feature.multi_py3_min_minor
        if multi_py3_min_minor:
            cmd += f" --py-limited-api=cp{sys.version_info.major}{multi_py3_min_minor}"
        cmd += f" build --build-base={self.build_root.name}"
        cmd += f" --parallel={self.build.job_num}" if self.build.job_num else ""
        _, ext = self._get_setuptools_build_ext_config_setting()
        if ext:
            cmd += f" build_ext {ext}"
        cmd = f" --config-setting=--build-option='{cmd}'"
        return cmd
    
    def _duration(self, ts: datetime) -> str:
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        duration_str = f"Duration {duration} secs"
        if self.remain_timeout:
            self.remain_timeout = max(self.remain_timeout - duration, 0)
            duration_str += f" Remain {self.remain_timeout} secs"
        return duration_str

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    BuildCtrl.main()

