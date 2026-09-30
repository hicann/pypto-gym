# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The aclnn custom-op project generator (D-013: a template directory instead of the old tars).

``generate_project(artifacts, out_dir, ...)`` lays out a CANN custom-op project the way the old
``OpExec`` did — ``op_kernel/`` holds the cce artifact, ``op_host/`` the tiling / infer-shape / OpDef
file the manifest describes, ``CMakePresets.json`` the CANN path and compute unit — and returns the
project directory. Everything the host side needs is in ``manifest.json`` (parameter kinds, dtypes,
dims, outputs, workspace sizes as expressions over the scalar parameters), so this module never
looks at the IR.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..backends.base import Artifacts

# The vendored CANN custom-op template, trimmed to what its build reaches: the replay code generator, the
# AICPU json helpers, makeself's own man page / README / Makefile and the empty util/__init__.py are on no
# path from build.sh. A build with and without them installs the same artifacts byte for byte, so they are
# not carried; take them from the CANN sample again if a future preset needs them.
TEMPLATE = Path(__file__).with_name("aclnn") / "template"

GE_DTYPE = {
    "f32": "ge::DT_FLOAT", "f16": "ge::DT_FLOAT16", "bf16": "ge::DT_BF16",
    "i8": "ge::DT_INT8", "u8": "ge::DT_UINT8", "i16": "ge::DT_INT16", "u16": "ge::DT_UINT16",
    "i32": "ge::DT_INT32", "u32": "ge::DT_UINT32", "i64": "ge::DT_INT64", "u64": "ge::DT_UINT64",
    "e4m3": "ge::DT_FLOAT8_E4M3FN", "e5m2": "ge::DT_FLOAT8_E5M2", "hif8": "ge::DT_HIFLOAT8", "b1": "ge::DT_BOOL",
    "c32": "ge::DT_COMPLEX32", "c64": "ge::DT_COMPLEX64",
    "e8m0": "ge::DT_UINT8",  # byte carrier, as harness.ACL_DTYPE
}
# tiling field type and OpDef attribute kind per scalar dtype
SCALAR_C = {"i32": "int32_t", "i64": "int64_t", "u32": "uint32_t", "f32": "float", "f16": "float", "b1": "int32_t",
            "i8": "int32_t", "u8": "int32_t", "i16": "int32_t", "u16": "int32_t"}
STORAGE_C = {"f32": "float", "f16": "uint16_t", "bf16": "uint16_t", "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t",
             "u16": "uint16_t", "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
             "e4m3": "uint8_t", "e5m2": "uint8_t", "hif8": "uint8_t", "e8m0": "uint8_t", "b1": "uint8_t",
             "fp4_e2m1": "uint8_t", "fp4_e1m2": "uint8_t"}

# The dynamic UB the runtime hands SIMT launches on a5 vector cores (the old OpExec's SIMT_UB_CAP_KB).
from ..devices import SIMT_UB_CAP_KB  # noqa: E402


def declares_dyn_ub(device: str) -> bool:
    """Whether this family's host declares a dynamic UB size at all.

    ``SIMT_UB_CAP_KB`` is a c310 number: 216 KiB of an a5 vector core's 256 KiB. The c220 family's
    whole UB is 192 KiB, so declaring it there asked the runtime for 24 KiB the core does not have,
    on every a2/a3 launch that is not pure cube. Those launches need no declaration.
    """
    from .. import devices as _devices

    return _devices.load(device).arch == "c310"


def camel(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def optype_snake(op_type: str) -> str:
    """CANN's own op-type -> file / kernel name rule (``cmake/util/ascendc_impl_build.py::optype_snake``): an
    underscore before every capital, then lowercase — so digits never start a part."""
    s = op_type[:1].lower() + op_type[1:]
    return re.sub(r"([A-Z])", r"_\1", s).lower()


def _is_float_scalar(dtype: str) -> bool:
    return dtype in ("f32", "f16", "bf16")


class HostSpec:
    """What the host side needs, read from the artifact manifest."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.m = manifest
        self.kernel = str(manifest["kernel"])
        self.op = camel(self.kernel)
        self.tensors = [p for p in manifest["params"] if p["kind"] == "tensor"]
        self.lists = [p for p in manifest["params"] if p["kind"] == "list"]  # gmlist parameters: DYNAMIC inputs / outputs
        self.scalars = [p for p in manifest["params"] if p["kind"] == "scalar"]
        self.inputs = [p for p in manifest["params"] if p["kind"] in ("tensor", "list") and not p["output"]]
        self.outputs = [p for p in manifest["params"] if p["kind"] in ("tensor", "list") and p["output"]]
        self.workspaces = list(manifest.get("workspaces", []))
        self.mode = str(manifest.get("mode", "mix"))
        self.block_dim = manifest.get("block_dim")
        self.exported_block_dim = manifest.get("exported_block_dim")  # an imported kernel's fixed launch (RFC-0015)
        self.exported_from = manifest.get("exported_from")
        self.device = str(manifest.get("device", "950"))
        for p in self.scalars:
            if p["dtype"] not in SCALAR_C:
                raise ValueError(f"scalar parameter {p['name']}: dtype {p['dtype']} cannot be a tiling attribute")
        for p in self.tensors + self.lists:
            if p["dtype"] not in GE_DTYPE:
                raise ValueError(f"tensor parameter {p['name']}: dtype {p['dtype']} has no GE data type")

    def scalar_names(self) -> set[str]:
        return {p["name"] for p in self.scalars}

    def dim_expr(self, d: Any, where: str) -> str:
        if isinstance(d, int):
            return str(d)
        if isinstance(d, str):
            for p in self.scalars:
                if d == p.get("ir_name", p["name"]):
                    return p["name"]
        raise ValueError(f"{where}: dimension {d!r} is not a literal or a scalar parameter (shape symbols must be "
                         "passed explicitly for the aclnn launcher)")

    def workspace_bytes_expr(self) -> str:
        if not self.workspaces:
            return "0"
        ends = []
        for w in self.workspaces:
            esize = {"fp4_e2m1": 1, "fp4_e1m2": 1}.get(w["dtype"])
            if esize is None:
                esize = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "float": 4, "uint32_t": 4, "int32_t": 4,
                         "int64_t": 8, "uint64_t": 8}[STORAGE_C[w["dtype"]]]
            numel = w["numel"] if isinstance(w["numel"], int) else f"({w['numel']})"
            offset = w["offset"] if isinstance(w["offset"], int) else f"({w['offset']})"
            ends.append(f"((size_t)({offset}) + (size_t)({numel}) * {esize})")
        expr = ends[0]
        for e in ends[1:]:
            expr = f"std::max<size_t>({expr}, {e})"
        return expr


PAD_FIELD = "ascriptor_pad"  # the one tiling field of a kernel without scalar parameters


def tiling_header(spec: HostSpec) -> str:
    lines = ['#include "register/tilingdata_base.h"', "", "namespace optiling {", f"BEGIN_TILING_DATA_DEF({spec.op}TilingData)"]
    for p in spec.scalars:
        lines.append(f"  TILING_DATA_FIELD_DEF({SCALAR_C[p['dtype']]}, {p['name']});")
    if not spec.scalars:  # CANN's tiling-def parser rejects an empty struct: one padding field for a kernel without scalars
        lines.append(f"  TILING_DATA_FIELD_DEF(int32_t, {PAD_FIELD});")
    lines += ["END_TILING_DATA_DEF;", "", f"REGISTER_TILING_DATA_CLASS({spec.op}, {spec.op}TilingData)", "}", ""]
    return "\n".join(lines)


def host_source(spec: HostSpec, compute_unit: str, block_dim: int | None) -> str:
    op, tiling = spec.op, f"{spec.op}TilingData"
    core_num = "GetCoreNumAiv" if spec.mode == "vec" else "GetCoreNumAic"
    from .launch_config import require_block_dim

    block_dim = require_block_dim(spec.exported_block_dim, block_dim, "the host source (SetBlockDim)", kernel=spec.kernel,
                                  source=spec.exported_from)
    bd = block_dim if block_dim is not None else spec.block_dim
    L = [f'#include "{spec.kernel}_tiling.h"', '#include "register/op_def_registry.h"', '#include "tiling/tiling_api.h"',
         "#include <algorithm>", "", "namespace optiling {",
         "static ge::graphStatus TilingFunc(gert::TilingContext* context)", "{",
         "    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());",
         f"    uint32_t coreNum = ascendcPlatform.{core_num}();", f"    {tiling} tiling;", "    auto attrs = context->GetAttrs();"]
    for i, p in enumerate(spec.scalars):
        ct = SCALAR_C[p["dtype"]]
        if _is_float_scalar(p["dtype"]):
            L.append(f"    const {ct} {p['name']} = *attrs->GetAttrPointer<float>({i});")
        else:
            L.append(f"    const {ct} {p['name']} = ({ct})*attrs->GetAttrPointer<int64_t>({i});")
        L.append(f"    tiling.set_{p['name']}({p['name']});")
    if not spec.scalars:
        L.append(f"    tiling.set_{PAD_FIELD}(0);")
    L.append(f"    context->SetBlockDim({bd if bd is not None else 'coreNum'});")
    if spec.mode != "cube" and declares_dyn_ub(spec.device):
        L.append(f"    context->SetDynUBufSize({SIMT_UB_CAP_KB * 1024});")
    L += ["    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());",
          "    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());",
          f"    size_t userWorkspaceSize = {spec.workspace_bytes_expr()};",
          "    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());",
          "    size_t* currentWorkspace = context->GetWorkspaceSizes(1);",
          "    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;",
          "    (void)coreNum;", "    return ge::GRAPH_SUCCESS;", "}", "}", "", "namespace ge {"]
    dyn = any(p["kind"] == "list" for p in spec.outputs)
    if dyn:  # the output index of InferShape / InferDataType is the instance index: a DYNAMIC output's members shift it
        L += ["// the instance index of the first member of IR output ir_index (an output after a DYNAMIC one is shifted",
              "// by that one's member count) and the member count of ir_index",
              "static size_t OutputInstance(const gert::ExtendedKernelContext* context, size_t ir_index)", "{",
              "    const gert::AnchorInstanceInfo* info = context->GetIrOutputInstanceInfo(ir_index);",
              "    return info != nullptr ? info->GetInstanceStart() : ir_index;", "}",
              "static size_t OutputInstances(const gert::ExtendedKernelContext* context, size_t ir_index)", "{",
              "    const gert::AnchorInstanceInfo* info = context->GetIrOutputInstanceInfo(ir_index);",
              "    return info != nullptr ? info->GetInstanceNum() : 1;", "}", ""]
    L += ["static ge::graphStatus InferShape(gert::InferShapeContext* context)", "{", "    auto attrs = context->GetAttrs();"]
    for i, p in enumerate(spec.scalars):
        ct = SCALAR_C[p["dtype"]]
        if _is_float_scalar(p["dtype"]):
            L.append(f"    const {ct} {p['name']} = *attrs->GetAttrPointer<float>({i});")
        else:
            L.append(f"    const {ct} {p['name']} = ({ct})*attrs->GetAttrPointer<int64_t>({i});")
        L.append(f"    (void){p['name']};")
    for k, p in enumerate(spec.outputs):
        if p["kind"] == "list":  # a DYNAMIC output: the members keep the shapes of the caller's tensors (D-060)
            L.append(f"    // {p['name']}: DYNAMIC output, its members' shapes are the caller's")
            continue
        idx = f"OutputInstance(context, {k})" if dyn else str(k)
        L.append(f"    gert::Shape* {p['name']}_shape = context->GetOutputShape({idx});")
        L.append(f"    {p['name']}_shape->SetDimNum(0);")
        for d in p["dims"]:
            L.append(f"    {p['name']}_shape->AppendDim({spec.dim_expr(d, p['name'])});")
    L += ["    return GRAPH_SUCCESS;", "}", "", "static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)", "{"]
    for k, p in enumerate(spec.outputs):
        if p["kind"] == "list":
            L += [f"    for (size_t j = 0; j < OutputInstances(context, {k}); ++j) {{",
                  f"        context->SetOutputDataType(OutputInstance(context, {k}) + j, {GE_DTYPE[p['dtype']]});", "    }"]
        else:
            L.append(f"    context->SetOutputDataType({f'OutputInstance(context, {k})' if dyn else k}, {GE_DTYPE[p['dtype']]});")
    L += ["    return GRAPH_SUCCESS;", "}", "}", "", "namespace ops {", f"class {op} : public OpDef {{", "public:",
          f"    explicit {op}(const char* name) : OpDef(name)", "    {"]
    for p in spec.inputs:
        L += [f'        this->Input("{p["name"]}")', f"            .ParamType({'DYNAMIC' if p['kind'] == 'list' else 'REQUIRED'})", f"            .DataType({{{GE_DTYPE[p['dtype']]}}})",
              "            .Format({ge::FORMAT_ND})", "            .UnknownShapeFormat({ge::FORMAT_ND});"]
    for p in spec.outputs:
        L += [f'        this->Output("{p["name"]}")', f"            .ParamType({'DYNAMIC' if p['kind'] == 'list' else 'REQUIRED'})", f"            .DataType({{{GE_DTYPE[p['dtype']]}}})",
              "            .Format({ge::FORMAT_ND})", "            .UnknownShapeFormat({ge::FORMAT_ND});"]
    for p in spec.scalars:
        kind = "Float(0)" if _is_float_scalar(p["dtype"]) else "Int(0)"
        L += [f'        this->Attr("{p["name"]}")', "            .AttrType(REQUIRED)", f"            .{kind};"]
    L += ["        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);", "",
          "        this->AICore().SetTiling(optiling::TilingFunc);", "        OpAICoreConfig aicoreConfig;",
          f'        this->AICore().AddConfig("{compute_unit}", aicoreConfig);', "    }", "};", "", f"OP_ADD({op});", "}", ""]
    return "\n".join(L)


def presets(cann_path: str, compute_unit: str) -> str:
    data = json.loads((TEMPLATE / "CMakePresets.json").read_text(encoding="utf-8"))
    for preset in data.get("configurePresets", []):
        cv = preset.get("cacheVariables", {})
        cv["ASCEND_CANN_PACKAGE_PATH"]["value"] = cann_path
        cv["ASCEND_COMPUTE_UNIT"]["value"] = compute_unit
    return json.dumps(data, indent=4) + "\n"


def generate_project(artifacts: Artifacts, out_dir: Path, *, cann_path: str, compute_unit: str,
                     block_dim: int | None = None) -> Path:
    """Write the custom-op project for ``artifacts`` under ``out_dir`` (created / refreshed) and return it."""
    out_dir = Path(out_dir)
    spec = HostSpec(artifacts.metadata)
    host = host_source(spec, compute_unit, block_dim)  # refuses a launch an imported kernel was not exported for, first
    if not (out_dir / "build.sh").is_file():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(TEMPLATE, out_dir)
    kernel_dir = out_dir / "op_kernel"
    host_dir = out_dir / "op_host"
    for d in (kernel_dir, host_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            if old.name != "CMakeLists.txt":
                old.unlink()
    for name, data in artifacts.files.items():
        if name == "manifest.json":
            continue
        (kernel_dir / name).write_bytes(data)
    (host_dir / f"{spec.kernel}_tiling.h").write_text(tiling_header(spec), encoding="utf-8")
    (host_dir / f"{spec.kernel}.cpp").write_text(host, encoding="utf-8")
    (out_dir / "CMakePresets.json").write_text(presets(cann_path, compute_unit), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(artifacts.metadata, indent=1) + "\n", encoding="utf-8")
    return out_dir


__all__ = ["HostSpec", "TEMPLATE", "GE_DTYPE", "SCALAR_C", "STORAGE_C", "SIMT_UB_CAP_KB", "camel", "declares_dyn_ub",
           "generate_project", "tiling_header", "host_source", "presets"]
