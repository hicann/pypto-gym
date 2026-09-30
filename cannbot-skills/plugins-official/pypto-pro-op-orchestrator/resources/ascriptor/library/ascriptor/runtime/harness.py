# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The host test program of the aclnn launcher and the binary argument files it reads.

The program is generated once per kernel from the manifest and stays shape-agnostic: every run
writes ``input/args.txt`` (tensor shapes and scalar values), ``input/<name>.bin`` for the inputs,
and reads ``output/<name>.bin`` back (``<name>.<j>.bin`` per member of a list output). The program
creates ACL tensors from the shapes, calls the
custom op's ``aclnn<Op>GetWorkspaceSize`` / ``aclnn<Op>`` pair and writes the outputs; it is the
old ``test.cpp`` / ``tensorx.h`` pair without the values baked into the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .project import HostSpec, camel

ACL_DTYPE = {
    "f32": "ACL_FLOAT", "f16": "ACL_FLOAT16", "bf16": "ACL_BF16", "i8": "ACL_INT8", "u8": "ACL_UINT8",
    "i16": "ACL_INT16", "u16": "ACL_UINT16", "i32": "ACL_INT32", "u32": "ACL_UINT32", "i64": "ACL_INT64",
    "u64": "ACL_UINT64", "e4m3": "ACL_FLOAT8_E4M3FN", "e5m2": "ACL_FLOAT8_E5M2", "hif8": "ACL_HIFLOAT8", "b1": "ACL_BOOL",
    "c32": "ACL_COMPLEX32", "c64": "ACL_COMPLEX64",
    "e8m0": "ACL_UINT8",  # E8M0 planes travel as bytes, as Pro launches pass them
}
ESIZE = {"f32": 4, "f16": 2, "bf16": 2, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "i32": 4, "u32": 4, "i64": 8, "u64": 8,
         "e4m3": 1, "e5m2": 1, "hif8": 1, "e8m0": 1, "b1": 1, "fp4_e2m1": 1, "fp4_e1m2": 1, "c32": 4, "c64": 8}

HARNESS_HEADER = r'''#pragma once
// ascriptor aclnn harness support: a host/device tensor pair described by shape + ACL dtype.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <type_traits>
#include <sstream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define ASCRIP_CHECK(x)                                                                 \
    do {                                                                                \
        auto __ret = (x);                                                               \
        if (__ret != ACL_SUCCESS) {                                                     \
            std::cerr << __FILE__ << ":" << __LINE__ << " " #x " failed: " << __ret << std::endl; \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

struct HostTensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    aclDataType dtype = ACL_FLOAT;
    int64_t esize = 4;
    int64_t numel = 1;
    void* host = nullptr;
    void* device = nullptr;
    aclTensor* acl = nullptr;

    void init(const std::vector<int64_t>& s, aclDataType dt, int64_t es)
    {
        shape = s;
        dtype = dt;
        esize = es;
        numel = 1;
        for (auto d : shape) numel *= d;
        strides.assign(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
        int64_t bytes = numel * esize;
        host = std::calloc(bytes > 0 ? bytes : 1, 1);
        ASCRIP_CHECK(aclrtMalloc(&device, bytes > 0 ? bytes : 32, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    bool load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.read((char*)host, numel * esize);
        return (int64_t)f.gcount() == numel * esize;
    }
    void save(const std::string& path)
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*)host, numel * esize);
    }
    void to_device() { ASCRIP_CHECK(aclrtMemcpy(device, numel * esize, host, numel * esize, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void to_host() { ASCRIP_CHECK(aclrtMemcpy(host, numel * esize, device, numel * esize, ACL_MEMCPY_DEVICE_TO_HOST)); }
    aclTensor* tensor()
    {
        if (acl == nullptr) {
            acl = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), device);
        }
        return acl;
    }
    void release()
    {
        if (acl) aclDestroyTensor(acl);
        if (device) aclrtFree(device);
        std::free(host);
        acl = nullptr;
        device = nullptr;
        host = nullptr;
    }
};

// input/args.txt: "T name ndim d0 d1 ..." per tensor (inputs and outputs), "S name value" per scalar,
// "L name count" per tensor list (input or output) whose members are the tensors "name.0", "name.1", ...
struct Args {
    std::map<std::string, std::vector<int64_t>> shapes;
    std::map<std::string, std::string> scalars;
    std::map<std::string, int> lists;
    bool read(const std::string& path)
    {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::string kind, name;
            if (!(is >> kind >> name)) continue;
            if (kind == "T") {
                int n = 0;
                is >> n;
                std::vector<int64_t> s(n);
                for (int i = 0; i < n; ++i) is >> s[i];
                shapes[name] = s;
            } else if (kind == "S") {
                std::string v;
                is >> v;
                scalars[name] = v;
            } else if (kind == "L") {
                int n = 0;
                is >> n;
                lists[name] = n;
            }
        }
        return true;
    }
    int list(const std::string& name) const
    {
        auto it = lists.find(name);
        if (it == lists.end()) {
            std::cerr << "args.txt: no list " << name << std::endl;
            std::exit(2);
        }
        return it->second;
    }
    const std::vector<int64_t>& shape(const std::string& name) const
    {
        auto it = shapes.find(name);
        if (it == shapes.end()) {
            std::cerr << "args.txt: no shape for " << name << std::endl;
            std::exit(2);
        }
        return it->second;
    }
    template <typename T>
    T scalar(const std::string& name) const
    {
        auto it = scalars.find(name);
        if (it == scalars.end()) {
            std::cerr << "args.txt: no value for " << name << std::endl;
            std::exit(2);
        }
        if constexpr (std::is_integral<T>::value) return (T)std::stoll(it->second);
        return (T)std::strtod(it->second.c_str(), nullptr);
    }
};
'''


def harness_source(spec: HostSpec, *, seed_outputs: bool = False) -> str:
    op = camel(spec.kernel)
    L = ["#include <iostream>", "#include <string>", "#include <vector>", '#include "acl/acl.h"', '#include "ascrip_harness.h"',
         f'#include "aclnn_{spec.kernel}.h"', "", "int main(int argc, char** argv)", "{",
         '    std::string root = argc > 1 ? argv[1] : ".";',
         "    Args args;",
         '    if (!args.read(root + "/input/args.txt")) { std::cerr << "cannot read input/args.txt" << std::endl; return 2; }',
         "    int32_t deviceId = 0;", "    aclrtStream stream = nullptr;",
         "    ASCRIP_CHECK(aclInit(nullptr));", "    ASCRIP_CHECK(aclrtSetDevice(deviceId));", "    ASCRIP_CHECK(aclrtCreateStream(&stream));"]
    for p in spec.scalars:
        ct = "float" if p["dtype"] in ("f32", "f16", "bf16") else "int64_t"
        L.append(f'    const {ct} {p["name"]} = args.scalar<{ct}>("{p["name"]}");')
    for p in spec.tensors:
        L.append(f'    HostTensor {p["name"]};')
        L.append(f'    {p["name"]}.init(args.shape("{p["name"]}"), {ACL_DTYPE[p["dtype"]]}, {ESIZE[p["dtype"]]});')
        if not p["output"] or seed_outputs:
            L.append(f'    if (!{p["name"]}.load(root + "/input/{p["name"]}.bin")) {{ std::cerr << "cannot read input {p["name"]}" << std::endl; return 2; }}')
            L.append(f'    {p["name"]}.to_device();')
        else:
            L.append(f'    std::memset({p["name"]}.host, 0xFF, {p["name"]}.numel * {p["name"]}.esize);')
            L.append(f'    {p["name"]}.to_device();')
    for p in spec.lists:  # a gmlist parameter: one HostTensor per member, handed over as an aclTensorList; the members
        n = p["name"]     # of an output list are created from the shapes in args.txt and poisoned like a tensor output
        if p["output"] and not seed_outputs:
            fill = f'        std::memset({n}_members[j].host, 0xFF, {n}_members[j].numel * {n}_members[j].esize);'
        else:
            fill = f'        if (!{n}_members[j].load(root + "/input/" + member + ".bin")) {{ std::cerr << "cannot read input " << member << std::endl; return 2; }}'
        L += [f'    std::vector<HostTensor> {n}_members(args.list("{n}"));', f'    std::vector<aclTensor*> {n}_ptrs;',
              f'    for (int j = 0; j < (int){n}_members.size(); ++j) {{',
              f'        std::string member = "{n}." + std::to_string(j);',
              f'        {n}_members[j].init(args.shape(member), {ACL_DTYPE[p["dtype"]]}, {ESIZE[p["dtype"]]});',
              fill,
              f'        {n}_members[j].to_device();', f'        {n}_ptrs.push_back({n}_members[j].tensor());', '    }',
              f'    aclTensorList* {n} = aclCreateTensorList({n}_ptrs.data(), {n}_ptrs.size());']

    def handle(p: dict[str, Any]) -> str:
        return p["name"] if p["kind"] == "list" else f'{p["name"]}.tensor()'

    call = [handle(p) for p in spec.inputs] + [p["name"] for p in spec.scalars] + [handle(p) for p in spec.outputs]
    # ASCRIPTOR_REPEAT > 1 is the PERFORMANCE mode: the same launch, N times, so a profiler
    # collects N task records to take a median over. Every iteration after the first re-uploads
    # the inputs and re-poisons the outputs, so an accumulating kernel computes the same thing
    # each time and the outputs saved below are still the single-run outputs. An aclnn executor
    # is consumed by its call, so each repeat takes a fresh one (host-side work only - it does
    # not enter the device task the profiler times).
    refresh: list[str] = []
    for p in spec.tensors:
        if p["output"] and not seed_outputs:
            refresh.append(f'        std::memset({p["name"]}.host, 0xFF, {p["name"]}.numel * {p["name"]}.esize);')
        else:
            refresh.append(f'        if (!{p["name"]}.load(root + "/input/{p["name"]}.bin")) return 2;')
        refresh.append(f'        {p["name"]}.to_device();')
    for p in spec.lists:
        n = p["name"]
        refresh.append(f'        for (int j = 0; j < (int){n}_members.size(); ++j) {{')
        if p["output"] and not seed_outputs:
            refresh.append(f'            std::memset({n}_members[j].host, 0xFF, {n}_members[j].numel * {n}_members[j].esize);')
        else:
            refresh.append(f'            if (!{n}_members[j].load(root + "/input/{n}." + std::to_string(j) + ".bin")) return 2;')
        refresh += [f'            {n}_members[j].to_device();', '        }']
    L += ["    uint64_t workspaceSize = 0;", "    aclOpExecutor* executor = nullptr;", "    void* workspaceAddr = nullptr;",
          f"    ASCRIP_CHECK(aclnn{op}GetWorkspaceSize({', '.join(call)}, &workspaceSize, &executor));",
          "    if (workspaceSize > 0) { ASCRIP_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST)); }",
          '    const char* ascripRepEnv = std::getenv("ASCRIPTOR_REPEAT");',
          "    int ascripReps = ascripRepEnv ? std::atoi(ascripRepEnv) : 1;",
          "    if (ascripReps < 1) ascripReps = 1;",
          "    for (int ascripR = 0; ascripR < ascripReps; ++ascripR) {",
          "        if (ascripR) {",
          *refresh,
          "            uint64_t ws2 = 0;",
          "            aclOpExecutor* ex2 = nullptr;",
          f"            ASCRIP_CHECK(aclnn{op}GetWorkspaceSize({', '.join(call)}, &ws2, &ex2));",
          '            if (ws2 > workspaceSize) { std::cerr << "repeat: workspace grew" << std::endl; return 2; }',
          "            executor = ex2;",
          "        }",
          f"        ASCRIP_CHECK(aclnn{op}(workspaceAddr, workspaceSize, executor, stream));",
          "        ASCRIP_CHECK(aclrtSynchronizeStream(stream));",
          "    }"]
    for p in spec.outputs:
        n = p["name"]
        if p["kind"] == "list":
            L += [f'    for (int j = 0; j < (int){n}_members.size(); ++j) {{', f'        {n}_members[j].to_host();',
                  f'        {n}_members[j].save(root + "/output/{n}." + std::to_string(j) + ".bin");', '    }']
        else:
            L += [f'    {n}.to_host();', f'    {n}.save(root + "/output/{n}.bin");']
    for p in spec.tensors:
        L.append(f'    {p["name"]}.release();')
    for p in spec.lists:  # aclDestroyTensorList destroys the member aclTensors; the HostTensors keep only the memory
        L += [f'    aclDestroyTensorList({p["name"]});', f'    for (auto& m : {p["name"]}_members) {{ m.acl = nullptr; m.release(); }}']
    L += ["    if (workspaceAddr) aclrtFree(workspaceAddr);", "    aclrtDestroyStream(stream);", "    aclrtResetDevice(deviceId);",
          "    aclFinalize();", '    std::cout << "ASCRIP_HARNESS_OK" << std::endl;', "    return 0;", "}", ""]
    return "\n".join(L)


def write_harness(spec: HostSpec, test_dir: Path, *, seed_outputs: bool = False) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "input").mkdir(exist_ok=True)
    (test_dir / "output").mkdir(exist_ok=True)
    (test_dir / "ascrip_harness.h").write_text(HARNESS_HEADER, encoding="utf-8")
    (test_dir / "test.cpp").write_text(harness_source(spec, seed_outputs=seed_outputs), encoding="utf-8")


# ---------------------------------------------------------------- argument files

def write_args(spec: HostSpec, test_dir: Path, tensors: dict[str, Any], scalars: dict[str, Any], *, seed_outputs: bool = False) -> None:
    """``tensors``: name -> numpy array in the tensor's own dtype, or ``(logical_shape, byte_array)`` for dtypes
    numpy has no type for (bf16, fp8, ...: the bytes travel in a uint8 array, the shape is the tensor's).
    Initialized outputs are serialized only when seed_outputs is requested."""
    import numpy as np

    lines = []
    for p in spec.tensors:
        shape, _ = _shape_and_array(tensors[p["name"]])
        lines.append(f"T {p['name']} {len(shape)} " + " ".join(str(int(d)) for d in shape))
    for p in spec.lists:
        members = list(tensors[p["name"]])
        lines.append(f"L {p['name']} {len(members)}")
        for j, m in enumerate(members):
            shape, a = _shape_and_array(m)
            lines.append(f"T {p['name']}.{j} {len(shape)} " + " ".join(str(int(d)) for d in shape))
            if not p["output"] or seed_outputs:
                (test_dir / "input" / f"{p['name']}.{j}.bin").write_bytes(np.ascontiguousarray(a).tobytes())
    for p in spec.scalars:
        value = scalars[p['name']]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            value = int(value)
        lines.append(f"S {p['name']} {value!r}")
    (test_dir / "input" / "args.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in (spec.inputs + spec.outputs if seed_outputs else spec.inputs):
        if p["kind"] == "list":
            continue
        _, a = _shape_and_array(tensors[p["name"]])
        (test_dir / "input" / f"{p['name']}.bin").write_bytes(np.ascontiguousarray(a).tobytes())


def _shape_and_array(v: Any) -> tuple[list[int], Any]:
    if isinstance(v, tuple) and len(v) == 2:
        return list(v[0]), v[1]
    if hasattr(v, "shape"):
        return list(v.shape), v
    return list(v), None


def read_output(test_dir: Path, name: str, nbytes: int) -> bytes:
    data = (test_dir / "output" / f"{name}.bin").read_bytes()
    if len(data) < nbytes:
        raise RuntimeError(f"output {name}: {len(data)} bytes, expected {nbytes}")
    return data[:nbytes]


def read_outputs(test_dir: Path, spec: HostSpec, tensors: dict[str, Any]) -> dict[str, Any]:
    """Every output of ``spec`` as the harness wrote it, sized by the torch tensors the caller bound: the bytes of
    ``output/name.bin`` for a tensor, the list of the members' ``output/name.j.bin`` for a list output."""
    def nbytes(t: Any) -> int:
        return t.numel() * t.element_size()

    out: dict[str, Any] = {}
    for p in spec.outputs:
        if p["kind"] == "list":
            out[p["name"]] = [read_output(test_dir, f"{p['name']}.{j}", nbytes(m)) for j, m in enumerate(tensors[p["name"]])]
        else:
            out[p["name"]] = read_output(test_dir, p["name"], nbytes(tensors[p["name"]]))
    return out


__all__ = ["ACL_DTYPE", "ESIZE", "HARNESS_HEADER", "harness_source", "write_harness", "write_args", "read_output", "read_outputs"]
