# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the public delivery wrapper on an NPU without importing ascriptor."""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


class NoAscriptor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ascriptor" or fullname.startswith("ascriptor."):
            raise ImportError("standalone delivery must not depend on ascriptor")


sys.meta_path.insert(0, NoAscriptor())
if any(name == "ascriptor" or name.startswith("ascriptor.") for name in sys.modules):
    raise RuntimeError("standalone verification requires a process without preloaded ascriptor modules")


def main():
    import torch
    import torch_npu  # noqa: F401
    from generated.launch import normalize_outputs, decode_params
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    request = json.loads((root / "input.json").read_text())
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    tensors = {}
    for item in request["inputs"]:
        if item.get("kind") == "list":
            members = []
            for member in item["members"]:
                raw = bytearray((root / member["file"]).read_bytes())
                members.append(torch.frombuffer(raw, dtype=getattr(torch, item["dtype"])).reshape(member["shape"]))
            tensors[item["name"]] = members
            continue
        raw = bytearray((root / item["file"]).read_bytes())
        tensors[item["name"]] = torch.frombuffer(raw, dtype=getattr(torch, item["dtype"])).reshape(item["shape"])
    spec = importlib.util.spec_from_file_location("delivery", root / request["wrapper_file"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, request["wrapper"])
    outputs = None
    for _ in range(request["repeat"]):
        inputs = []
        for name in request["input_names"]:
            value = tensors[name]
            if isinstance(value, list):
                inputs.append([member.clone().to(device) for member in value])
            else:
                inputs.append(value.clone().to(device))
        outputs = function(*inputs, **decode_params(request["params"]))
        torch.npu.synchronize()
    declared = request.get("outputs", [{"name": name} for name in request["output_names"]])
    values = normalize_outputs(outputs, declared)
    directory = root / "output"
    directory.mkdir(exist_ok=True)
    metadata = {}
    def write_tensor(name, tensor):
        value = tensor.detach().cpu().contiguous()
        filename = f"{name}.bin"
        (directory / filename).write_bytes(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."), "file": filename}
    for item in declared:
        name, value = item["name"], values[item["name"]]
        if item.get("is_list", False):
            if not isinstance(value, (list, tuple)) or not value:
                raise AssertionError(f"expected a non-empty TensorList output: {name}")
            metadata[name] = {"kind": "list", "members": [write_tensor(f"{name}.{i}", t) for i, t in enumerate(value)]}
        else:
            metadata[name] = write_tensor(name, value)
    (directory / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
    imported = any(name == "ascriptor" or name.startswith("ascriptor.") for name in sys.modules)
    if imported:
        raise AssertionError("standalone execution imported ascriptor")
    print(json.dumps({"standalone": True, "runtime": "pypto_pro", "repeat": request["repeat"],
                      "outputs": metadata, "ascriptor_imported": False}))


if __name__ == "__main__":
    main()
