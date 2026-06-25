#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Apply explicit PyPTO runtime patches for local HF model snapshots."""

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("runtime_patch")


@dataclass(frozen=True)
class PatchSpec:
    family: str
    config_auto_map: dict
    files_to_copy: tuple
    cache_needles: tuple


PRESET_SPECS = {
    "gemma4_31b_it": PatchSpec(
        family="gemma4",
        config_auto_map={
            "AutoConfig": "configuration_gemma4.Gemma4Config",
            "AutoModelForImageTextToText": "modeling_gemma4.Gemma4ForConditionalGeneration",
        },
        files_to_copy=(
            ("modeling_gemma4.py", "src/pypto_gym/transformers/gemma4_31b_it/modeling_gemma4.py"),
            ("configuration_gemma4.py", "src/pypto_gym/transformers/gemma4_31b_it/configuration_gemma4.py"),
        ),
        cache_needles=("gemma", "gemma4"),
    ),
    "llada2_moe": PatchSpec(
        family="llada2",
        config_auto_map={
            "AutoConfig": "configuration_llada2_moe.LLaDA2MoeConfig",
            "AutoModelForCausalLM": "modeling_llada2_moe.LLaDA2MoeModelLM",
        },
        files_to_copy=(
            ("modeling_llada2_moe.py", "src/pypto_gym/transformers/llada2_moe/modeling_llada2_moe.py"),
            (
                "configuration_llada2_moe.py",
                "src/pypto_gym/transformers/llada2_moe/configuration_llada2_moe.py",
            ),
        ),
        cache_needles=("llada", "llada2"),
    ),
    "minimax_m27": PatchSpec(
        family="minimax_m27",
        config_auto_map={
            "AutoConfig": "configuration_minimax_m27.MiniMaxM27Config",
            "AutoModel": "modeling_minimax_m27.MiniMaxM2Model",
            "AutoModelForCausalLM": "modeling_minimax_m27.MiniMaxM2ForCausalLM",
        },
        files_to_copy=(
            ("modeling_minimax_m27.py", "src/pypto_gym/transformers/minimax_m27/modeling_minimax_m27.py"),
            (
                "configuration_minimax_m27.py",
                "src/pypto_gym/transformers/minimax_m27/configuration_minimax_m27.py",
            ),
        ),
        cache_needles=("minimax", "m27", "m2"),
    ),
    "minimax_m3": PatchSpec(
        family="minimax_m3",
        config_auto_map={
            "AutoConfig": "configuration_minimax_m3.MiniMaxM3Config",
            "AutoModel": "modeling_minimax_m3.MiniMaxM3Model",
            "AutoModelForCausalLM": "modeling_minimax_m3.MiniMaxM3ForCausalLM",
        },
        files_to_copy=(
            ("modeling_minimax_m3.py", "src/pypto_gym/transformers/minimax_m3/modeling_minimax_m3.py"),
            (
                "configuration_minimax_m3.py",
                "src/pypto_gym/transformers/minimax_m3/configuration_minimax_m3.py",
            ),
        ),
        cache_needles=("minimax", "m3"),
    ),
}


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_source_root(source_root):
    if source_root:
        return os.path.abspath(source_root)
    return repo_root()


def require_local_model_path(model_path):
    if not os.path.isdir(model_path):
        raise ValueError(
            f"MODEL_PATH must be a local directory: {model_path}. "
            "Download the HuggingFace model first with modeling/transformers/download_hf_model.py."
        )


def clear_hf_module_cache(*needles):
    cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if not os.path.isdir(cache_root):
        return
    lower_needles = tuple(item.lower() for item in needles)
    for name in os.listdir(cache_root):
        if any(item in name.lower() for item in lower_needles):
            shutil.rmtree(os.path.join(cache_root, name), ignore_errors=True)


def prepare_output_dir(output_dir, force):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        if force:
            shutil.rmtree(output_dir)
        elif os.listdir(output_dir):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def require_patch_sources(root_dir, spec):
    missing = []
    for _, src_relpath in spec.files_to_copy:
        src_path = os.path.join(root_dir, src_relpath)
        if not os.path.isfile(src_path):
            missing.append(src_relpath)
    if missing:
        raise FileNotFoundError(
            f"Missing PyPTO patch source file(s) for {spec.family}: {', '.join(missing)}"
        )


def create_patch_overlay(model_path, spec, output_dir, force=False, source_root=None):
    require_local_model_path(model_path)
    root_dir = resolve_source_root(source_root)
    require_patch_sources(root_dir, spec)
    overlay_dir = prepare_output_dir(output_dir, force)

    skipped = {"config.json"} | {dst_name for dst_name, _ in spec.files_to_copy}
    for name in os.listdir(model_path):
        if name in skipped:
            continue
        os.symlink(os.path.join(model_path, name), os.path.join(overlay_dir, name))

    with open(os.path.join(model_path, "config.json"), "r") as config_file:
        config = json.load(config_file)
    config["auto_map"] = spec.config_auto_map
    with open(os.path.join(overlay_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)

    for dst_name, src_relpath in spec.files_to_copy:
        shutil.copy2(os.path.join(root_dir, src_relpath), os.path.join(overlay_dir, dst_name))

    clear_hf_module_cache(*spec.cache_needles)
    logger.info("[PyPTO] Created patched modeling overlay at %s", overlay_dir)
    return overlay_dir


def backup_once(path):
    if os.path.exists(path):
        backup_path = path + ".pypto_orig"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)


def patch_model_dir(model_path, spec, source_root=None):
    require_local_model_path(model_path)
    root_dir = resolve_source_root(source_root)
    require_patch_sources(root_dir, spec)
    config_path = os.path.join(model_path, "config.json")
    backup_once(config_path)
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    config["auto_map"] = spec.config_auto_map
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)

    for dst_name, src_relpath in spec.files_to_copy:
        dst_path = os.path.join(model_path, dst_name)
        backup_once(dst_path)
        shutil.copy2(os.path.join(root_dir, src_relpath), dst_path)

    clear_hf_module_cache(*spec.cache_needles)
    logger.info("[PyPTO] Patched local model directory at %s", model_path)
    return model_path


def split_key_value(raw_value, option_name):
    if "=" not in raw_value:
        raise ValueError(f"{option_name} must use KEY=VALUE syntax: {raw_value}")
    key, value = raw_value.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"{option_name} must use non-empty KEY=VALUE syntax: {raw_value}")
    return key, value


def parse_key_values(raw_values, option_name):
    parsed = {}
    for raw_value in raw_values or ():
        key, value = split_key_value(raw_value, option_name)
        parsed[key] = value
    return parsed


def parse_copy_specs(raw_values):
    return tuple(split_key_value(raw_value, "--copy") for raw_value in raw_values or ())


def has_custom_spec_args(args):
    return bool(args.auto_map or args.copy_files or args.family_name or args.cache_needle)


def build_custom_spec(args):
    config_auto_map = parse_key_values(args.auto_map, "--auto-map")
    files_to_copy = parse_copy_specs(args.copy_files)
    if not config_auto_map:
        raise ValueError("Custom patch mode requires at least one --auto-map KEY=VALUE")
    if not files_to_copy:
        raise ValueError("Custom patch mode requires at least one --copy DEST=SOURCE")
    return PatchSpec(
        family=args.family_name or "custom",
        config_auto_map=config_auto_map,
        files_to_copy=files_to_copy,
        cache_needles=tuple(args.cache_needle or ()),
    )


def build_patch_spec(args):
    if args.model_family:
        if has_custom_spec_args(args):
            raise ValueError("Use either --model-family or custom --auto-map/--copy options, not both")
        return PRESET_SPECS[args.model_family]
    return build_custom_spec(args)


def apply_runtime_patch(args, spec):
    if args.output_dir:
        return create_patch_overlay(
            args.model_path,
            spec,
            args.output_dir,
            force=args.force,
            source_root=args.source_root,
        )
    return patch_model_dir(args.model_path, spec, source_root=args.source_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a local PyPTO runtime patch")
    parser.add_argument("--model-family", choices=tuple(sorted(PRESET_SPECS)),
                        help="Known model family preset to patch")
    parser.add_argument("--model-path", required=True, help="Local model snapshot directory")
    parser.add_argument("--output-dir", default=None,
                        help="Optional patched overlay output directory")
    parser.add_argument("--source-root", default=None,
                        help="Optional pypto-gym checkout that provides patch source files")
    parser.add_argument("--family-name", default=None,
                        help="Custom family name used in diagnostics")
    parser.add_argument("--auto-map", action="append", default=None,
                        help="Custom auto_map entry, for example AutoConfig=configuration_x.Config")
    parser.add_argument("--copy", dest="copy_files", action="append", default=None,
                        help="Custom file copy entry, DEST=SOURCE relative to --source-root")
    parser.add_argument("--cache-needle", action="append", default=None,
                        help="Custom HuggingFace module cache name fragment to clear")
    parser.add_argument("--force", action="store_true", help="Replace a non-empty output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = build_patch_spec(args)
    patched_path = apply_runtime_patch(args, spec)
    logger.info("export MODEL_PATH=%s", patched_path)


if __name__ == "__main__":
    main()
