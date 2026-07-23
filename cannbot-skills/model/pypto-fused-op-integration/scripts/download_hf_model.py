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
"""Download a HuggingFace model card into a local MODEL_PATH directory."""

import argparse
import importlib
import inspect
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_hf_model")


def parse_args():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model snapshot")
    parser.add_argument("--model-id", required=True, help="HuggingFace model card id")
    parser.add_argument("--output-dir", required=True, help="Local model directory")
    parser.add_argument("--revision", default=None, help="Optional model revision")
    parser.add_argument("--token", default=None, help="Optional HuggingFace token")
    parser.add_argument("--allow-pattern", action="append", default=None,
                        help="Pattern to include; can be passed multiple times")
    parser.add_argument("--ignore-pattern", action="append", default=None,
                        help="Pattern to exclude; can be passed multiple times")
    return parser.parse_args()


def download_model(args):
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download model snapshots.") from exc
    snapshot_download = huggingface_hub.snapshot_download

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    kwargs = {
        "repo_id": args.model_id,
        "local_dir": output_dir,
        "revision": args.revision,
        "token": args.token,
        "allow_patterns": args.allow_pattern,
        "ignore_patterns": args.ignore_pattern,
    }
    if args.token and "token" not in inspect.signature(snapshot_download).parameters:
        kwargs.pop("token")
        kwargs["use_auth_token"] = args.token
    snapshot_path = snapshot_download(**{key: val for key, val in kwargs.items() if val is not None})
    return os.path.abspath(snapshot_path)


def main():
    args = parse_args()
    local_path = download_model(args)
    logger.info("Downloaded %s to %s", args.model_id, local_path)
    logger.info("export MODEL_PATH=%s", local_path)


if __name__ == "__main__":
    main()
