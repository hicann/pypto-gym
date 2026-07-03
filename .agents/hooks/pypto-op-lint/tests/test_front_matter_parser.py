#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from .helpers import load_lint_module


def test_parse_front_matter_returns_empty_when_missing():
    mod = load_lint_module()
    meta, body = mod._parse_front_matter("# SPEC\n")  # noqa: G.CLS.11
    assert not meta
    assert body == "# SPEC\n"


def test_parse_front_matter_extracts_values():
    mod = load_lint_module()
    text = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
---
# SPEC
"""
    meta, body = mod._parse_front_matter(text)  # noqa: G.CLS.11
    assert meta["op_name"] == "demo"
    assert meta["supported_dtypes"] == ["bfloat16"]
    assert body.startswith("# SPEC")


def test_validate_doc_schema_requires_fields_by_doc_type():
    mod = load_lint_module()
    spec_errors = mod._validate_doc_schema("SPEC", {"op_name": "x"})  # noqa: G.CLS.11
    design_errors = mod._validate_doc_schema("DESIGN", {"op_name": "x"})  # noqa: G.CLS.11
    api_errors = mod._validate_doc_schema("API_REPORT", {})  # noqa: G.CLS.11
    assert any("supported_dtypes" in err for err in spec_errors)
    assert any("dynamic_axes" in err for err in design_errors)
    assert any("op_name" in err for err in api_errors)
