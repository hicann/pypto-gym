#!/usr/bin/env python3
# coding: utf-8
"""Pre-import torch_npu so that @pypto.frontend.jit decorators in kernel
impl files can call torch.npu.is_available() during module collection."""
import torch_npu  # noqa: F401
