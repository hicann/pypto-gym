#!/usr/bin/env python3
# coding: utf-8
"""
PyPTO 算子库顶层 __init__.py — Spatial-SSRL-3B
"""

import os

USE_PTO = os.environ.get("USE_PTO", "0") == "1"
USE_PTO_RMS_NORM = USE_PTO

from .rms_norm import rms_norm_wrapper

__all__ = ["USE_PTO", "USE_PTO_RMS_NORM", "rms_norm_wrapper"]
