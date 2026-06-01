#!/usr/bin/env python3
"""Attention softmax -- pure PyTorch reference."""

import torch
import torch.nn.functional as F


def attn_softmax_golden(scores: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Reference softmax: F.softmax(scores * scale, dim=-1)."""
    return F.softmax(scores * scale, dim=-1)
