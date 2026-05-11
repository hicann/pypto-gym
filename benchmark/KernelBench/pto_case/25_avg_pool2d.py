#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: AvgPool2d.

Pure PyTorch golden for 2D average pooling with SAME/VALID padding modes.
Equivalent to tf.nn.avg_pool2d.
"""

import math
import torch
import torch.nn as nn

FORMULA = "out[n, c, oh, ow] = (1/(kh*kw)) * sum_{i,j} input[n, c, oh*sh+i, ow*sw+j]"
DYNAMIC_AXIS = ["B", "C", "H", "W"]


class Model(nn.Module):
    def __init__(
        self,
        kernel_h: int = 2,
        kernel_w: int = 2,
        stride_h: int = 2,
        stride_w: int = 2,
        padding_mode: str = "SAME",
    ):
        super().__init__()
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kh, kw = self.kernel_h, self.kernel_w
        sh, sw = self.stride_h, self.stride_w

        if self.padding_mode.upper() == "SAME":
            out_h = (H + sh - 1) // sh
            out_w = (W + sw - 1) // sw
            pad_h = max(0, (out_h - 1) * sh + kh - H)
            pad_w = max(0, (out_w - 1) * sw + kw - W)
            t_pad = pad_h // 2
            b_pad = pad_h - t_pad
            l_pad = pad_w // 2
            r_pad = pad_w - l_pad
            x_padded = torch.nn.functional.pad(
                x, (l_pad, r_pad, t_pad, b_pad), mode="constant", value=0
            )
            out_h = (H + t_pad + b_pad - kh) // sh + 1
        elif self.padding_mode.upper() == "VALID":
            t_pad = b_pad = l_pad = r_pad = 0
            x_padded = x
            out_h = (H - kh) // sh + 1
        else:
            raise ValueError(
                f"Invalid padding_mode: {self.padding_mode}"
            )

        out = torch.zeros(B, C, out_h, out_w, dtype=x.dtype, device=x.device)
        for oh in range(out_h):
            h_start = oh * sh
            for ow in range(out_w):
                w_start = ow * sw
                window = x_padded[:, :, h_start:h_start + kh, w_start:w_start + kw]
                out[:, :, oh, ow] = window.mean(dim=(2, 3))
        return out


def get_inputs():
    B, C, H, W = 2, 3, 6, 6
    x = torch.randn(B, C, H, W, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [2, 2, 2, 2, "SAME"]
