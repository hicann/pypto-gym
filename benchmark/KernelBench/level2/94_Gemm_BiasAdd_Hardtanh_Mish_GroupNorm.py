import torch
import torch.nn as nn


FORMULA = (
    "z[b, n] = x[b, k] @ W^T[k, n] + bias_linear[n] + bias_extra[n]; "
    "h[b, n] = clamp(z[b, n], -1, 1); "
    "m[b, n] = h[b, n] * tanh(softplus(h[b, n])); "
    "m_grouped[b, g, c] = m[b, g*C_g + c] with C_g = out_features / num_groups; "
    "mu[b, g] = mean_c(m_grouped[b, g, c]); var[b, g] = var_c(m_grouped[b, g, c]); "
    "n_norm[b, g, c] = (m_grouped[b, g, c] - mu[b, g]) / sqrt(var[b, g] + eps) * gamma[g*C_g + c] + beta[g*C_g + c]; "
    "out[b, n] = n_norm_flat[b, n]"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x


batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]