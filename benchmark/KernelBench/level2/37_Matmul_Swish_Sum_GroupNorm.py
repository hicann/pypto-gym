import torch
import torch.nn as nn


FORMULA = (
    "z[b, n] = x[b, k] @ W^T[k, n] + bias[n]; "
    "s[b, n] = sigmoid(z[b, n]) * z[b, n]; "
    "u[b, n] = s[b, n] + extra_bias[n]; "
    "u_grouped[b, g, c] = u[b, g*C_g + c] with C_g = out_features / num_groups; "
    "mu[b, g] = mean_c(u_grouped[b, g, c]); var[b, g] = var_c(u_grouped[b, g, c]); "
    "n_norm[b, g, c] = (u_grouped[b, g, c] - mu[b, g]) / sqrt(var[b, g] + eps) * gamma[g*C_g + c] + beta[g*C_g + c]; "
    "out[b, n] = n_norm_flat[b, n]"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]