import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # original version:
        # self.W = torch.nn.Parameter(torch.randn(out_features, in_features), device=device, dtype=dtype)
        # chatgpt version:
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ChatGPT: Add assert
        assert x.shape[-1] == self.in_features, "the feature dimension of x is not equal to linear weight"
        result = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out") # x @ self.W.T
        return result
    
