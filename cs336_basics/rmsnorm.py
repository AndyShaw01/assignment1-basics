import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # g
        self.eps = eps
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32) # [B, T, D]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # [B, T, 1]
        result = (x / rms) * self.weight
        return result.to(in_dtype)