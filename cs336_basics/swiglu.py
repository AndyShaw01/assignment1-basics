import torch
from cs336_basics.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff , device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model , device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [... d_model]
        # self.w1_weight(x) : [... d_ff]
        # silu/gate : [... d_ff] 
        # self.w3_weight(x) : [... d_ff]
        # silu * value : [... d_ff]
        # self.w2_weight(silu * value) : [...  d_model]
        # silu * value * w2_weight
        silu = self.w1(x) * torch.sigmoid(self.w1(x)) 
        value = self.w3(x)
        result = self.w2(silu * value)
        return result
        
class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)