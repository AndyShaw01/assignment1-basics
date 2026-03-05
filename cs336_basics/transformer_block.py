import torch

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta=theta, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        y = self.norm_1(x)
        token_positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.attn(y, using_rope=True, token_positions=token_positions)
        z = self.norm_2(x)
        x = x + self.ffn(z)

        return x
        