import torch

from einops import einsum
from cs336_basics.linear import Linear
from cs336_basics.softmax import Softmax
from cs336_basics.rope import RoPE

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length, theta=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # keep "int"
        self.q_proj = Linear(self.d_model, self.d_model, device=device)
        self.k_proj = Linear(self.d_model, self.d_model, device=device)
        self.v_proj = Linear(self.d_model, self.d_model, device=device)
        self.output_proj = Linear(self.d_model, self.d_model, device=device)
        self.softmax = Softmax()
        if theta:
            self.rope = RoPE(theta, self.head_dim, max_seq_length, device=device)
        mask = torch.triu(
            torch.ones(max_seq_length, max_seq_length, device=device, dtype=torch.bool), 
            diagonal=1
        ) # future token is true
        self.register_buffer("mask", mask, persistent=False) # deterministic, in the forward function, we can call the "mask" by "self.mask"

    def forward(self, x, using_rope=False, token_positions=None):
        B, T, _ = x.shape
        # [B, T, D] -> [B, T, D]
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        # [B, T, D] -> [B, T, H, H_D] -> [B, H, T, H_D]
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if using_rope:
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)
        # attn_scores
        attn_scores = einsum(queries, keys, "... q d_in, ... k d_in -> ... q k") / self.head_dim ** 0.5
        mask = self.mask[:T, :T]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = self.softmax(attn_scores, dim=-1)
        output = einsum(attn_weights, values, "... q k, ... k d_in -> ... q d_in")
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.output_proj(output)

        return output