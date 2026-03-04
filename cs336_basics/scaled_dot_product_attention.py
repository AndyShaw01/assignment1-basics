import torch
from einops import einsum
from cs336_basics.softmax import Softmax

class Scaled_Dot_Product_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(self, Q, K ,V, mask):
        # Q: ... q d_k
        # K: ... k d_k
        ## Q need to dot product with V, so the last dimension must be equal
        # V: ... k d_v
        attn_scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / K.shape[-1] ** 0.5
        attn_scores = attn_scores.masked_fill(~mask, -torch.inf)
        attn_weight = self.softmax(attn_scores, dim=-1)
        output = einsum(attn_weight, V, "... q k, ... k d_v -> ... q d_v")
        return output
        
