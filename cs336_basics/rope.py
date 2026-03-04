import torch

class RoPE(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.d_k = d_k
        self.d = d_k // 2
        # self.cos = [[0 for _ in range(d_k / 2)] for _ in range(max_seq_len)]
        # self.sin = [[0 for _ in range(d_k / 2)] for _ in range(max_seq_len)]
        theta_base = torch.tensor(theta, device=device)
        self.cos_table = torch.zeros(max_seq_len, self.d, device=device)
        self.sin_table = torch.zeros(max_seq_len, self.d, device=device)
        for i in range(max_seq_len):
            pos = torch.tensor(i, device=device)
            for k in range(self.d):
                temp_theta = i / torch.pow(theta_base, (2.0 * k) / float(self.d_k))
                self.cos_table[pos, k] = torch.cos(temp_theta)
                self.sin_table[pos, k] = torch.sin(temp_theta)
        self.register_buffer("cos", self.cos_table, persistent=True)
        self.register_buffer("sin", self.sin_table, persistent=True)

    def forward(self, x, token_positions):
        # x.shape: [B, T, D_K]
        cos = self.cos[token_positions] # [t, D_K // 2]
        sin = self.sin[token_positions] # [t, D_K // 2]

        x_even = x[..., 0::2]
        x_old = x[..., 1::2]

        y_even = x_even * cos - x_old * sin
        y_old = x_even * sin + x_old * cos

        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_old

        return y