import torch

class RoPE(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None, improved_version=True):
        super().__init__()
        self.d_k = d_k
        self.d = d_k // 2
        if improved_version:
            # positions: [T]
            pos = torch.arange(max_seq_len, device=device)  # (T,)
            # pair indices: [d_k/2]
            k = torch.arange(self.d, device=device)
            # exponent: (2k)/d_k  -> shape [d_k/2]
            exponent = (2.0 * k) / float(d_k)
            # inv_freq = 1 / theta^(2k/d_k)  -> shape [d_k/2]
            theta_base = torch.tensor(theta, device=device)
            inv_freq = torch.pow(theta_base, -exponent)  # (d/2,)
            angles = pos[:, None] * inv_freq[None, :]
            cos_table = torch.cos(angles)
            sin_table = torch.sin(angles)

        else:
            theta_base = torch.tensor(theta, device=device)
            cos_table = torch.zeros(max_seq_len, self.d, device=device)
            sin_table = torch.zeros(max_seq_len, self.d, device=device)
            for i in range(max_seq_len):
                pos = torch.tensor(i, device=device)
                for k in range(self.d):
                    temp_theta = i / torch.pow(theta_base, (2.0 * k) / float(self.d_k))
                    cos_table[pos, k] = torch.cos(temp_theta)
                    sin_table[pos, k] = torch.sin(temp_theta)
        self.register_buffer("cos", cos_table, persistent=True)
        self.register_buffer("sin", sin_table, persistent=True)

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