from collections.abc import Callable, Iterable
from typing import Optional

import torch
import math



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay    
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 0) + 1
                grad = p.grad.data

                m_t = beta_1 * m + (1 - beta_1) * grad
                v_t = beta_2 * v + (1 - beta_2) * (grad**2)
                
                m_hat = m_t / (1 - beta_1**t)
                v_hat = v_t / (1 - beta_2**t)

                p.data = p.data * (1 - lr * weight_decay) - lr * m_hat / (torch.sqrt(v_hat) + eps)

                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t
        return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data # loss.backward function will fill "p.grad"
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss