import torch

class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        # inputs: [... vocab_size], e.g. [batch_size, vocab_size]
        # targets: [...], e.g. [batch_size]
        max_c = torch.amax(inputs, dim=-1, keepdim=True)
        exp_shifted = torch.exp(inputs - max_c)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_c + torch.log(sum_exp)
        
        targets_logits = inputs.gather(dim=-1,index=targets.unsqueeze(-1))
        losses = log_sum_exp - targets_logits
        loss = losses.mean()
        return loss