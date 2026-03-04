import torch

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_features, dim):
        # max_c = torch.amax(in_features, dim=self.dim) # [1]
        max_c = torch.amax(in_features, dim=dim, keepdim=True)
        # in_feature [... ]
        shift_features = in_features - max_c 
        sum_features = torch.sum(torch.exp(shift_features), dim=dim, keepdim=True)
        softmax = torch.exp(shift_features) / sum_features
        return softmax