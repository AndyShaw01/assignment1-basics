import torch

class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, in_features):
        # max_c = torch.amax(in_features, dim=self.dim) # [1]
        max_c = torch.amax(in_features, dim=self.dim, keepdim=True)
        # in_feature [... ]
        shift_features = in_features - max_c 
        sum_features = torch.sum(torch.exp(shift_features), dim=self.dim, keepdim=True)
        softmax = torch.exp(shift_features) / sum_features
        return softmax