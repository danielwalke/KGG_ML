import torch
from torch import nn


class RunningMinMaxScaler(nn.Module):
    """A Min-Max scaler that computes running statistics, similar to BatchNorm."""

    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_min', torch.zeros(num_features))
        self.register_buffer('running_max', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_min = x.data.min(dim=0).values
            batch_max = x.data.max(dim=0).values
            if self.num_batches_tracked == 0:
                self.running_min.copy_(batch_min)
                self.running_max.copy_(batch_max)
            else:
                self.running_min = (1 - self.momentum) * self.running_min + self.momentum * batch_min
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * batch_max
            self.num_batches_tracked += 1
            return (x - batch_min) / (batch_max - batch_min + 1e-8)
        else:
            return (x - self.running_min) / (self.running_max - self.running_min + 1e-8)