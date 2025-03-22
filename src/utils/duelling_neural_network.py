
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingNN(nn.Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues
    