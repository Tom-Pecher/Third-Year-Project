
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DQN(nn.Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, filename:str, path="src/saved") -> None:
        torch.save(self.state_dict(), Path(path) / filename)

    def load(self, filename:str, path="src/saved") -> None:
        self.load_state_dict(torch.load(Path(path) / filename))
