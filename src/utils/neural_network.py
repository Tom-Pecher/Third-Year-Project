
# Neural networks are used by the agents as function approximators (to approximate the Q-values of the states).

import torch
import torch.nn as nn
import torch.nn.functional as F

# A standard neural network class:
class NN(nn.Module):

    # Create a neural network with the given input and output sizes:
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    # Perform a forward pass:
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
