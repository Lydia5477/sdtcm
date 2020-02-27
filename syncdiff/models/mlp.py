import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 3):
        super(MLP, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, inputs):
        logits = self._classifier(inputs)

        return logits
