# gnn_pipeline/models/projector.py
import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)
