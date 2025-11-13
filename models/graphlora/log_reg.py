import torch
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)
