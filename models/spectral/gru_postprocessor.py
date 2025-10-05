import torch
import torch.nn as nn

class GRUPostProcessor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, time, bands)
        out, _ = self.gru(x)
        gain = self.sigmoid(self.fc(out))  # Gain ∈ [0, 1]
        return gain * x  # Apply gain mask to DSP output
