import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                            padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        return out

class GateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        self.conv = DilatedConv(in_channels, out_channels*2, kernel_size, dilation)
        self.bn = nn.BatchNorm2d(out_channels*2)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        # Gating mechanism
        z = torch.chunk(out, 2, dim=1)
        return torch.tanh(z[0]) * torch.sigmoid(z[1])

class GTCRN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=16, kernel_size=3, groups=8):
        super().__init__()
        
        # Dilated convolution blocks with gating
        self.gate1 = GateBlock(in_channels, hidden_channels, kernel_size, dilation=1)
        self.gate2 = GateBlock(hidden_channels, hidden_channels, kernel_size, dilation=2)
        self.gate3 = GateBlock(hidden_channels, hidden_channels, kernel_size, dilation=4)
        self.gate4 = GateBlock(hidden_channels, hidden_channels, kernel_size, dilation=8)
        
        # Group recurrent layers for frequency-wise processing
        self.gru_groups = nn.ModuleList([
            nn.GRU(input_size=hidden_channels // groups,
                  hidden_size=hidden_channels // groups,
                  batch_first=True,
                  bidirectional=True)
            for _ in range(groups)
        ])
        
        # Final 1x1 convolution
        self.out_conv = nn.Conv2d(hidden_channels, 1, 1)
        
    def _split_groups(self, x, groups):
        batch, channels, time, freq = x.size()
        x = x.view(batch, groups, -1, time, freq)
        return x
    
    def _merge_groups(self, x):
        batch, groups, channels, time, freq = x.size()
        x = x.view(batch, -1, time, freq)
        return x
    
    def forward(self, x):
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Multi-scale feature extraction with gating
        out = self.gate1(x)
        print(f"After gate1: {out.shape}")
        out = self.gate2(out)
        print(f"After gate2: {out.shape}")
        out = self.gate3(out)
        print(f"After gate3: {out.shape}")
        out = self.gate4(out)
        print(f"After gate4: {out.shape}")
        
        # Group processing
        batch, channels, time, freq = out.size()
        groups = len(self.gru_groups)
        
        # Split into groups
        out = self._split_groups(out, groups)
        
        # Print shape for debugging
        # Process each group with GRU
        batch_size = out.size(0)
        hidden_per_group = out.size(1) // groups
        
        for i, gru in enumerate(self.gru_groups):
            # Calculate start and end indices for this group's channels
            start_ch = i * hidden_per_group
            end_ch = (i + 1) * hidden_per_group
            
            # Extract this group's features
            group = out[:, start_ch:end_ch]  # (batch, hidden_per_group, time, freq)
            
            # Reshape for GRU: combine batch and freq dims
            group = group.permute(0, 3, 2, 1)  # (batch, freq, time, channels)
            batch_freq = group.size(0) * group.size(1)
            group = group.reshape(batch_freq, -1, hidden_per_group)  # (batch*freq, time, channels)
            
            # Process through GRU
            group_out, _ = gru(group)  # (batch*freq, time, channels*2)
            
            # Reshape back
            group_out = group_out[..., :hidden_per_group]  # Take first half of bidirectional output
            group_out = group_out.reshape(batch_size, -1, group.size(1), hidden_per_group)  # (batch, freq, time, channels)
            group_out = group_out.permute(0, 3, 2, 1)  # (batch, channels, time, freq)
            
            # Update the output tensor
            out[:, start_ch:end_ch] = group_out
        
        # Merge groups back
        out = self._merge_groups(out)
        
        # Final convolution
        out = self.out_conv(out)
        
        return out