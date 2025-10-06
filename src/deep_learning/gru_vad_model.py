"""
Lightweight GRU-based Voice Activity Detection (VAD) Model

This module implements a causal GRU-based VAD model designed for real-time
speech enhancement applications. The model operates on 8ms frames to match
the Wiener filter frame structure.

Architecture inspired by:
- Zhang, X.-L., & Wang, D. (2016). "A deep ensemble learning method for 
  monaural speech separation." IEEE/ACM Trans. Audio, Speech, Language Process.
- Zazo, R., et al. (2016). "Feature learning with raw-waveform CLDNNs for 
  Voice Activity Detection." Interspeech.

Key Features:
- Causal processing (no future context)
- Operates on 8ms frames (128 samples at 16kHz)
- Lightweight architecture (~50K parameters)
- Returns frame-level VAD probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GRU_VAD(nn.Module):
    """Lightweight GRU-based Voice Activity Detection model.
    
    This model processes audio frames and predicts whether each frame contains
    speech (1) or non-speech (0). The architecture is designed to be causal
    and lightweight for real-time applications.
    
    Architecture:
        Input (128,) -> Feature Extractor (Conv1D layers) -> 
        GRU layers (bidirectional=False for causality) -> 
        Classifier -> Output probability
    
    Args:
        input_size (int): Number of samples per frame. Default: 128 (8ms at 16kHz)
        hidden_size (int): Number of hidden units in GRU layers. Default: 64
        num_layers (int): Number of GRU layers. Default: 2
        dropout (float): Dropout rate between GRU layers. Default: 0.2
        
    Input Shape:
        - Training: (batch, seq_len, input_size) where seq_len is number of frames
        - Inference: (batch, 1, input_size) for single frame processing
        
    Output Shape:
        - (batch, seq_len) with values in [0, 1] representing speech probability
    """
    
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(GRU_VAD, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction: 1D convolutions for local temporal patterns
        # This reduces dimensionality and extracts relevant features
        self.feature_extractor = nn.Sequential(
            # Conv1: Extract low-level features
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64
            
            # Conv2: Extract mid-level features
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 -> 32
            
            # Conv3: Extract high-level features
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 32 -> 16
        )
        
        # Calculate feature dimension after convolutions
        self.feature_dim = 64 * 16  # channels * reduced_length
        
        # GRU layers for temporal modeling (causal - no bidirectional)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # IMPORTANT: Causal processing only
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional hidden state from previous forward pass
                   Shape: (num_layers, batch, hidden_size)
                   
        Returns:
            Tuple of (output, hidden_state):
                - output: VAD probabilities of shape (batch, seq_len)
                - hidden_state: Updated hidden state for next forward pass
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for conv1d: (batch * seq_len, 1, input_size)
        x_reshaped = x.reshape(batch_size * seq_len, 1, self.input_size)
        
        # Extract features
        features = self.feature_extractor(x_reshaped)  # (batch*seq_len, 64, 16)
        
        # Flatten features
        features = features.reshape(batch_size * seq_len, -1)  # (batch*seq_len, 1024)
        
        # Reshape back for GRU: (batch, seq_len, feature_dim)
        features = features.reshape(batch_size, seq_len, -1)
        
        # Apply GRU
        if hidden is not None:
            gru_out, hidden = self.gru(features, hidden)
        else:
            gru_out, hidden = self.gru(features)
        
        # Classify each frame
        # Reshape for classifier: (batch * seq_len, hidden_size)
        gru_out_reshaped = gru_out.reshape(batch_size * seq_len, -1)
        output = self.classifier(gru_out_reshaped)  # (batch*seq_len, 1)
        
        # Reshape output: (batch, seq_len)
        output = output.reshape(batch_size, seq_len)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for GRU.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Zero-initialized hidden state of shape (num_layers, batch, hidden_size)
        """
        return torch.zeros(
            self.num_layers, 
            batch_size, 
            self.hidden_size, 
            device=device
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRU_VAD_Lite(nn.Module):
    """Ultra-lightweight GRU-VAD variant with fewer parameters.
    
    This is a simplified version with ~15K parameters for resource-constrained
    applications.
    """
    
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(GRU_VAD_Lite, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Simplified feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        self.feature_dim = 32 * 32
        
        # Single GRU layer
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        x_reshaped = x.reshape(batch_size * seq_len, 1, self.input_size)
        features = self.feature_extractor(x_reshaped)
        features = features.reshape(batch_size * seq_len, -1)
        features = features.reshape(batch_size, seq_len, -1)
        
        # GRU
        if hidden is not None:
            gru_out, hidden = self.gru(features, hidden)
        else:
            gru_out, hidden = self.gru(features)
        
        # Classify
        gru_out_reshaped = gru_out.reshape(batch_size * seq_len, -1)
        output = self.classifier(gru_out_reshaped)
        output = output.reshape(batch_size, seq_len)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(
            self.num_layers, 
            batch_size, 
            self.hidden_size, 
            device=device
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test standard model
    model = GRU_VAD().to(device)
    print(f"GRU_VAD parameters: {model.count_parameters():,}")
    
    # Test with sample input
    batch_size, seq_len = 4, 100
    x = torch.randn(batch_size, seq_len, 128).to(device)
    
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test lite model
    model_lite = GRU_VAD_Lite().to(device)
    print(f"\nGRU_VAD_Lite parameters: {model_lite.count_parameters():,}")
    
    output_lite, hidden_lite = model_lite(x)
    print(f"Lite output shape: {output_lite.shape}")
