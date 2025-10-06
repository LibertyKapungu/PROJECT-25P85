import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add GTCRN to path
sys.path.append(str(Path(__file__).parent / 'gtcrn'))

# Import GTCRN components
from gtcrn import GTCRN as OriginalGTCRN

class GTCRNAdapter(nn.Module):
    """
    Adapter for GTCRN to work with your DSP outputs.
    Handles frequency bin conversion and adds residual learning.
    """
    
    def __init__(self, n_fft=256, hop_length=128, hidden_channels=16):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Core GTCRN model
        self.gtcrn = OriginalGTCRN(
            in_channels=2,  # DSP output + noisy input
            hidden_channels=hidden_channels,
            kernel_size=3,
            groups=8
        )
        
        # Optional: Add frequency attention
        self.freq_attention = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(),
            nn.Linear(64, 129),
            nn.Sigmoid()
        )
    
    def forward(self, dsp_mag, noisy_mag):
        """
        Args:
            dsp_mag: DSP enhanced magnitudes (B, T, F)
            noisy_mag: Noisy magnitudes (B, T, F)
        
        Returns:
            refined: Enhanced magnitudes (B, T, F)
            residual: Learned residual correction (B, T, F)
        """
        # Stack inputs
        x = torch.stack([dsp_mag, noisy_mag], dim=1)  # (B, 2, T, F)
        
        # GTCRN processing
        refined = self.gtcrn(x)  # (B, 1, T, F)
        refined = refined.squeeze(1)  # (B, T, F)
        
        # Optional: Apply frequency attention
        freq_weights = self.freq_attention(refined.mean(1))  # (B, F)
        refined = refined * freq_weights.unsqueeze(1)
        
        # Compute residual
        residual = refined - dsp_mag
        
        return refined, residual


class CompressedLoss(nn.Module):
    """
    Loss function combining magnitude MSE and residual regularization
    with optional compression to match human perception.
    """
    
    def __init__(self, alpha=0.3, residual_weight=0.1):
        super().__init__()
        self.alpha = alpha  # Compression factor
        self.residual_weight = residual_weight
    
    def forward(self, pred, target, residual):
        # Compress magnitudes
        pred_comp = torch.pow(pred, self.alpha)
        target_comp = torch.pow(target, self.alpha)
        
        # Magnitude loss
        mag_loss = nn.functional.mse_loss(pred_comp, target_comp)
        
        # Residual regularization (encourage small, sparse corrections)
        res_loss = (residual.abs().mean() + 
                   (residual ** 2).mean())
        
        total_loss = mag_loss + self.residual_weight * res_loss
        
        return total_loss, mag_loss, res_loss


def prepare_training_pair(dsp_mag, noisy_mag, clean_mag):
    """
    Convert spectrograms to format expected by GTCRN.
    Args:
        dsp_mag: DSP enhanced magnitudes (257, T)
        noisy_mag: Noisy magnitudes (257, T) 
        clean_mag: Clean magnitudes (257, T)
    
    Returns:
        Tensors with shape (T, 129) - frames x reduced freq bins
    """
    # Convert to tensors if numpy
    if isinstance(dsp_mag, np.ndarray):
        dsp_mag = torch.from_numpy(dsp_mag).float()
    if isinstance(noisy_mag, np.ndarray):
        noisy_mag = torch.from_numpy(noisy_mag).float()
    if isinstance(clean_mag, np.ndarray):
        clean_mag = torch.from_numpy(clean_mag).float()
    
    # Trim to 129 frequency bins (match GTCRN expected input)
    dsp_mag = dsp_mag[:129].T    # (T, 129)
    noisy_mag = noisy_mag[:129].T
    clean_mag = clean_mag[:129].T
    
    return dsp_mag, noisy_mag, clean_mag