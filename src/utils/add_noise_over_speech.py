"""
Audio noise addition module for speech enhancement research.

Adapted from:
M. D. Sacchi, "Adding noise with a desired signal-to-noise ratio," 
Signal Analysis and Imaging Group, Univ. of Alberta.
[Online]. Available: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
"""

import torch
import torchaudio
import torchaudio.transforms as T
import os
from pathlib import Path
from typing import Optional, Tuple, Union


def add_noise_over_speech(
    clean_audio: torch.Tensor,
    clean_sr: int,
    noise_audio: torch.Tensor,
    noise_sr: int,
    snr_db: float,
    output_dir: Optional[Union[str, Path]] = None,
    sr: Optional[int] = None,
    clean_name: Optional[str] = None,
    noise_name: Optional[str] = None,
 ) -> Tuple[torch.Tensor, int]:
    """
    Add noise to clean speech at a specified SNR using PyTorch.

    This variant accepts already-loaded torchaudio tensors instead of file
    paths. The expected shapes for `clean_audio` and `noise_audio` are
    (channels, samples) or (samples,) for mono. `clean_sr` and `noise_sr`
    should be their respective sample rates.

    Args:
        clean_audio: Loaded clean audio tensor
        clean_sr: Sample rate for the clean audio
        noise_audio: Loaded noise audio tensor
        noise_sr: Sample rate for the noise audio
        snr_db: Desired signal-to-noise ratio in dB
        output_dir: Directory to save the noisy audio. If None, returns tensor
        sr: Target sample rate. If None, uses the clean speech sample rate
        clean_name: Optional base name of the clean file for saved filename
        noise_name: Optional base name of the noise file for saved filename

    Returns:
        Tuple of (`noisy_audio`, `sample_rate`). If `output_dir` is provided the
        noisy audio will also be written to disk, but the tensor is returned in
        all cases.
    """
    
    # Select device - prefer CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine target sample rate
    target_sr = sr if sr is not None else clean_sr
    
    # If inputs are 1D (samples,), promote to (1, samples) for transforms
    if clean_audio.dim() == 1:
        clean = clean_audio.unsqueeze(0)
    else:
        clean = clean_audio

    if noise_audio.dim() == 1:
        noise = noise_audio.unsqueeze(0)
    else:
        noise = noise_audio

    # Resample if needed using modern resampling
    if clean_sr != target_sr:
        clean = T.Resample(clean_sr, target_sr)(clean)
    if noise_sr != target_sr:
        noise = T.Resample(noise_sr, target_sr)(noise)
    
    # Move to device
    clean = clean.to(device)
    noise = noise.to(device)
    
    # Convert to mono if multichannel
    if clean.shape[0] > 1:
        clean = torch.mean(clean, dim=0, keepdim=True)
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)
    
    # Remove channel dimension for processing
    clean = clean.squeeze(0)
    noise = noise.squeeze(0)
    
    # Match noise length to clean length
    clean_len = clean.shape[0]
    noise_len = noise.shape[0]
    
    if noise_len < clean_len:
        # Repeat noise to match or exceed clean length
        num_repeats = (clean_len + noise_len - 1) // noise_len  # Ceiling division
        noise = noise.repeat(num_repeats)
    
    # Truncate noise to exact clean length
    noise = noise[:clean_len]
    
    # Calculate RMS values
    rms_clean = torch.sqrt(torch.mean(clean**2))
    rms_noise = torch.sqrt(torch.mean(noise**2))
    
    # Calculate desired noise RMS based on SNR
    # SNR = 20 * log10(RMS_signal / RMS_noise)
    # Therefore: RMS_noise_desired = RMS_signal / 10^(SNR/20)
    snr_linear = 10**(snr_db / 20)
    desired_rms_noise = rms_clean / snr_linear
    
    # Scale noise to achieve desired SNR
    noise_scaling_factor = desired_rms_noise / (rms_noise + 1e-8)  # Add small epsilon to avoid division by zero
    noise_scaled = noise * noise_scaling_factor
    
    # Add noise to clean signal
    noisy_audio = clean + noise_scaled
    
    # Normalize to prevent clipping
    max_amplitude = torch.max(torch.abs(noisy_audio))
    if max_amplitude > 1.0:
        noisy_audio = noisy_audio / max_amplitude
        print(f"Warning: Audio was normalized by factor {max_amplitude:.3f} to prevent clipping")
    
    # Handle output
    if output_dir is not None:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate descriptive filename
        clean_base = Path(clean_name).stem if clean_name is not None else "CLEAN"
        noise_base = Path(noise_name).stem if noise_name is not None else "NOISE"
        output_filename = f"NOISY_{clean_base}_{noise_base}_SNR{snr_db:.1f}dB.wav"
        
        full_output_path = output_path / output_filename
        
        # Add channel dimension back for saving
        noisy_audio_save = noisy_audio.unsqueeze(0).cpu()
        torchaudio.save(full_output_path, noisy_audio_save, target_sr)
        
        print(f"Noisy audio saved to: {full_output_path}")

    return noisy_audio.cpu(), target_sr


if __name__ == "__main__":
    # Example usage
    # Load example files with torchaudio and call the tensor-based API
    clean_path = 'audio_files/clean_speech.wav'
    noise_path = 'noise_files/background_noise.wav'
    clean_tensor, clean_rate = torchaudio.load(clean_path)
    noise_tensor, noise_rate = torchaudio.load(noise_path)

    add_noise_over_speech(
        clean_audio=clean_tensor,
        clean_sr=clean_rate,
        noise_audio=noise_tensor,
        noise_sr=noise_rate,
        snr_db=5.0,
        output_dir='output',
        clean_name=Path(clean_path).name,
        noise_name=Path(noise_path).name,
    )