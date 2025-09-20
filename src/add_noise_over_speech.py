##################
#
# Adapted from 
# 
# M. D. Sacchi, "Adding noise with a desired signal-to-noise ratio," Signal Analysis and Imaging Group, 
# Univ. of Alberta. [Online]. Available: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
##################

import torch
import torchaudio
import torchaudio.transforms as T

def add_noise_over_speech(clean_speech_file: str, noise_file: str, snr_db: float, output_path: str=None, sr: int=None):
    """
    Overlaps a clean speech file with noise at a specified SNR using torchaudio.
    
    Parameters:
        clean_speech_file (str): Path to the clean speech audio file.
        noise_file (str): Path to the noise audio file.
        snr_db (float): Desired signal-to-noise ratio in dB.
        output_path (str, optional): Path to save the noisy audio. If None, returns the tensor.
        sr (int, optional): Sample rate to use. If None, uses the clean speech sample rate.
        
    Returns:
        noisy_audio (torch.Tensor): Tensor containing noisy audio if output_path is None.
    """
    
    clean, clean_sr = torchaudio.load(clean_speech_file)
    noise, noise_sr = torchaudio.load(noise_file)
    
    if sr is None:
        target_sr = clean_sr  # Use clean speech sample rate as default
    else:
        target_sr = sr
    
    # Resample if necessary
    if clean_sr != target_sr:
        resampler = T.Resample(clean_sr, target_sr)
        clean = resampler(clean)
    
    if noise_sr != target_sr:
        resampler = T.Resample(noise_sr, target_sr)
        noise = resampler(noise)
    
    # Convert to mono if multichannel (take mean across channels)
    if clean.shape[0] > 1:
        clean = torch.mean(clean, dim=0, keepdim=True)
    
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)
    
    # Remove channel dimension for processing
    clean = clean.squeeze(0)
    noise = noise.squeeze(0)
    
    # Repeat or truncate noise to match clean speech length
    if noise.shape[0] < clean.shape[0]:
        repeats = int(torch.ceil(torch.tensor(clean.shape[0] / noise.shape[0])).item())
        noise = noise.repeat(repeats)
    noise = noise[:clean.shape[0]]
    
    # Calculate RMS values
    rms_clean = torch.sqrt(torch.mean(clean**2))
    rms_noise = torch.sqrt(torch.mean(noise**2))
    
    # Calculate desired noise RMS based on SNR
    desired_rms_noise = rms_clean / (10**(snr_db / 20))
    
    # Scale noise to achieve desired SNR
    noise_scaled = noise * (desired_rms_noise / (rms_noise + 1e-8))
    noisy_audio = clean + noise_scaled
    
    # Normalize if needed to avoid clipping
    max_val = torch.max(torch.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    
    if output_path:
        # Add channel dimension back for saving
        noisy_audio_save = noisy_audio.unsqueeze(0)
        torchaudio.save(output_path, noisy_audio_save, target_sr)
    else:
        return noisy_audio