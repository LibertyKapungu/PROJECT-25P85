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
import os

def add_noise_over_speech(
    clean_dir: str, clean_filename: str,
    noise_dir: str, noise_filename: str,
    snr_db: float,
    output_dir: str = None,
    sr: int = None
):
    """
    Overlaps a clean speech file with noise at a specified SNR using torchaudio.
    
    Parameters:
        clean_dir (str): Directory containing the clean speech file.
        clean_filename (str): Name of the clean speech file.
        noise_dir (str): Directory containing the noise file.
        noise_filename (str): Name of the noise file.
        snr_db (float): Desired signal-to-noise ratio in dB.
        output_dir (str, optional): Directory to save the noisy audio. If None, returns tensor.
        output_filename (str, optional): Filename for the noisy audio. If None, automatically generated with metadata.
        sr (int, optional): Sample rate to use. If None, uses the clean speech sample rate.
        
    Returns:
        noisy_audio (torch.Tensor): Tensor containing noisy audio if output_dir is None.
    """

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build paths
    clean_path = os.path.join(clean_dir, clean_filename)
    noise_path = os.path.join(noise_dir, noise_filename)

    clean, clean_sr = torchaudio.load(clean_path)
    noise, noise_sr = torchaudio.load(noise_path)

    if sr is None:
        target_sr = clean_sr  
    else:
        target_sr = sr

    # Resample if needed
    if clean_sr != target_sr:
        resampler = T.Resample(clean_sr, target_sr).to(device)
        clean = resampler(clean.to(device))
    else:
        clean = clean.to(device)

    if noise_sr != target_sr:
        resampler = T.Resample(noise_sr, target_sr).to(device)
        noise = resampler(noise.to(device))
    else:
        noise = noise.to(device)

    # Convert to mono if multichannel
    if clean.shape[0] > 1:
        clean = torch.mean(clean, dim=0, keepdim=True)
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)

    # Remove channel dimension
    clean = clean.squeeze(0)
    noise = noise.squeeze(0)

    # Match noise length to clean length
    if noise.shape[0] < clean.shape[0]:
        repeats = int(torch.ceil(torch.tensor(clean.shape[0] / noise.shape[0], device=device)).item())
        noise = noise.repeat(repeats)
    noise = noise[:clean.shape[0]]

    # RMS calculations
    rms_clean = torch.sqrt(torch.mean(clean**2))
    rms_noise = torch.sqrt(torch.mean(noise**2))

    desired_rms_noise = rms_clean / (10**(snr_db / 20))

    # Scale noise
    noise_scaled = noise * (desired_rms_noise / (rms_noise + 1e-8))
    noisy_audio = clean + noise_scaled

    # Normalize if needed
    max_val = torch.max(torch.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    # Determine output filename
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        base_clean = os.path.splitext(clean_filename)[0]
        base_noise = os.path.splitext(noise_filename)[0]
        output_filename = f"NOISY_{base_clean}_{base_noise}_SNR_{snr_db:.1f}.wav"

        output_path = os.path.join(output_dir, output_filename)

        noisy_audio_save = noisy_audio.unsqueeze(0).cpu()  # add channel back
        torchaudio.save(output_path, noisy_audio_save, target_sr)

        print(f"Noisy audio saved to {output_path}")
    else:
        return noisy_audio.cpu()


# add_noise_over_speech(
#     'audio_stuff', 'S_56_02.wav',
#     'audio_stuff', 'sp21_station_sn0.wav',
#     -5, 'yoh/')