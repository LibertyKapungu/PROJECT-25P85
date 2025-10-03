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

    This function handles different audio length scenarios to avoid statistical artifacts:
    - If noise > speech: Truncates noise to a random frame of speech size (no repetition)
    - If speech > noise: Uses overlap-crossfade extension for smooth transitions
    
    The expected shapes for `clean_audio` and `noise_audio` are (channels, samples) 
    or (samples,) for mono. Both signals are automatically resampled to the target 
    sample rate and converted to mono if necessary.

    Args:
        clean_audio: Loaded clean audio tensor (channels, samples) or (samples,)
        clean_sr: Sample rate for the clean audio in Hz
        noise_audio: Loaded noise audio tensor (channels, samples) or (samples,)
        noise_sr: Sample rate for the noise audio in Hz
        snr_db: Desired signal-to-noise ratio in dB
        output_dir: Directory to save the noisy audio. If None, only returns tensor
        sr: Target sample rate in Hz. If None, uses the clean speech sample rate
        clean_name: Optional base name of the clean file for generating output filename
        noise_name: Optional base name of the noise file for generating output filename

    Returns:
        Tuple of (noisy_audio, sample_rate):
        - noisy_audio: torch.Tensor of shape (samples,) containing the mixed audio
        - sample_rate: int, the sample rate of the output audio
        
        If `output_dir` is provided, the noisy audio is also saved to disk with
        a descriptive filename including SNR information.
        
    Note:
        For reproducible results, set torch.manual_seed() before calling this function
        in your calling code, as random frame selection uses torch.randint().
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
    
    # Calculate sizes and implement length matching logic to avoid repetitions
    clean_len = clean.shape[0]
    noise_len = noise.shape[0]
    clean_duration_seconds = clean_len / target_sr
    noise_duration_seconds = noise_len / target_sr
    
    print(f"Clean speech duration: {clean_duration_seconds:.2f}s ({clean_len} samples)")
    print(f"Noise duration: {noise_duration_seconds:.2f}s ({noise_len} samples)")
    
    if noise_len > clean_len:
        # Case 1: Noise is longer than speech - truncate noise to random frame of speech size
        max_start_idx = noise_len - clean_len
        random_start = torch.randint(0, max_start_idx + 1, (1,)).item()
        noise = noise[random_start:random_start + clean_len]
        print(f"Noise truncated to random frame starting at sample {random_start}")
        
    elif clean_len > noise_len:
        # Case 2: Speech is longer than noise - use overlap-crossfade method for natural extension
        crossfade_samples = min(int(0.05 * target_sr), noise_len // 8)  # 50ms crossfade
        num_full_copies = (clean_len - crossfade_samples) // (noise_len - crossfade_samples)
        
        noise_extended = torch.zeros(clean_len, device=noise.device, dtype=noise.dtype)
        pos = 0
        
        for i in range(num_full_copies + 1):
            if pos >= clean_len:
                break
                
            end_pos = min(pos + noise_len, clean_len)
            copy_len = end_pos - pos
            
            if i == 0:
                # First copy - no crossfade at start
                noise_extended[pos:end_pos] = noise[:copy_len]
            else:
                # Subsequent copies - apply crossfade
                if crossfade_samples > 0 and pos >= crossfade_samples:
                    # Crossfade region
                    fade_out = torch.linspace(1, 0, crossfade_samples, device=noise.device)
                    fade_in = torch.linspace(0, 1, crossfade_samples, device=noise.device)
                    
                    crossfade_start = pos - crossfade_samples
                    crossfade_end = pos
                    
                    # Apply crossfade
                    noise_extended[crossfade_start:crossfade_end] *= fade_out
                    noise_extended[crossfade_start:crossfade_end] += noise[:crossfade_samples] * fade_in
                    
                    # Add rest of the signal
                    remaining_start = pos
                    remaining_end = min(pos + noise_len - crossfade_samples, clean_len)
                    remaining_len = remaining_end - remaining_start
                    noise_extended[remaining_start:remaining_end] = noise[crossfade_samples:crossfade_samples + remaining_len]
                else:
                    noise_extended[pos:end_pos] = noise[:copy_len]
            
            pos += noise_len - crossfade_samples
        
        noise = noise_extended
        print(f"Noise extended using overlap-crossfade method with {crossfade_samples}-sample crossfades")
    
    # Ensure both signals have the same length after processing
    final_len = min(clean.shape[0], noise.shape[0])
    clean = clean[:final_len]
    noise = noise[:final_len]
    
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
        print(f"[Warning]: Audio was normalized by factor {max_amplitude:.3f} to prevent clipping")
    
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
    clean_path = 'sound_data/raw/EARS_DATASET/p001/emo_adoration_freeform.wav'
    noise_path = 'sound_data/raw/WHAM_NOISE_DATASET/tt/22ga010d_1.5482_052o020t_-1.5482.wav'
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