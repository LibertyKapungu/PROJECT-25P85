"""
Wiener Filter Speech Enhancement Implementation

This implementation is based on the algorithm by Scalart and Filho (1996)
and follows the MATLAB code by Yi Hu and Philipos C. Loizou from:
"Speech Enhancement: Theory and Practice, 2nd Edition"

References:
[1] Scalart, P. and Filho, J. (1996). "Speech enhancement based on a priori 
    signal to noise estimation." Proc. IEEE Int. Conf. Acoustics, Speech, 
    and Signal Processing, 629-632.
"""

import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple
from scipy.signal.windows import hamming


def wiener_filter(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur: int = 20,
    causal: bool = True
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Apply Wiener filtering for speech enhancement with causal processing option.

    This function accepts a pre-loaded noisy audio tensor (as returned by
    `torchaudio.load`) and its sample rate. It performs frame-based Wiener
    filtering using a decision-directed a priori SNR estimator and optionally
    saves the enhanced output to disk with a descriptive filename.

    Args:
        noisy_audio: Required. Noisy audio tensor. Shape can be (channels, samples)
            or (samples,) for mono. The tensor will be converted to mono if
            multi-channel.
        fs: Required. Sampling rate (Hz) corresponding to `noisy_audio`.
        output_dir: Optional directory to save the enhanced audio. If None,
            the function returns the enhanced tensor and sample rate.
        output_file: Optional base name used when saving the enhanced audio.
            If provided together with `output_dir` a WAV file will be written.
        input_name: Optional short name to include in the generated filename
            (useful when calling with tensors). Defaults to "wiener_as_" if
            not provided.
        mu: Smoothing factor for noise update (0 < mu < 1).
        a_dd: Decision-directed factor for a priori SNR (0 < a_dd < 1).
        eta: Voice activity detection threshold (positive float).
        frame_dur: Frame duration in milliseconds (positive integer).
        causal: If True, use causal processing (real-time compatible). When
            False, 50% overlap processing (non-causal) is used.

    Returns:
        If `output_dir` and `output_file` are provided, the function saves the
        enhanced audio to disk and returns None. Otherwise it returns a tuple
        `(enhanced_audio, fs)` where `enhanced_audio` is a 1-D CPU tensor.

    Raises:
        ValueError: If numeric parameters are out of valid ranges.

    Note:
        When `causal=True`, the algorithm processes frames sequentially without
        look-ahead, making it suitable for real-time applications. This may
        slightly reduce performance compared to non-causal processing.
    """
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Input handling: tensor input (waveform + sample rate) are required
    waveform = noisy_audio.clone()
    # Allow caller to supply a name used for saved filenames
    input_name = input_name if input_name is not None else "wiener_as_"
    print("Processing tensor input")
    
    # Convert to mono if stereo
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    else:
        waveform = waveform.squeeze()
    
    # Move to device
    waveform = waveform.to(device)
    
    # Validate parameters
    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur <= 0:
        raise ValueError("frame_dur must be positive")
    
    # Calculate frame parameters
    frame_samples = int(frame_dur * fs / 1000)  # Frame length in samples
    
    if causal:
        # Causal processing: no overlap, sequential frame processing
        hop_samples = frame_samples  # No overlap for causal processing
        print(f"CAUSAL MODE: Frame duration: {frame_dur}ms ({frame_samples} samples)")
        print(f"No overlap processing for real-time compatibility")
    else:
        # Non-causal processing: 50% overlap
        hop_samples = frame_samples // 2
        print(f"NON-CAUSAL MODE: Frame duration: {frame_dur}ms ({frame_samples} samples)")
        print(f"Hop length: {hop_samples} samples (50% overlap)")
    
    print(f"Sampling rate: {fs}Hz")
    
    # Create Hamming window
    hamming_window = torch.tensor(
        hamming(frame_samples), 
        dtype=torch.float32, 
        device=device
    )
    
    # Window normalization factor
    window_power = torch.sum(hamming_window ** 2) / frame_samples
    
    # Estimate initial noise spectrum from first 120ms (assumed noise-only)
    noise_duration_ms = 120
    noise_samples = int(fs * noise_duration_ms / 1000)
    noise_samples = min(noise_samples, len(waveform) // 4)  # Don't use more than 25% of signal
    
    if noise_samples < frame_samples:
        print("Warning: Audio too short for reliable noise estimation")
        noise_samples = min(len(waveform) // 2, frame_samples * 2)
    
    first_segment = waveform[:noise_samples]
    print(f"Noise estimation from first {noise_samples/fs*1000:.1f}ms")
    
    # Estimate noise power spectrum using overlapping frames
    num_noise_frames = max(1, (noise_samples - frame_samples) // hop_samples + 1)
    noise_power_spectrum = torch.zeros(frame_samples, device=device)
    
    for i in range(num_noise_frames):
        start_idx = i * hop_samples
        end_idx = start_idx + frame_samples
        
        if end_idx <= noise_samples:
            noise_frame = first_segment[start_idx:end_idx] * hamming_window
            noise_fft = torch.fft.fft(noise_frame, n=frame_samples)
            noise_power_spectrum += torch.abs(noise_fft) ** 2 / (frame_samples * window_power)
    
    noise_power_spectrum /= num_noise_frames
    
    # Add small epsilon to prevent division by zero
    noise_power_spectrum = torch.clamp(noise_power_spectrum, min=1e-10)
    
    # Frame-by-frame processing with causal or non-causal approach
    signal_length = len(waveform)
    
    if causal:
        # Causal processing: process frames sequentially without overlap
        num_frames = signal_length // frame_samples
        enhanced_signal = torch.zeros_like(waveform)
        
        # Pad signal if needed for complete frames
        if signal_length % frame_samples != 0:
            padding_needed = frame_samples - (signal_length % frame_samples)
            waveform = torch.cat([waveform, torch.zeros(padding_needed, device=device)])
            enhanced_signal = torch.cat([enhanced_signal, torch.zeros(padding_needed, device=device)])
            num_frames += 1
    else:
        # Non-causal processing: overlap-add reconstruction
        num_frames = (signal_length - frame_samples) // hop_samples + 1
        enhanced_signal = torch.zeros_like(waveform)
    
    vad_decisions = torch.zeros(num_frames, device=device)
    
    # Initialize variables for decision-directed approach
    G_prev = None  # Previous gain
    posteri_prev = None  # Previous posterior SNR
    
    print(f"Processing {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        if causal:
            # Causal frame extraction: sequential, non-overlapping
            start_idx = frame_idx * frame_samples
            end_idx = start_idx + frame_samples
            
            if end_idx > len(waveform):
                break  # Skip incomplete frames in causal mode
                
            current_frame = waveform[start_idx:end_idx]
        else:
            # Non-causal frame extraction: overlapping
            start_idx = frame_idx * hop_samples
            end_idx = start_idx + frame_samples
            
            if end_idx > signal_length:
                # Zero-pad if necessary
                current_frame = torch.zeros(frame_samples, device=device)
                available_samples = signal_length - start_idx
                current_frame[:available_samples] = waveform[start_idx:signal_length]
            else:
                current_frame = waveform[start_idx:end_idx]
        
        # Apply window
        windowed_frame = current_frame * hamming_window
        
        # FFT
        noisy_fft = torch.fft.fft(windowed_frame, n=frame_samples)
        noisy_power_spectrum = torch.abs(noisy_fft) ** 2 / (frame_samples * window_power)
        
        # Compute posterior SNR
        posterior_snr = noisy_power_spectrum / noise_power_spectrum
        
        # Compute a priori SNR using decision-directed approach
        if frame_idx == 0:
            # For first frame, use spectral subtraction rule
            prior_snr = torch.clamp(posterior_snr - 1, min=0.01)
        else:
            # Decision-directed approach (causal: only uses past information)
            spectral_subtraction_term = torch.clamp(posterior_snr - 1, min=0)
            prior_snr = (a_dd * (G_prev ** 2) * posteri_prev + 
                        (1 - a_dd) * spectral_subtraction_term)
            prior_snr = torch.clamp(prior_snr, min=0.01)
        
        # Voice Activity Detection using log-likelihood ratio
        log_likelihood_ratio = (posterior_snr * prior_snr / (1 + prior_snr) - 
                               torch.log1p(prior_snr))
        vad_metric = torch.mean(log_likelihood_ratio).item()
        vad_decisions[frame_idx] = vad_metric
        
        # Update noise spectrum if VAD indicates noise
        # In causal mode, this update only uses current and past information
        if vad_metric < eta:  # Noise frame
            noise_power_spectrum = (mu * noise_power_spectrum + 
                                  (1 - mu) * noisy_power_spectrum)
            noise_power_spectrum = torch.clamp(noise_power_spectrum, min=1e-10)
        
        # Compute Wiener gain
        wiener_gain = torch.sqrt(prior_snr / (1 + prior_snr))
        wiener_gain = torch.clamp(wiener_gain, min=0.01, max=1.0)
        
        # Apply gain to noisy spectrum
        enhanced_fft = noisy_fft * wiener_gain
        
        # IFFT to get time domain signal
        enhanced_frame = torch.fft.ifft(enhanced_fft, n=frame_samples).real
        
        # Frame reconstruction
        if causal:
            # Causal reconstruction: direct replacement (no overlap-add)
            enhanced_signal[start_idx:end_idx] = enhanced_frame
        else:
            # Non-causal reconstruction: overlap-add
            if frame_idx == 0:
                enhanced_signal[start_idx:start_idx + hop_samples] = enhanced_frame[:hop_samples]
            else:
                # Add overlap region
                overlap_end = min(end_idx, signal_length)
                overlap_start = start_idx
                overlap_length = min(hop_samples, overlap_end - overlap_start)
                
                if overlap_length > 0:
                    enhanced_signal[overlap_start:overlap_start + overlap_length] += enhanced_frame[:overlap_length]
            
            # Add non-overlapping region
            non_overlap_start = start_idx + hop_samples
            non_overlap_end = min(end_idx, signal_length)
            non_overlap_length = non_overlap_end - non_overlap_start
            
            if non_overlap_length > 0:
                enhanced_signal[non_overlap_start:non_overlap_end] = enhanced_frame[hop_samples:hop_samples + non_overlap_length]
        
        # Store for next iteration (causal: only past information)
        G_prev = wiener_gain
        posteri_prev = posterior_snr
    
    # Trim enhanced signal back to original length if padding was added
    if causal and len(enhanced_signal) > signal_length:
        enhanced_signal = enhanced_signal[:signal_length]
    
    # Normalize output to prevent clipping
    max_amplitude = torch.max(torch.abs(enhanced_signal))
    if max_amplitude > 1.0:
        enhanced_signal = enhanced_signal / max_amplitude
        print(f"Output normalized by factor {max_amplitude:.3f}")
    
    # Calculate enhancement statistics
    voice_frames = torch.sum(vad_decisions >= eta).item()
    noise_frames = num_frames - voice_frames
    
    print(f"\nProcessing complete:")
    print(f"  Voice frames: {voice_frames}/{num_frames} ({voice_frames/num_frames*100:.1f}%)")
    print(f"  Noise frames: {noise_frames}/{num_frames} ({noise_frames/num_frames*100:.1f}%)")
    print(f"  VAD threshold: {eta}")
    
    # Save enhanced audio if output path provided
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create descriptive filename
        metadata_parts = [
            f"FRAME{frame_dur}ms",
            f"MU{mu:.3f}".replace('.', 'p'),
            f"ADD{a_dd:.3f}".replace('.', 'p'), 
            f"ETA{eta:.3f}".replace('.', 'p')
        ]
        
        if causal:
            metadata_parts.append("CAUSAL")
        else:
            metadata_parts.append("NONCAUSAL")
        
        output_filename = f"{output_file}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        # Save with proper tensor shape for torchaudio
        enhanced_for_save = enhanced_signal.unsqueeze(0).cpu()
        torchaudio.save(full_output_path, enhanced_for_save, fs)
        
        print(f"Enhanced audio saved to: {full_output_path}")
        
    return enhanced_signal.cpu(), fs

if __name__ == "__main__":
    # Example usage - single file
    # Load an example noisy file and call the tensor-based API
    noisy_path = 'noisy_audio/noisy_speech.wav'
    noisy_tensor, noisy_rate = torchaudio.load(noisy_path)

    wiener_filter(
        noisy_audio=noisy_tensor,
        fs=noisy_rate,
        output_dir="enhanced_audio",
        output_file="wiener_enhanced",
        mu=0.98,
        a_dd=0.95,
        eta=0.15,
        frame_dur=20
    )
