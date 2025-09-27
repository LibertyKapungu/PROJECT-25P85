import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple

############################################################
# Wiener_AS Algorithm (Python Translation)
#
# This implementation is translated from the MATLAB code by
# Philipos C. Loizou, provided on the CD that accompanies his book:
#
#   "Speech Enhancement: Theory and Practice, 2nd Edition"
#   ['https://www.routledge.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781466504219']
#
# The original MATLAB code:
#   Authors: Yi Hu and Philipos C. Loizou
#   Copyright (c) 2006 by Philipos C. Loizou
#   $Revision: 0.0 $   $Date: 10/09/2006 $
#
# References:
#   [1] Scalart, P. and Filho, J. (1996).
#       "Speech enhancement based on a priori signal to noise estimation."
#       Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing, 629â€"632.
#
# Notes:
#   - This Python version follows the original algorithm structure.
#   - Some parameter values may need tuning for optimal performance.
#   - Modified for causal processing based on wiener_as.py implementation.
#   - Uses torch tensors and same windowing as wiener_as.py
############################################################


def wiener_as(
    noisy_audio: Optional[torch.Tensor] = None,
    fs: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur_ms: int = 20,
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Implements the Wiener filtering algorithm based on a priori SNR estimation [1].
    
    This function can be called in two ways:
    1. Legacy mode: wiener_as(filename='input.wav', outfile='output.wav')
    2. Modern mode: wiener_as(noisy_audio=tensor, fs=sample_rate, ...)
            
    Args:
        noisy_audio (Optional[torch.Tensor]): Input noisy speech signal (mono, 1D tensor)
        fs (Optional[int]): Sampling frequency in Hz
        output_dir (Optional[Union[str, Path]]): Directory to save enhanced audio
        output_file (Optional[str]): Output filename prefix
        input_name (Optional[str]): Input filename for metadata
        mu (float): Smoothing factor in noise spectrum update
        a_dd (float): Smoothing factor in priori update
        eta (float): VAD threshold
        frame_dur_ms (int): Frame duration in milliseconds
        
    Returns:
        Optional[Tuple[torch.Tensor, int]]: Tuple containing enhanced speech and fs,
        or None if saving to file
          
    Example calls:
        enhanced, sr = wiener_as(noisy_audio=tensor, fs=16000)
    --------------------------------------------------------------------------------
    """
    
    # --- device + basic setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Handle input modes ---
    if noisy_audio is not None and fs is not None:
        waveform = noisy_audio.clone()
        input_name = input_name if input_name is not None else "wiener_as_"
        print("Processing tensor input")
        
    else:
        print('Usage: wiener_as(noisy_audio=tensor, fs=sample_rate) or wiener_as(filename=file.wav, outfile=out.wav)\n')
        return

    waveform = waveform.to(device)

    # Validate parameters
    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur_ms <= 0:
        raise ValueError("frame_dur_ms must be positive")

    # --- Frame / window setup (same as wiener_as.py) ---
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2
    L = frame_samples  # Keep original variable name from Loizou code

    # Use same windowing as wiener_as.py - squared-root Hann windows
    hann = torch.hann_window(L, periodic=False, device=device)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()

    # normalization constant (same as original hamming calculation)
    U = (analysis_win @ analysis_win) / L
    
    # first 120 ms is noise only
    len_120ms = int(fs * 0.120)
    first_120ms = waveform[:len_120ms]
    
    # =============now use Welch's method to estimate power spectrum with
    # Hann window and 50% overlap (adapted from original)
    nsubframes = max(1, (len(first_120ms) - L) // hop + 1)
    noise_ps = torch.zeros(L, device=device)
    
    for j in range(nsubframes):
        n_start = j * hop
        noise = first_120ms[n_start:n_start + L]
        if noise.numel() < L:
            noise = torch.nn.functional.pad(noise, (0, L - noise.numel()))
        noise = noise * analysis_win
        noise_fft = torch.fft.fft(noise, n=L)
        noise_ps = noise_ps + (noise_fft.abs() ** 2) / (L * U)
    
    noise_ps = noise_ps / nsubframes
    
    # ==============
    # number of noisy speech frames (causal processing)
    n_frames = (len(waveform) - L) // hop + 1
    out_len = (n_frames - 1) * hop + L
    
    # Initialize arrays for causal processing
    enhanced_speech = torch.zeros(out_len, device=device)
    norm = torch.zeros(out_len, device=device)
    
    # State variables for causal decision-directed prior
    G_prev = torch.ones(L, device=device)
    posteri_prev = torch.ones(L, device=device)
    
    for j in range(n_frames):
        n_start = j * hop
        noisy = waveform[n_start:n_start + L]
        if noisy.numel() < L:
            noisy = torch.nn.functional.pad(noisy, (0, L - noisy.numel()))
            
        noisy = noisy * analysis_win
        noisy_fft = torch.fft.fft(noisy, n=L)
        noisy_ps = (noisy_fft.abs() ** 2) / (L * U)
        
        # ============ voice activity detection (causal) ============
        if j == 0:  # initialize posteri
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev ** 2) * posteri_prev + (1 - a_dd) * posteri_prime
        
        log_sigma_k = posteri * priori / (1 + priori) - torch.log1p(priori)
        vad_decision = log_sigma_k.mean()
        
        if vad_decision < eta:
            # noise only frame found
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
        
        # ============ end of vad ============
        
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))  # gain function
        
        enhanced = torch.fft.ifft(noisy_fft * G, n=L).real
        
        # Causal overlap-add synthesis (non-WOLA, but using same windows)
        synth_seg = enhanced * synth_win
        enhanced_speech[n_start:n_start + L] += synth_seg
        norm[n_start:n_start + L] += synth_win ** 2
        
        # Update state variables for next frame (causal)
        G_prev = G
        posteri_prev = posteri
    
    # Normalize overlap regions
    mask = norm > 1e-8
    enhanced_speech[mask] /= norm[mask]
    
    # Trim to original length
    enhanced_speech = enhanced_speech[:len(waveform)]
    
    # --- Handle outputs ---
    if output_dir is not None and output_file is not None:
        # Modern mode - save using torchaudio with metadata
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_parts = [
            f"FRAME{frame_dur_ms}ms",
            f"MU{mu:.3f}".replace('.', 'p'),
            f"ADD{a_dd:.3f}".replace('.', 'p'),
            f"ETA{eta:.3f}".replace('.', 'p')
        ]

        output_file = output_file.replace(".wav", "")
        input_name = input_name.replace(".wav", "")

        output_filename = f"{output_file}_{input_name}{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        torchaudio.save(full_output_path, enhanced_speech.cpu().unsqueeze(0), fs)
        print(f"Enhanced audio saved to: {full_output_path}")
        return None
    
    else:
        # Return tensor mode
        return enhanced_speech.cpu(), fs