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
#       Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing, 629–632.
#
# Notes:
#   - This Python version follows the original algorithm structure.
#   - Some parameter values may need tuning for optimal performance.
#   - Updated to work with tensor inputs and outputs
############################################################

import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple

def wiener_pcl_filter(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur_ms: int = 20,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Implements the Wiener filtering algorithm based on a priori SNR estimation.

    This function implements the Wiener filtering algorithm as described by Scalart 
    and Filho (1996), following Loizou's original MATLAB implementation. Uses
    Hamming windows with 50% overlap for frame processing.

    Args:
        noisy_audio (torch.Tensor): Input noisy speech signal (mono, 1D tensor)
        fs (int): Sampling frequency in Hz
        output_dir (Optional[Union[str, Path]], optional): Directory to save enhanced audio. Defaults to None.
        output_file (Optional[str], optional): Output filename prefix. Defaults to None.
        input_name (Optional[str], optional): Input filename for metadata. Defaults to None.
        mu (float, optional): Noise power update parameter. Defaults to 0.98.
        a_dd (float, optional): Decision-directed a priori SNR smoothing. Defaults to 0.98.
        eta (float, optional): VAD threshold. Defaults to 0.15.
        frame_dur_ms (int, optional): Frame duration in milliseconds. Defaults to 20.

    Returns:
        Optional[Tuple[torch.Tensor, int]]: Tuple containing:
            - Enhanced speech signal as torch.Tensor
            - Sampling frequency
            Returns None if output_dir and output_file are provided (saves to file instead)

    Raises:
        ValueError: If mu, a_dd not in (0,1), or if eta, frame_dur_ms <= 0

    Notes:
        - Initial noise estimate uses first 120ms of signal
        - Uses CUDA if available, falls back to CPU
        - Implements VAD-based noise updating
        - Uses Hamming windows with overlap-add processing
    """

    # --- device + basic setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    waveform = noisy_audio.clone()
    input_name = input_name if input_name is not None else "WF_PCL_"
    print("Processing tensor input")

    waveform = waveform.to(device)

    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur_ms <= 0:
        raise ValueError("frame_dur_ms must be positive")

    # --- Frame / window setup ---
    L = int(frame_dur_ms * fs / 1000)  # frame length
    hamming_win = torch.hamming_window(L, periodic=False, device=device)
    U = torch.dot(hamming_win, hamming_win) / L  # normalization factor
    
    # --- Initial noise PSD estimate (first 120 ms) ---
    len_120ms = int(fs / 1000 * 120)
    first_120ms = waveform[:len_120ms]
    
    # Use Welch's method with Hamming window and 50% overlap
    nsubframes = int(torch.floor(torch.tensor(len_120ms / (L / 2))).item()) - 1
    noise_ps = torch.zeros(L, device=device)
    n_start = 0
    
    for j in range(nsubframes):
        noise_seg = first_120ms[n_start:n_start + L]
        if noise_seg.numel() < L:
            noise_seg = torch.nn.functional.pad(noise_seg, (0, L - noise_seg.numel()))
        noise_seg = noise_seg * hamming_win
        noise_fft = torch.fft.fft(noise_seg, n=L)
        noise_ps += (noise_fft.abs() ** 2) / (L * U)
        n_start += int(L / 2)
    
    noise_ps /= nsubframes
    
    # --- Process frames with 50% overlap ---
    len1 = int(L / 2)
    nframes = int(torch.floor(torch.tensor(len(waveform) / len1)).item()) - 1
    n_start = 0
    
    # Initialize arrays
    enhanced_speech = torch.zeros(len(waveform), device=device)
    vad = torch.zeros(len(waveform), device=device)  # VAD decision array
    vad_decision = torch.zeros(nframes, device=device)  # VAD decision per frame
    
    # State variables for decision-directed prior
    G_prev = None
    posteri_prev = None
    overlap = None
    
    for j in range(nframes):
        noisy_seg = waveform[n_start:n_start + L]
        if noisy_seg.numel() < L:
            noisy_seg = torch.nn.functional.pad(noisy_seg, (0, L - noisy_seg.numel()))
            
        noisy_seg = noisy_seg * hamming_win
        noisy_fft = torch.fft.fft(noisy_seg, n=L)
        noisy_ps = (noisy_fft.abs() ** 2) / (L * U)
        
        # --- Voice activity detection (exact MATLAB translation) ---
        if j == 0:  # MATLAB: if (j == 1) % initialize posteri
            posteri = noisy_ps / noise_ps
            posteri_prime = posteri - 1.0
            posteri_prime = torch.clamp(posteri_prime, min=0.0)  # MATLAB: find(posteri_prime < 0) = 0
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / noise_ps
            posteri_prime = posteri - 1.0
            posteri_prime = torch.clamp(posteri_prime, min=0.0)
            priori = a_dd * (G_prev ** 2) * posteri_prev + (1 - a_dd) * posteri_prime
        
        log_sigma_k = posteri * priori / (1 + priori) - torch.log(1 + priori)
        vad_decision[j] = torch.sum(log_sigma_k) / L
        
        if vad_decision[j] < eta:
            # noise only frame found
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
            vad[n_start:n_start + L] = 0
        else:
            vad[n_start:n_start + L] = 1
        
        # --- Wiener gain ---
        G = torch.sqrt(priori / (1 + priori))
        
        # Apply gain and IFFT
        enhanced_fft = noisy_fft * G
        enhanced = torch.fft.ifft(enhanced_fft, n=L).real
        
        # Overlap-add reconstruction
        if j == 0:
            enhanced_speech[n_start:n_start + len1] = enhanced[:len1]
        else:
            enhanced_speech[n_start:n_start + len1] = overlap + enhanced[:len1]
        
        overlap = enhanced[len1:L]
        n_start += len1
        
        # Update state variables
        G_prev = G
        posteri_prev = posteri
    
    # Add final overlap (matches MATLAB: enhanced_speech(n_start:n_start+L/2-1) = overlap)
    if overlap is not None:
        end_idx = min(n_start + len1, len(enhanced_speech))
        overlap_len = end_idx - n_start
        if overlap_len > 0:
            enhanced_speech[n_start:end_idx] = overlap[:overlap_len]

    # --- Save to file if requested ---
    if output_dir is not None and output_file is not None:
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
        torchaudio.save(full_output_path, enhanced_speech.unsqueeze(0), fs)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced_speech, fs