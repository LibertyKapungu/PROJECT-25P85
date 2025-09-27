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
#   [2] Crochiere, R. (1980).
#       "A weighted overlap-add method of short-time Fourier analysis/synthesis."
#       IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(1), 99-102.
#
# Notes:
#   - This Python version follows the original algorithm structure.
#   - Some parameter values may need tuning for optimal performance.
#   - Implementation uses weighted overlap-add (WOLA) for frame synthesis,
#     ensuring perfect reconstruction with 50% frame overlap.
############################################################

import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple

def wiener_filter(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur_ms: int = 8,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Implements the a-priori SNR-based Wiener filter for speech enhancement.

    This function implements the Wiener filtering algorithm based on a-priori SNR 
    estimation as described by Scalart and Filho (1996). The implementation uses
    weighted overlap-add (WOLA) processing with Hann windows and 50% overlap.

    Args:
        noisy_audio (torch.Tensor): Input noisy speech signal (mono, 1D tensor)
        fs (int): Sampling frequency in Hz
        output_dir (Optional[Union[str, Path]], optional): Directory to save enhanced audio. Defaults to None.
        output_file (Optional[str], optional): Output filename prefix. Defaults to None.
        input_name (Optional[str], optional): Input filename for metadata. Defaults to None.
        mu (float, optional): Noise power update parameter. Defaults to 0.98.
        a_dd (float, optional): Decision-directed a priori SNR smoothing. Defaults to 0.98.
        eta (float, optional): VAD threshold. Defaults to 0.15.
        frame_dur_ms (int, optional): Frame duration in milliseconds. Defaults to 8.

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
        - Uses power complementary windows for perfect reconstruction
    """

    # --- device + basic setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    waveform = noisy_audio.clone()
    input_name = input_name if input_name is not None else "wiener_as_"
    print("Processing tensor input")

    waveform = waveform.to(device)

    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur_ms <= 0:
        raise ValueError("frame_dur must be positive")

        # --- Frame / window setup ---
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2

    # squared-root Hann analysis/synthesis windows
    hann = torch.hann_window(frame_samples, periodic=False)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()

    # normalization constant
    U = (analysis_win @ analysis_win) / frame_samples

    # --- Initial noise PSD estimate (first 120 ms) ---
    len_120ms = int(fs * 0.120)
    init_seg = waveform[:len_120ms]
    nsubframes = max(1, (len(init_seg) - frame_samples) // hop + 1)

    noise_ps = torch.zeros(frame_samples)
    for j in range(nsubframes):
        seg = init_seg[j * hop:j * hop + frame_samples]
        if seg.numel() < frame_samples:
            seg = torch.nn.functional.pad(seg, (0, frame_samples - seg.numel()))
        wseg = seg * analysis_win
        X = torch.fft.fft(wseg, n=frame_samples)
        noise_ps += (X.abs() ** 2) / (frame_samples * U)
    noise_ps /= nsubframes

    # --- Prepare output ---
    n_frames = (len(waveform) - frame_samples) // hop + 1
    out_len = (n_frames - 1) * hop + frame_samples
    enhanced = torch.zeros(out_len)
    norm = torch.zeros(out_len)

    # --- State variables for decision-directed prior ---
    G_prev = torch.ones(frame_samples)
    posteri_prev = torch.ones(frame_samples)

    # --- Process each frame (causal WOLA loop) ---
    for j in range(n_frames):
        n_start = j * hop
        frame = waveform[n_start:n_start + frame_samples]
        if frame.numel() < frame_samples:
            frame = torch.nn.functional.pad(frame, (0, frame_samples - frame.numel()))

        win_frame = frame * analysis_win
        X = torch.fft.fft(win_frame, n=frame_samples)
        noisy_ps = (X.abs() ** 2) / (frame_samples * U)

        # posteriori & priori SNR
        if j == 0:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev**2) * posteri_prev + (1 - a_dd) * posteri_prime

        # VAD / noise update
        log_sigma_k = posteri * priori / (1 + priori) - torch.log1p(priori)
        vad_decision = log_sigma_k.mean()
        if vad_decision < eta:
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps

        # Wiener gain
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))

        # Apply gain + IFFT
        Y = X * G
        y_ifft = torch.fft.ifft(Y).real

        # WOLA synthesis
        synth_seg = y_ifft * synth_win
        enhanced[n_start:n_start + frame_samples] += synth_seg
        norm[n_start:n_start + frame_samples] += synth_win**2

        # update states
        G_prev = G
        posteri_prev = posteri

    # normalize WOLA overlap
    mask = norm > 1e-8
    enhanced[mask] /= norm[mask]

    # trim to original length
    enhanced = enhanced[:len(waveform)]


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
        torchaudio.save(full_output_path, enhanced.unsqueeze(0), fs)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced, fs