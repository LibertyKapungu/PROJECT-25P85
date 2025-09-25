# Replace your old wiener_filter with this updated function
import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple
from scipy.signal.windows import hamming
from collections import deque

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
    causal: bool = True,
    
    gain_smooth_alpha: float = 0.85,   # smoothing for gain (0<alpha<1)
    min_gain: float = 0.01,            # absolute floor for gain
    spectral_floor_beta: float = 0.02, # spectral floor for magnitudes (0..0.1)
    noise_min_history_frames: int = 8, # frames for simple min-tracking
    freq_smooth_len: int = 3           # small freq smoothing kernel (odd)
) -> Optional[Tuple[torch.Tensor, int]]:

    # --- device + basic setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    waveform = noisy_audio.clone()
    input_name = input_name if input_name is not None else "wiener_as_"
    print("Processing tensor input")

    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    else:
        waveform = waveform.squeeze()

    waveform = waveform.to(device)

    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur <= 0:
        raise ValueError("frame_dur must be positive")

    frame_samples = int(frame_dur * fs / 1000)

    # --- IMPORTANT: use overlap even in causal mode (WOLA-style reconstruction)
    # Using 50% overlap reduces gating and musical noise. This is still causal if we
    # buffer frames and output at hop intervals (increased latency = frame length).
    hop_samples = frame_samples // 2
    print(f"{'CAUSAL' if causal else 'NON-CAUSAL'} MODE: Frame {frame_dur}ms ({frame_samples} samples), hop {hop_samples} samples")

    # window & window power
    hamming_window = torch.tensor(hamming(frame_samples), dtype=torch.float32, device=device)
    window_power = torch.sum(hamming_window ** 2) / frame_samples

    # initial noise estimate from first 120ms
    noise_duration_ms = 120
    noise_samples = int(fs * noise_duration_ms / 1000)
    noise_samples = min(noise_samples, len(waveform) // 4)
    
    if noise_samples < frame_samples:
        print("Warning: Audio too short for reliable noise estimation")
        noise_samples = min(len(waveform) // 2, frame_samples * 2)

    first_segment = waveform[:noise_samples]
    print(f"Noise estimation from first {noise_samples/fs*1000:.1f}ms")

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
    noise_power_spectrum = torch.clamp(noise_power_spectrum, min=1e-10)

    signal_length = len(waveform)
    # pad to fit integer number of hops + frames
    n_hops = (signal_length - frame_samples) // hop_samples + 1 if signal_length >= frame_samples else 1
    pad_needed = (n_hops * hop_samples + frame_samples) - signal_length

    if pad_needed > 0:
        waveform = torch.cat([waveform, torch.zeros(pad_needed, device=device)])

    total_length = len(waveform)
    num_frames = (total_length - frame_samples) // hop_samples + 1

    # Outputs & accumulators (WOLA-style for both modes)
    enhanced_accum = torch.zeros(total_length, device=device)
    window_sum = torch.zeros(total_length, device=device)

    vad_decisions = torch.zeros(num_frames, device=device)

    # for decision-directed a priori SNR
    G_prev = torch.ones(frame_samples, device=device) * 0.5
    posteri_prev = torch.ones(frame_samples, device=device) * 1.0

    # additional state for smoothing gains and min-noise tracking
    smoothed_gain = torch.ones(frame_samples, device=device) * 1.0
    # simple per-bin minimum history using deque of tensors (small memory)
    noise_history = deque(maxlen=noise_min_history_frames)
    # initialize history with initial noise estimate
    for _ in range(noise_min_history_frames):
        noise_history.append(noise_power_spectrum.clone())

    # small frequency smoothing kernel (uniform)
    if freq_smooth_len > 1:
        k = freq_smooth_len
        freq_kernel = torch.ones(k, device=device) / float(k)
    else:
        freq_kernel = None

    eps_window = 1e-3
    hamming_window_clamped = torch.clamp(hamming_window, min=eps_window)

    print(f"Processing {num_frames} frames with WOLA-style reconstruction ...")

    for frame_idx in range(num_frames):
        start_idx = frame_idx * hop_samples
        end_idx = start_idx + frame_samples
        current_frame = waveform[start_idx:end_idx] * hamming_window

        # FFT
        noisy_fft = torch.fft.fft(current_frame, n=frame_samples)
        noisy_power_spectrum = torch.abs(noisy_fft) ** 2 / (frame_samples * window_power)

        # optional frequency smoothing to reduce narrow spikes (lowers variance)
        if freq_kernel is not None:
            # simple 1D conv with circular padding in freq domain (small kernel)
            padded = torch.nn.functional.pad(noisy_power_spectrum.unsqueeze(0).unsqueeze(0),
                                             (k//2, k//2), mode='reflect')
            conv = torch.nn.functional.conv1d(padded, freq_kernel.view(1,1,-1))
            noisy_power_smoothed = conv.squeeze()
        else:
            noisy_power_smoothed = noisy_power_spectrum

        # posterior SNR
        posterior_snr = noisy_power_smoothed / noise_power_spectrum
        posterior_snr = torch.clamp(posterior_snr, min=1e-6)

        # decision-directed a priori SNR (uses past estimates only)
        if frame_idx == 0:
            prior_snr = torch.clamp(posterior_snr - 1, min=0.01)
        else:
            spectral_subtraction_term = torch.clamp(posterior_snr - 1, min=0.0)
            # use previous gain & previous posterior in the DD formula (vector)
            prior_snr = (a_dd * (G_prev ** 2) * posteri_prev + (1 - a_dd) * spectral_subtraction_term)
            prior_snr = torch.clamp(prior_snr, min=0.01)

        # VAD metric (log-likelihood ratio style)
        log_likelihood_ratio = (posterior_snr * prior_snr / (1 + prior_snr) - torch.log1p(prior_snr))
        vad_metric = torch.mean(log_likelihood_ratio).item()
        vad_decisions[frame_idx] = vad_metric

        # --- Noise update strategy: combine exponential update + minimum tracking
        # If frame likely noise, update quickly; if speech present, update slowly.
        if vad_metric < eta:
            # update instantaneous noise estimate (fast)
            noise_instant = noisy_power_smoothed
            noise_power_spectrum = mu * noise_power_spectrum + (1 - mu) * noise_instant
            noise_power_spectrum = torch.clamp(noise_power_spectrum, min=1e-10)
        else:
            # speech present: slow update to avoid over-estimation
            noise_power_spectrum = mu * noise_power_spectrum + (1 - mu) * noisy_power_smoothed * 0.1
            noise_power_spectrum = torch.clamp(noise_power_spectrum, min=1e-10)

        # update noise history and compute a minima-based floor (simple minimum statistics)
        noise_history.append(noise_power_spectrum.clone())
        min_noise = torch.min(torch.stack(list(noise_history)), dim=0)[0]
        # combine the running estimate with the local minima to be conservative
        noise_power_spectrum = torch.maximum(noise_power_spectrum, 0.8 * min_noise)

        # Compute Wiener gain (square-root Wiener as before)
        wiener_gain = torch.sqrt(prior_snr / (1 + prior_snr))
        # apply spectral floor to gain to avoid deep valleys (prevents musical spikes)
        wiener_gain = torch.clamp(wiener_gain, min=min_gain)
        # impose an upper limit
        wiener_gain = torch.clamp(wiener_gain, max=1.0)

        # --- Temporal smoothing of the *gain* to suppress frame-to-frame variation
        # Adaptive strategy: if VAD indicates stable noise -> more smoothing; if speech -> less smoothing
        if vad_metric < eta:
            alpha = gain_smooth_alpha  # more smoothing in noise frames
        else:
            # allow faster adaptation during speech activity (less smoothing)
            alpha = 0.6 + 0.4 * (1 - min(0.9, abs(vad_metric)))  # heuristic but prevents extremes
            alpha = float(max(0.5, min(alpha, gain_smooth_alpha)))

        smoothed_gain = alpha * smoothed_gain + (1 - alpha) * wiener_gain

        # apply small additional spectral floor to the smoothed gain (spectral-flooring principle)
        smoothed_gain = torch.clamp(smoothed_gain, min=spectral_floor_beta)

        # apply gain to complex spectrum (magnitude scaling)
        enhanced_fft = noisy_fft * smoothed_gain

        # IFFT
        enhanced_frame = torch.fft.ifft(enhanced_fft, n=frame_samples).real

        # WOLA reconstruction (add to accumulator with the analysis/synthesis window)
        enhanced_accum[start_idx:end_idx] += enhanced_frame * hamming_window  # synthesis uses same window here
        window_sum[start_idx:end_idx] += hamming_window_clamped

        # update previous values for DD estimator
        G_prev = smoothed_gain
        posteri_prev = posterior_snr

    # finalize: divide out window sum (safe)
    safe_window = torch.clamp(window_sum, min=eps_window)
    enhanced_signal = enhanced_accum / safe_window
    enhanced_signal = enhanced_signal[:signal_length]  # trim padding

    # normalize to prevent clipping (same as before)
    max_amplitude = torch.max(torch.abs(enhanced_signal))
    if max_amplitude > 1.0:
        enhanced_signal = enhanced_signal / max_amplitude
        print(f"Output normalized by factor {max_amplitude:.3f}")

    # stats & save (unchanged semantics)
    voice_frames = torch.sum(vad_decisions >= eta).item()
    noise_frames = num_frames - voice_frames

    print(f"\nProcessing complete:")
    print(f"  Voice frames: {voice_frames}/{num_frames} ({voice_frames/num_frames*100:.1f}%)")
    print(f"  Noise frames: {noise_frames}/{num_frames} ({noise_frames/num_frames*100:.1f}%)")
    print(f"  VAD threshold: {eta}")

    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_parts = [
            f"FRAME{frame_dur}ms",
            f"MU{mu:.3f}".replace('.', 'p'),
            f"ADD{a_dd:.3f}".replace('.', 'p'),
            f"ETA{eta:.3f}".replace('.', 'p')
        ]

        metadata_parts.append("CAUSAL_WOLA" if causal else "NONCAUSAL")
        output_filename = f"{output_file}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        enhanced_for_save = enhanced_signal.unsqueeze(0).cpu()
        torchaudio.save(full_output_path, enhanced_for_save, fs)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced_signal.cpu(), fs
