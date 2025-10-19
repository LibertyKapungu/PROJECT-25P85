import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple

def wiener_filter_with_residual(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur_ms: int = 8,
    residual_method: str = 'power',  # 'power', 'time', or 'gain_floor'
    alpha: float = 0.05,  # Residual noise factor (5%)
    freq_dependent: bool = False,  # Use frequency-dependent residual like spectral subtraction
) -> Optional[Tuple[torch.Tensor, int]]:
    """Wiener filter with residual noise addition to prevent musical noise.
    
    This implementation adds controlled residual noise back to the enhanced signal
    to reduce musical noise artifacts, similar to multi-band spectral subtraction.
    
    Args:
        noisy_audio: Input noisy speech signal (mono, 1D tensor)
        fs: Sampling frequency in Hz
        output_dir: Directory to save enhanced audio
        output_file: Output filename prefix
        input_name: Input filename for metadata
        mu: Noise power update parameter (default: 0.98)
        a_dd: Decision-directed a priori SNR smoothing (default: 0.98)
        eta: VAD threshold (default: 0.15)
        frame_dur_ms: Frame duration in milliseconds (default: 8)
        residual_method: Method for adding residual noise:
            - 'power': Add to power spectrum (most accurate)
            - 'time': Blend in time domain (simplest)
            - 'gain_floor': Impose minimum gain (most elegant)
        alpha: Residual noise factor (default: 0.05 = 5%)
        freq_dependent: If True, use higher residual for high frequencies
        
    Returns:
        Tuple of (enhanced_signal, sampling_rate) or None if saving to file
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Residual method: {residual_method}, alpha: {alpha}, freq_dependent: {freq_dependent}")

    waveform = noisy_audio.clone().to(device)
    input_name = input_name if input_name is not None else "WF_residual"

    # Validation
    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if frame_dur_ms <= 0:
        raise ValueError("frame_dur must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if residual_method not in ['power', 'time', 'gain_floor']:
        raise ValueError("residual_method must be 'power', 'time', or 'gain_floor'")

    # Frame/window setup
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2

    hann = torch.hann_window(frame_samples, periodic=False, device=device)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()
    U = (analysis_win @ analysis_win) / frame_samples

    # # Initial noise PSD estimate
    # len_120ms = int(fs * 0.120)
    # init_seg = waveform[:len_120ms]
    # nsubframes = max(1, (len(init_seg) - frame_samples) // hop + 1)

    # noise_ps = torch.zeros(frame_samples, device=device)
    # for j in range(nsubframes):
    #     seg = init_seg[j * hop:j * hop + frame_samples]
    #     if seg.numel() < frame_samples:
    #         seg = torch.nn.functional.pad(seg, (0, frame_samples - seg.numel()))
    #     wseg = seg * analysis_win
    #     X = torch.fft.fft(wseg, n=frame_samples)
    #     noise_ps += (X.abs() ** 2) / (frame_samples * U)
    # noise_ps /= nsubframes

    len_120ms = int(fs * 0.120)
    init_seg = waveform[:len_120ms]
    nsubframes = max(1, (len(init_seg) - frame_samples) // hop + 1)

    # Collect power spectra for initial frames
    init_frame_powers = []
    for j in range(nsubframes):
        seg = init_seg[j * hop:j * hop + frame_samples]
        if seg.numel() < frame_samples:
            seg = torch.nn.functional.pad(seg, (0, frame_samples - seg.numel()))
        wseg = seg * analysis_win
        X = torch.fft.fft(wseg, n=frame_samples)
        power = (X.abs() ** 2) / (frame_samples * U)
        init_frame_powers.append(power)

    init_frame_powers = torch.stack(init_frame_powers)

    # Use 10th percentile (more robust than mean for post-GTCRN)
    noise_ps = torch.quantile(init_frame_powers, 0.05, dim=0)

    print(f"Initial noise estimate from {nsubframes} frames (10th percentile):")
    print(f"  Mean noise power: {10*torch.log10(noise_ps.mean() + 1e-16):.2f} dB")

    # Frequency-dependent residual factors (like spectral subtraction)
    if freq_dependent and residual_method == 'power':
        # Higher frequencies get less residual (1% vs 5%)
        # Cutoff at 75% of Nyquist frequency
        cutoff_bin = int(0.75 * frame_samples / 2)
        alpha_vec = torch.ones(frame_samples, device=device) * alpha
        alpha_vec[cutoff_bin:frame_samples - cutoff_bin] *= 0.2  # 1% for high freq
        print(f"Using frequency-dependent residual: {alpha*100:.1f}% (low), {alpha*20:.1f}% (high)")
    else:
        alpha_vec = torch.ones(frame_samples, device=device) * alpha

    # Prepare output
    n_frames = (len(waveform) - frame_samples) // hop + 1
    out_len = (n_frames - 1) * hop + frame_samples
    enhanced = torch.zeros(out_len, device=device)
    norm = torch.zeros(out_len, device=device)

    # State variables
    G_prev = torch.ones(frame_samples, device=device)
    posteri_prev = torch.ones(frame_samples, device=device)

    # Process each frame
    for j in range(n_frames):
        n_start = j * hop
        frame = waveform[n_start:n_start + frame_samples]
        if frame.numel() < frame_samples:
            frame = torch.nn.functional.pad(frame, (0, frame_samples - frame.numel()))

        win_frame = frame * analysis_win
        X = torch.fft.fft(win_frame, n=frame_samples)
        noisy_ps = (X.abs() ** 2) / (frame_samples * U)

        # Posteriori & priori SNR
        if j == 0:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev**2) * posteri_prev + (1 - a_dd) * posteri_prime

        # VAD / noise update
        # log_sigma_k = posteri * priori / (1 + priori) - torch.log1p(priori)
        # vad_decision = log_sigma_k.mean()
        # if vad_decision < eta:
        #     noise_ps = mu * noise_ps + (1 - mu) * noisy_ps

        # SNR-based VAD (more reliable after GTCRN enhancement)
        frame_signal_power = noisy_ps.mean()
        frame_noise_power = noise_ps.mean()
        frame_snr_db = 10 * torch.log10((frame_signal_power / (frame_noise_power + 1e-16)) + 1e-16)

        # If frame is quiet (< 10 dB above noise floor), update noise
        vad_threshold_db = 20.0  # Adjustable: higher = more conservative
        if frame_snr_db < vad_threshold_db:
            # This is likely noise/silence, update estimate
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
            
        # Debug (remove after testing)
        # if j % 100 == 0:
        #     print(f"Frame {j}: SNR = {frame_snr_db:.1f} dB, {'SPEECH' if frame_snr_db >= vad_threshold_db else 'NOISE'}")

        # Compute base Wiener gain
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))
        
        # Prevent over-suppression (like spectral subtraction residual)
        # min_gain = 0.3  # 30% minimum (adjustable: 0.2-0.5)
        # G = torch.clamp(G, min=min_gain)

        # # Apply gain + IFFT
        # Y = X * G

        # s = 0  # Random for now so can run the below function? 

        # # Apply residual noise method
        # if residual_method == 'power':
        #     # # Method 1: Add residual to power spectrum
        #     # enhanced_ps = (G**2) * noisy_ps + alpha_vec * noisy_ps
        #     # G_modified = torch.sqrt(torch.clamp(enhanced_ps / (noisy_ps + 1e-16), max=1.0))
        #     # Y = X * G_modified
        #     s += 1 
            
        # elif residual_method == 'gain_floor':
        #     # Method 2: Impose minimum gain
        #     G_modified = torch.clamp(G, min=alpha)
        #     Y = X * G_modified
            
        # else:  # residual_method == 'time'
        #     # Method 3: Will blend in time domain after IFFT
        #     Y = X * G

                # Compute base Wiener gain
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))

        if residual_method == 'power':
            # Power domain processing with floor and residual
            signal_power = noisy_ps
            enhanced_power = (G**2) * signal_power
            
            # Spectral floor (prevents going to zero)
            FLOOR = 0.02
            enhanced_power = torch.clamp(enhanced_power, min=FLOOR * signal_power)
            
            # Add residual (key to preventing musical noise!)
            if freq_dependent:
                enhanced_power = enhanced_power + alpha_vec * signal_power
            else:
                enhanced_power = enhanced_power + alpha * signal_power
            
            # Clamp to prevent amplification
            enhanced_power = torch.clamp(enhanced_power, max=signal_power)
            
            # Convert to gain
            G_final = torch.sqrt(enhanced_power / (signal_power + 1e-16))
            Y = X * G_final
            
        elif residual_method == 'gain_floor':
            min_gain = alpha  # Use alpha as min_gain
            G_final = torch.clamp(G, min=min_gain)
            Y = X * G_final
            
        else:  # 'time'
            G_final = G
            Y = X * G

        # IFFT
        y_ifft = torch.fft.ifft(Y).real

        # WOLA synthesis
        synth_seg = y_ifft * synth_win
        enhanced[n_start:n_start + frame_samples] += synth_seg
        norm[n_start:n_start + frame_samples] += synth_win**2

        if residual_method in ['power', 'gain_floor']:
            G_prev = G_final
        else:
            G_prev = G
        posteri_prev = posteri

    # Normalize WOLA overlap
    mask = norm > 1e-8
    enhanced[mask] /= norm[mask]

    # Trim to original length
    enhanced = enhanced[:len(waveform)]

    # Method 3: Time-domain blending (if selected)
    if residual_method == 'time':
        enhanced = (1 - alpha) * enhanced + alpha * waveform
        print(f"Applied time-domain blending: {(1-alpha)*100:.1f}% enhanced + {alpha*100:.1f}% noisy")

    # Save output
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        method_str = residual_method.upper()
        freq_str = "FREQDEP" if freq_dependent else "UNIFORM"
        
        metadata_parts = [
            f"FRAME{frame_dur_ms}ms",
            f"METHOD{method_str}",
            f"ALPHA{int(alpha*100)}PCT",
            freq_str
        ]

        output_file = output_file.replace(".wav", "")
        input_name = input_name.replace(".wav", "")

        output_filename = f"{output_file}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        torchaudio.save(full_output_path, enhanced.unsqueeze(0).cpu(), fs)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced.cpu(), fs


# Example usage demonstrating all three methods
if __name__ == "__main__":
    # Load test audio
    noisy_audio, fs = torchaudio.load("C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\enh_noisy_input.wav")
    noisy_audio = noisy_audio.mean(dim=0)  # Convert to mono
    
    # Method 1: Power spectrum residual (most similar to spectral subtraction)
    enhanced1, _ = wiener_filter_with_residual(
        noisy_audio, fs,
        residual_method='power',
        alpha=0.05,
        freq_dependent=True,  # 5% low freq, 1% high freq
        output_dir="./output",
        output_file="enh1.wav", 
        input_name="test"
    )
    
    # # Method 2: Gain floor (cleanest implementation)
    # enhanced2, _ = wiener_filter_with_residual(
    #     noisy_audio, fs,
    #     residual_method='gain_floor',
    #     alpha=0.15, 
    #     output_dir="./output",
    #     output_file="enh2.wav", 
    #     input_name="test"
    # )
    
    # # Method 3: Time domain blending (simplest)
    # enhanced3, _ = wiener_filter_with_residual(
    #     noisy_audio, fs,
    #     residual_method='time',
    #     alpha=0.05,
    #     output_dir="./output",
    #     output_file="enh3.wav", 
    #     input_name="test"
    # )