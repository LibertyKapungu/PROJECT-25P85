"""
Enhanced Multi-band Spectral Subtraction with Speech Recovery Mechanisms

NEW FEATURES for handling GTCRN over-suppression:
1. Residual speech detection in noisy input
2. Selective speech restoration from original noisy signal
3. Harmonic structure analysis
4. Adaptive mixing based on confidence

The key insight: Keep the ORIGINAL NOISY signal and selectively
blend it back when GTCRN appears to have removed speech.
"""

import numpy as np
from scipy.fft import fft, ifft
import scipy.signal
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import torchaudio

def detect_speech_suppression(original_noisy, gtcrn_output, frame_mag_noisy, 
                              frame_mag_gtcrn, threshold=0.7):
    """
    Detect frames where GTCRN may have over-suppressed speech.
    
    Args:
        original_noisy: Original noisy magnitude spectrum
        gtcrn_output: GTCRN enhanced magnitude spectrum
        frame_mag_noisy: Full frame magnitude from noisy
        frame_mag_gtcrn: Full frame magnitude from GTCRN
        threshold: Suppression threshold (0.7 = 70% reduction triggers recovery)
    
    Returns:
        recovery_mask: Binary mask indicating frames to recover
        recovery_weight: Confidence weight for blending (0-1)
    """
    # Calculate suppression ratio per frequency bin
    suppression_ratio = frame_mag_gtcrn / (frame_mag_noisy + 1e-10)
    
    # Detect excessive suppression in speech-critical bands (300-3400 Hz)
    # For 16kHz, 512-point FFT: bins ~10-108 cover speech range
    speech_bands = slice(10, 108)
    speech_suppression = suppression_ratio[speech_bands]
    
    # Check if too much energy was removed
    avg_suppression = np.mean(speech_suppression)
    
    # If average suppression > threshold, flag for recovery
    needs_recovery = avg_suppression < threshold
    
    # Calculate confidence weight (how much to blend back)
    # More aggressive suppression → higher recovery weight
    if needs_recovery:
        recovery_weight = np.clip((threshold - avg_suppression) / threshold, 0, 0.5)
    else:
        recovery_weight = 0.0
    
    return needs_recovery, recovery_weight

def harmonic_analysis(magnitude_spectrum, fs, fft_length):
    """
    Analyze harmonic structure to detect if speech harmonics were removed.
    
    Speech has clear harmonic structure (F0 and overtones).
    If GTCRN removed these, we should detect the loss.
    """
    # Convert to frequency bins for F0 range (80-300 Hz for adult speech)
    f0_min_bin = int(80 * fft_length / fs)
    f0_max_bin = int(300 * fft_length / fs)
    
    # Look for peaks in F0 range
    f0_region = magnitude_spectrum[f0_min_bin:f0_max_bin]
    
    if len(f0_region) == 0:
        return False, 0.0
    
    # Calculate prominence of peaks
    mean_energy = np.mean(f0_region)
    max_energy = np.max(f0_region)
    
    # If no clear peaks, likely over-suppressed
    peak_prominence = (max_energy - mean_energy) / (mean_energy + 1e-10)
    
    has_harmonics = peak_prominence > 2.0  # Clear peak should be 2x above mean
    harmonic_confidence = np.clip(peak_prominence / 5.0, 0, 1)
    
    return has_harmonics, harmonic_confidence

def spectral_flux_detector(current_frame, previous_frame, threshold=0.5):
    """
    Detect sudden spectral changes that might indicate speech onset.
    
    If GTCRN suppressed speech onset, spectral flux will be abnormally low.
    """
    if previous_frame is None:
        return False, 0.0
    
    # Calculate spectral flux (change between frames)
    diff = np.abs(current_frame - previous_frame)
    flux = np.sum(diff) / (np.sum(current_frame) + 1e-10)
    
    # Very low flux in GTCRN output might indicate missing transients
    is_suspicious = flux < threshold
    confidence = (threshold - flux) / threshold if is_suspicious else 0.0
    
    return is_suspicious, np.clip(confidence, 0, 0.3)

def adaptive_speech_recovery(
    noisy_magnitude,      # Original noisy magnitude spectrum
    gtcrn_magnitude,      # GTCRN enhanced magnitude spectrum
    noisy_phase,          # Phase from noisy (for reconstruction)
    gtcrn_phase,          # Phase from GTCRN
    fs,
    fft_length,
    previous_gtcrn_mag=None,
    recovery_strength=0.3  # Maximum recovery weight
):
    """
    Intelligently blend back speech from noisy signal when GTCRN over-suppresses.
    
    Returns:
        recovered_magnitude: Adaptively mixed magnitude spectrum
        recovery_info: Dict with debug information
    """
    n_bins = len(noisy_magnitude)
    recovered_magnitude = gtcrn_magnitude.copy()
    
    # 1. Detect over-suppression
    needs_recovery, suppression_weight = detect_speech_suppression(
        noisy_magnitude, gtcrn_magnitude, noisy_magnitude, gtcrn_magnitude
    )
    
    # 2. Analyze harmonic structure
    has_harmonics_noisy, harmonic_conf_noisy = harmonic_analysis(
        noisy_magnitude, fs, fft_length
    )
    has_harmonics_gtcrn, harmonic_conf_gtcrn = harmonic_analysis(
        gtcrn_magnitude, fs, fft_length
    )
    
    # 3. Check spectral flux
    suspicious_flux, flux_weight = spectral_flux_detector(
        gtcrn_magnitude, previous_gtcrn_mag
    )
    
    # Combine evidence: If noisy has harmonics but GTCRN doesn't → likely over-suppressed
    harmonics_lost = has_harmonics_noisy and not has_harmonics_gtcrn
    harmonic_weight = (harmonic_conf_noisy - harmonic_conf_gtcrn) if harmonics_lost else 0.0
    
    # Calculate total recovery weight
    total_recovery = np.clip(
        suppression_weight + harmonic_weight + flux_weight,
        0,
        recovery_strength
    )
    
    if total_recovery > 0.05:  # Threshold for applying recovery
        # Frequency-dependent recovery (preserve more mid-frequencies for speech)
        frequency_weights = np.ones(n_bins)
        
        # Emphasize speech-critical bands (300-3400 Hz)
        speech_start = int(300 * fft_length / fs)
        speech_end = int(3400 * fft_length / fs)
        frequency_weights[speech_start:speech_end] = 1.5
        
        # Reduce recovery in very low frequencies (< 200 Hz, likely noise)
        low_freq_end = int(200 * fft_length / fs)
        frequency_weights[:low_freq_end] = 0.3
        
        # Reduce recovery in very high frequencies (> 6000 Hz)
        high_freq_start = int(6000 * fft_length / fs)
        frequency_weights[high_freq_start:] = 0.5
        
        # Normalize
        frequency_weights = frequency_weights / np.max(frequency_weights)
        
        # Apply adaptive mixing
        recovery_mask = total_recovery * frequency_weights
        recovered_magnitude = (
            (1 - recovery_mask) * gtcrn_magnitude +
            recovery_mask * noisy_magnitude
        )
    
    recovery_info = {
        'needs_recovery': needs_recovery,
        'total_weight': total_recovery,
        'suppression_weight': suppression_weight,
        'harmonic_weight': harmonic_weight,
        'flux_weight': flux_weight,
        'harmonics_lost': harmonics_lost
    }
    
    return recovered_magnitude, recovery_info

def mband_with_speech_recovery(
    noisy_audio: torch.Tensor,
    gtcrn_audio: torch.Tensor,  # NEW: GTCRN output
    original_noisy: torch.Tensor,  # NEW: Keep original noisy for recovery
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    Nband: int = 8,
    Freq_spacing: str = 'linear',
    FRMSZ: int = 20,
    OVLP: int = 75,
    AVRGING: int = 3,
    Noisefr: int = 2,
    FLOOR: float = 0.3,
    VAD: int = 0,
    enable_speech_recovery: bool = True,  # NEW: Enable recovery mechanism
    recovery_strength: float = 0.3,  # NEW: Max recovery weight
) -> Tuple[torch.Tensor, int, dict]:
    """
    Enhanced multi-band spectral subtraction with intelligent speech recovery.
    
    NEW PARAMETERS:
        gtcrn_audio: GTCRN enhanced audio (what we're post-processing)
        original_noisy: Original noisy audio (for selective recovery)
        enable_speech_recovery: Enable the recovery mechanism
        recovery_strength: How much to blend back (0-1, default 0.3 = 30%)
    
    USAGE:
        enhanced, fs, stats = mband_with_speech_recovery(
            noisy_audio=noisy_speech,
            gtcrn_audio=gtcrn_enhanced,
            original_noisy=noisy_speech,  # Same as noisy_audio
            fs=16000,
            enable_speech_recovery=True,
            recovery_strength=0.3
        )
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to numpy
    gtcrn_speech = gtcrn_audio.squeeze().cpu().numpy()
    original_noisy_np = original_noisy.squeeze().cpu().numpy()
    
    frame_samples = int(np.floor(FRMSZ * fs / 1000))
    overla_samples = int(np.floor(frame_samples * OVLP / 100))
    hop = frame_samples - overla_samples
    
    # FFT length
    fftl = 2
    while fftl < frame_samples:
        fftl *= 2
    
    # Band setup (using linear for simplicity here)
    bandsz = [int(np.floor(fftl / (2 * Nband)))] * Nband
    lobin = [i * bandsz[0] for i in range(Nband)]
    hibin = [l + bandsz[0] - 1 for l in lobin]
    
    # Windows
    analysis_win = np.sqrt(np.hanning(frame_samples))
    U = np.sum(analysis_win**2) / frame_samples
    
    # Estimate noise from GTCRN output (residual noise)
    noise_pow = np.zeros(fftl)
    j = 0
    for k in range(Noisefr):
        if j + frame_samples > len(gtcrn_speech):
            break
        n_fft = fft(gtcrn_speech[j:j + frame_samples] * analysis_win, fftl)
        noise_pow += np.abs(n_fft) ** 2
        j += frame_samples
    n_spect = np.sqrt(noise_pow / Noisefr).reshape(-1, 1)
    
    # Process both signals frame-by-frame
    gtcrn_mag_frames = []
    gtcrn_ph_frames = []
    noisy_mag_frames = []
    noisy_ph_frames = []
    
    sample_pos = 0
    while sample_pos + frame_samples <= min(len(gtcrn_speech), len(original_noisy_np)):
        # GTCRN frame
        gtcrn_frame = gtcrn_speech[sample_pos:sample_pos + frame_samples]
        gtcrn_windowed = gtcrn_frame * analysis_win
        gtcrn_fft = fft(gtcrn_windowed, fftl)
        gtcrn_mag_frames.append(np.abs(gtcrn_fft))
        gtcrn_ph_frames.append(np.angle(gtcrn_fft))
        
        # Original noisy frame (for recovery)
        noisy_frame = original_noisy_np[sample_pos:sample_pos + frame_samples]
        noisy_windowed = noisy_frame * analysis_win
        noisy_fft = fft(noisy_windowed, fftl)
        noisy_mag_frames.append(np.abs(noisy_fft))
        noisy_ph_frames.append(np.angle(noisy_fft))
        
        sample_pos += hop
    
    if not gtcrn_mag_frames:
        print("Warning: No frames generated")
        return gtcrn_audio, fs, {}
    
    gtcrn_mag = np.array(gtcrn_mag_frames).T
    gtcrn_ph = np.array(gtcrn_ph_frames).T
    noisy_mag = np.array(noisy_mag_frames).T
    noisy_ph = np.array(noisy_ph_frames).T
    n_frames = len(gtcrn_mag_frames)
    
    # Apply recovery mechanism BEFORE spectral subtraction
    recovery_stats = {
        'frames_recovered': 0,
        'avg_recovery_weight': 0.0,
        'recovery_reasons': []
    }
    
    if enable_speech_recovery:
        recovered_mag = np.zeros_like(gtcrn_mag)
        total_recovery_weight = 0.0
        
        for i in range(n_frames):
            prev_mag = gtcrn_mag[:, i-1] if i > 0 else None
            
            recovered_frame, recovery_info = adaptive_speech_recovery(
                noisy_magnitude=noisy_mag[:, i],
                gtcrn_magnitude=gtcrn_mag[:, i],
                noisy_phase=noisy_ph[:, i],
                gtcrn_phase=gtcrn_ph[:, i],
                fs=fs,
                fft_length=fftl,
                previous_gtcrn_mag=prev_mag,
                recovery_strength=recovery_strength
            )
            
            recovered_mag[:, i] = recovered_frame
            
            if recovery_info['total_weight'] > 0.05:
                recovery_stats['frames_recovered'] += 1
                total_recovery_weight += recovery_info['total_weight']
                if recovery_info['harmonics_lost']:
                    recovery_stats['recovery_reasons'].append(f"Frame {i}: Harmonics lost")
        
        if recovery_stats['frames_recovered'] > 0:
            recovery_stats['avg_recovery_weight'] = total_recovery_weight / recovery_stats['frames_recovered']
        
        # Use recovered magnitude for further processing
        x_mag = recovered_mag
    else:
        x_mag = gtcrn_mag
    
    x_ph = gtcrn_ph  # Keep GTCRN phase (usually better)
    
    # Smoothing
    if AVRGING:
        filtb = [0.85, 0.15]
        x_magsm = np.zeros_like(x_mag)
        x_magsm[:, 0] = scipy.signal.lfilter(filtb, [1], x_mag[:, 0])
        
        for i in range(1, n_frames):
            x_tmp1 = np.concatenate([x_mag[frame_samples - overla_samples:, i - 1], x_mag[:, i]])
            x_tmp2 = scipy.signal.lfilter(filtb, [1], x_tmp1)
            x_magsm[:, i] = x_tmp2[-x_mag.shape[0]:]
        
        Wn2, Wn1, Wn0 = 0.05, 0.20, 0.75
        if n_frames > 1:
            x_magsm[:, 1] = Wn1 * x_magsm[:, 0] + Wn0 * x_magsm[:, 1]
            for i in range(2, n_frames):
                x_magsm[:, i] = Wn2 * x_magsm[:, i - 2] + Wn1 * x_magsm[:, i - 1] + Wn0 * x_mag[:, i]
    else:
        x_magsm = x_mag
    
    # Noise update (no VAD for post-GTCRN)
    n_spect = np.repeat(n_spect, n_frames, axis=1)
    
    # Calculate SNR and apply gentle spectral subtraction
    SNR_x = np.zeros((Nband, n_frames))
    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1 if i < Nband - 1 else fftl // 2 + 1
        
        for j in range(n_frames):
            sig_pow = (np.linalg.norm(x_magsm[start:stop, j], 2) ** 2) / (frame_samples * U)
            noise_pow = (np.linalg.norm(n_spect[start:stop, j], 2) ** 2) / (frame_samples * U)
            SNR_x[i, j] = 10 * np.log10(sig_pow / (noise_pow + 1e-10))
    
    # Gentle over-subtraction for post-GTCRN
    beta_x = np.ones_like(SNR_x) * 1.5  # Conservative
    
    # Spectral subtraction
    sub_speech_x = np.zeros((fftl // 2 + 1, n_frames))
    
    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1 if i < Nband - 1 else fftl // 2 + 1
        
        for j in range(n_frames):
            n_spec_sq = n_spect[start:stop, j] ** 2
            sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq
            
            z = np.where(sub_speech < 0)[0]
            if z.size > 0:
                sub_speech[z] = FLOOR * x_magsm[start:stop, j][z] ** 2
            
            # Minimal residual
            sub_speech = sub_speech + 0.01 * x_magsm[start:stop, j] ** 2
            sub_speech_x[start:stop, j] += sub_speech
    
    # Reconstruction
    enhanced_mag = np.sqrt(np.maximum(sub_speech_x, 0))
    enhanced_spectrum = np.zeros((fftl, n_frames), dtype=np.complex128)
    enhanced_spectrum[:fftl // 2 + 1, :] = enhanced_mag * np.exp(1j * x_ph[:fftl // 2 + 1, :])
    enhanced_spectrum[fftl // 2 + 1:, :] = np.conj(np.flipud(enhanced_spectrum[1:fftl // 2, :]))
    
    # IFFT
    y1_ifft = ifft(enhanced_spectrum, axis=0)
    y1_r = np.real(y1_ifft)
    
    # Overlap-add
    out = np.zeros((n_frames - 1) * hop + frame_samples)
    win_sum = np.zeros_like(out)
    
    for i in range(n_frames):
        start = i * hop
        out[start:start + frame_samples] += y1_r[:frame_samples, i] * analysis_win
        win_sum[start:start + frame_samples] += analysis_win ** 2
    
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    out = out[:len(gtcrn_speech)]
    
    # Normalize
    max_amp = np.max(np.abs(out))
    if max_amp > 1.0:
        out = out / max_amp
    
    enhanced_tensor = torch.tensor(out, dtype=torch.float32)
    
    # Print recovery statistics
    if enable_speech_recovery and recovery_stats['frames_recovered'] > 0:
        print(f"\nSpeech Recovery Stats:")
        print(f"  Frames recovered: {recovery_stats['frames_recovered']}/{n_frames} "
              f"({recovery_stats['frames_recovered']/n_frames*100:.1f}%)")
        print(f"  Avg recovery weight: {recovery_stats['avg_recovery_weight']:.3f}")
        if len(recovery_stats['recovery_reasons']) > 0:
            print(f"  First 3 reasons: {recovery_stats['recovery_reasons'][:3]}")
    
    return enhanced_tensor, fs, recovery_stats