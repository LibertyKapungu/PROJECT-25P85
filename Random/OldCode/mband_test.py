import numpy as np
from scipy.fft import fft, ifft
import scipy.signal
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import torchaudio
import torchaudio.transforms as T

def berouti(SNR):
    """Berouti's algorithm for computing over-subtraction factor"""
    a = np.ones_like(SNR)
    a[(SNR >= -5.0) & (SNR <= 20)] = 4 - SNR[(SNR >= -5.0) & (SNR <= 20)] * 3 / 20
    a[SNR < -5.0] = 4.75
    return a

def noiseupdt(x_magsm, n_spect, cmmnlen, nframes):
    """Voice Activity Detection - EXACT copy from working mband.py"""
    SPEECH = 1
    SILENCE = 0
    state = np.zeros(nframes * cmmnlen, dtype=int)
    judgevalue1 = np.zeros(nframes * cmmnlen)

    # Process first frame
    i = 0
    x_var = x_magsm[:, i] ** 2
    n_var = n_spect[:, i] ** 2
    rti = x_var / n_var - np.log10(x_var / n_var) - 1
    judgevalue = np.mean(rti)
    judgevalue1[i*cmmnlen:(i+1)*cmmnlen] = judgevalue
    
    if judgevalue > 0.4:
        state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH
    else:
        state[i*cmmnlen:(i+1)*cmmnlen] = SILENCE
        n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i]**2 + (1-0.9) * x_magsm[:, i]**2)
    
    # Process remaining frames
    for i in range(1, nframes):
        x_var = x_magsm[:, i] ** 2
        n_var = n_spect[:, i-1] ** 2
        rti = x_var / n_var - np.log10(x_var / n_var) - 1
        judgevalue = np.mean(rti)
        judgevalue1[i*cmmnlen:(i+1)*cmmnlen] = judgevalue
        
        if judgevalue > 0.45:
            state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH
            n_spect[:, i] = n_spect[:, i-1]
        else:
            state[i*cmmnlen:(i+1)*cmmnlen] = SILENCE
            n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i-1]**2 + (1-0.9) * x_magsm[:, i]**2)
    
    return n_spect, state

def calculate_delta_factors(lobin, hibin, fs, Nband, fftl):
    """Calculate frequency-dependent delta factors - EXACT from mband.py"""
    hibin = np.array(hibin) 
    upper_freq_hz = hibin * fs / (2 * fftl)
    delta_factors = np.where(
        upper_freq_hz <= 1000, 1.0,
        np.where(upper_freq_hz <= (fs / 2 - 1000), 2.5, 1.5)
    )
    return delta_factors

# ==============================================================================
# MAIN FUNCTION - Now matches mband.py EXACTLY
# ==============================================================================

def mband(
        noisy_audio: torch.Tensor,
        fs: int,
        # Testing parameters
        mode: str = 'standard',
        actual_noise_signal: Optional[torch.Tensor] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_file: Optional[str] = None,
        input_name: Optional[str] = None,
        Nband: int = 4,
        Freq_spacing: str = 'linear',
        FRMSZ: int = 8, 
        OVLP: int = 50, 
        AVRGING: int = 1,
        Noisefr: int = 1, 
        FLOOR: float = 0.002,
        VAD: int = 1
) -> Optional[Tuple[torch.Tensor, int]]:
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if mode == 'oracle' and actual_noise_signal is None:
        raise ValueError("Must provide 'actual_noise_signal' when mode is 'oracle'")

    # Handle tensor input 
    if noisy_audio.dim() > 1 and noisy_audio.shape[0] > 1:
        noisy_speech = torch.mean(noisy_audio, dim=0)
    else:
        noisy_speech = noisy_audio.squeeze()
    
    noisy_speech = noisy_speech.to(device)  
    input_name = input_name if input_name is not None else "spectral"

    # Frame setup
    frmelen = int(np.floor(FRMSZ * fs / 1000))
    ovlplen = int(np.floor(frmelen * OVLP / 100))
    cmmnlen = frmelen - ovlplen
    
    # FFT length
    fftl = 2
    while fftl < frmelen:
        fftl *= 2

    # Band setup
    if Freq_spacing.lower() == 'linear':
        bandsz = [int(np.floor(fftl / (2 * Nband)))] * Nband
        lobin = [i * bandsz[0] for i in range(Nband)]
        hibin = [l + bandsz[0] - 1 for l in lobin]
    else:
        raise ValueError('Error in selecting frequency spacing')

    # Window 
    win = np.sqrt(np.hanning(frmelen))
    U = np.sum(win**2) / frmelen  # This is calculated but not used

    # Noise estimation 
    if mode == 'standard':
        print("\n--- Running in STANDARD Mode ---")
        noise_pow = np.zeros(fftl)
        j = 0
        for k in range(Noisefr):
            n_fft = fft(noisy_speech[j:j + frmelen]* win, fftl)
            n_mag = np.abs(n_fft)
            n_magsq = n_mag ** 2
            noise_pow += n_magsq
            j += frmelen
        n_spect = np.sqrt(noise_pow / Noisefr).reshape(-1, 1)
    else:  # oracle mode
        print("\n--- Running in ORACLE Mode ---")
        noise_signal_np = actual_noise_signal.squeeze().cpu().numpy()
        noise_pow = np.zeros(fftl)
        sample_pos = 0
        noise_frame_count = 0
        # Process noise with same framing as speech
        while sample_pos + frmelen <= len(noise_signal_np):
            noise_frame = noise_signal_np[sample_pos:sample_pos + frmelen]
            n_fft = fft(noise_frame * win, fftl)
            n_mag = np.abs(n_fft)
            noise_pow += n_mag ** 2
            sample_pos += cmmnlen
            noise_frame_count += 1
        n_spect_initial = np.sqrt(noise_pow / max(noise_frame_count, 1)).reshape(-1, 1)
        n_spect = n_spect_initial

    # Frame-by-frame processing 
    x_mag_frames = []
    x_ph_frames = []
    sample_pos = 0
    frame_count = 0
    
    while sample_pos + frmelen <= len(noisy_speech):
        current_frame = noisy_speech[sample_pos:sample_pos + frmelen]
        windowed_frame = current_frame * win
        frame_fft = fft(windowed_frame, fftl)
        frame_mag = np.abs(frame_fft)
        frame_ph = np.angle(frame_fft)
        x_mag_frames.append(frame_mag)
        x_ph_frames.append(frame_ph)
        sample_pos += cmmnlen
        frame_count += 1

    # Convert to matrices
    if x_mag_frames:
        x_mag = np.array(x_mag_frames).T
        x_ph = np.array(x_ph_frames).T
        nframes = len(x_mag_frames)
    else:
        print("Warning: No frames generated - audio too short")
        return None

    # Spectral smoothing 
    if AVRGING:
        filtb = [0.9, 0.1]
        x_magsm = np.zeros_like(x_mag)
        x_magsm[:, 0] = scipy.signal.lfilter(filtb, [1], x_mag[:, 0])
        
        for i in range(1, nframes):
            x_tmp1 = np.concatenate([x_mag[frmelen - ovlplen:, i - 1], x_mag[:, i]])
            x_tmp2 = scipy.signal.lfilter(filtb, [1], x_tmp1)
            x_magsm[:, i] = x_tmp2[-x_mag.shape[0]:]
        
        # Weighted spectral estimate
        Wn2, Wn1, Wn0 = 0.09, 0.25, 0.66
        if nframes > 1:
            x_magsm[:, 1] = Wn1 * x_magsm[:, 0] + Wn0 * x_magsm[:, 1]
            for i in range(2, nframes):
                x_magsm[:, i] = (Wn2 * x_magsm[:, i - 2] + Wn1 * x_magsm[:, i - 1] + Wn0 * x_mag[:, i])
    else:
        x_magsm = x_mag

    # Noise update 
    if VAD and mode == 'standard':  # Only use VAD in standard mode
        n_spect_expanded = np.tile(n_spect, (1, nframes))
        n_spect, state = noiseupdt(x_magsm, n_spect_expanded, cmmnlen, nframes)
    else:
        n_spect = np.repeat(n_spect, nframes, axis=1)

    # Calculate SNR 
    SNR_x = np.zeros((Nband, nframes))
    for i in range(Nband):
        if i < Nband - 1:
            start = lobin[i]
            stop = hibin[i] + 1
        else:
            start = lobin[i]
            stop = fftl // 2 + 1
        
        for j in range(nframes):
            signal_power = np.linalg.norm(x_magsm[start:stop, j], 2) ** 2
            noise_power = np.linalg.norm(n_spect[start:stop, j], 2) ** 2
            SNR_x[i, j] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    beta_x = berouti(SNR_x)
    
    # Spectral subtraction 
    sub_speech_x = np.zeros((fftl // 2 + 1, nframes))
    delta_factors = calculate_delta_factors(lobin, hibin, fs, Nband, fftl)
    
    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1
        
        for j in range(nframes):
            n_spec_sq = n_spect[start:stop, j] ** 2
            sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * delta_factors[i]
            z = np.where(sub_speech < 0)[0]
            if z.size > 0:
                sub_speech[z] = FLOOR * x_magsm[start:stop, j][z] ** 2
            if i < Nband - 1:
                sub_speech = sub_speech + 0.05 * x_magsm[start:stop, j] ** 2
            else:
                sub_speech = sub_speech + 0.01 * x_magsm[start:stop, j] ** 2
            sub_speech_x[start:stop, j] += sub_speech

    # Reconstruction 
    enhanced_mag = np.sqrt(np.maximum(sub_speech_x, 0))
    enhanced_spectrum = np.zeros((fftl, nframes), dtype=np.complex128)
    enhanced_spectrum[:fftl // 2 + 1, :] = enhanced_mag * np.exp(1j * x_ph[:fftl // 2 + 1, :])
    enhanced_spectrum[fftl // 2 + 1:, :] = np.conj(np.flipud(enhanced_spectrum[1:fftl // 2, :]))
    
    # IFFT 
    y1_ifft = ifft(enhanced_spectrum, axis=0)
    y1_r = np.real(y1_ifft)

    # Overlap-Add 
    out = np.zeros((nframes - 1) * cmmnlen + frmelen)
    win_sum = np.zeros_like(out)
    
    for i in range(nframes):
        start = i * cmmnlen
        out[start:start + frmelen] += y1_r[:frmelen, i] * win
        win_sum[start:start + frmelen] += win**2
    
    # Normalize 
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    
    # Trim 
    out = out[:len(noisy_speech)]
    
    # Normalize amplitude - EXACT from mband.py
    max_amplitude = np.max(np.abs(out))
    if max_amplitude > 1.0:
        out = out / max_amplitude
        print(f"Output normalized by factor {max_amplitude:.3f}")

    enhanced_tensor = torch.tensor(out, dtype=torch.float32)

    # Save output
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path / output_file, enhanced_tensor.unsqueeze(0), fs)
        print(f"Enhanced audio saved to: {output_path / output_file}")
    
    return enhanced_tensor, fs

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_audio_data(clean_path, noise_path, target_sr, snr_db):
    """Robustly prepares audio data using RMS-based SNR calculation."""
    clean_audio, clean_sr = torchaudio.load(clean_path)
    noise_audio, noise_sr = torchaudio.load(noise_path)
    
    if clean_sr != target_sr:
        clean_audio = T.Resample(clean_sr, target_sr)(clean_audio)
    if noise_sr != target_sr:
        noise_audio = T.Resample(noise_sr, target_sr)(noise_audio)
    
    clean = clean_audio.mean(dim=0) if clean_audio.shape[0] > 1 else clean_audio.squeeze(0)
    noise = noise_audio.mean(dim=0) if noise_audio.shape[0] > 1 else noise_audio.squeeze(0)
    
    clean_len, noise_len = clean.shape[0], noise.shape[0]
    if noise_len > clean_len:
        start = torch.randint(0, noise_len - clean_len + 1, (1,)).item()
        noise = noise[start:start + clean_len]
    elif clean_len > noise_len:
        noise = noise.repeat(int(np.ceil(clean_len / noise_len)))[:clean_len]
    
    rms_clean = torch.sqrt(torch.mean(clean**2))
    rms_noise = torch.sqrt(torch.mean(noise**2))
    scale_factor = (rms_clean / (10**(snr_db / 20))) / (rms_noise + 1e-8)
    noise_scaled = noise * scale_factor
    noisy_speech = clean + noise_scaled
    
    return noisy_speech, clean, noise_scaled, target_sr

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    TARGET_SR = 16000
    TARGET_SNR_DB = 5
    OUTPUT_DIR = "final_audio_check_FIXED"

    clean_path = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\sound_data\raw\EARS_DATASET\p092\emo_adoration_freeform.wav")
    noise_path = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\sound_data\raw\NOIZEUS_NOISE_DATASET\Noise Recordings\cafeteria_babble.wav")

    print("Preparing audio data pair...")
    noisy_tensor, clean_tensor, noise_tensor, fs = prepare_audio_data(
        clean_path, noise_path, TARGET_SR, TARGET_SNR_DB
    )
    print(f"Data pair created at {TARGET_SNR_DB} dB SNR.")

    # Save intermediate files
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    torchaudio.save(Path(OUTPUT_DIR) / "01_clean_aligned.wav", clean_tensor.unsqueeze(0), fs)
    torchaudio.save(Path(OUTPUT_DIR) / "02_noise_scaled.wav", noise_tensor.unsqueeze(0), fs)
    torchaudio.save(Path(OUTPUT_DIR) / "03_noisy_input.wav", noisy_tensor.unsqueeze(0), fs)

    # Run STANDARD mode
    print("\n" + "="*70)
    print("RUNNING STANDARD MODE (should match mband.py now)")
    print("="*70)
    mband(
        noisy_audio=noisy_tensor,
        fs=fs,
        mode='standard',
        output_dir=OUTPUT_DIR,
        output_file="enhanced_STANDARD_FIXED.wav",
        Noisefr=1,  
        AVRGING=1,
        VAD=1
    )

    # Run ORACLE mode
    print("\n" + "="*70)
    print("RUNNING ORACLE MODE")
    print("="*70)
    mband(
        noisy_audio=noisy_tensor,
        fs=fs,
        mode='oracle',
        actual_noise_signal=noise_tensor,
        output_dir=OUTPUT_DIR,
        output_file="enhanced_ORACLE_FIXED.wav",
        Noisefr=1,  
        AVRGING=1,
        VAD=1
    )
    
    print("\n✅ Processing complete! Check output directory:", OUTPUT_DIR)