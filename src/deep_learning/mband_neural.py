import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft
import scipy.signal
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import torchaudio
import torchaudio.transforms as T

# ==================== TinyNoiseGru Model ====================
class TinyNoiseGru(torch.nn.Module):
    """Lightweight GRU-based Noise Power Spectrum Estimator."""
    def __init__(self, input_dim=32, hidden_dim=24, dropout=0.1):
        super().__init__()
        feature_dim = input_dim * 2
        
        self.pre = torch.nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=0, groups=4)
        self.norm = torch.nn.LayerNorm(feature_dim)
        self.gru1 = torch.nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.gru2 = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)
        self.drop = torch.nn.Dropout(dropout)
        
    def forward(self, x, h1=None, h2=None):
        x = x.transpose(1, 2)
        k = self.pre.kernel_size[0] if isinstance(self.pre.kernel_size, (list, tuple)) else self.pre.kernel_size
        pad_left = k - 1
        x = torch.nn.functional.pad(x, (pad_left, 0))
        
        x = self.pre(x).transpose(1, 2)
        x = self.norm(x)
        
        out1, h1 = self.gru1(x, h1)
        out1 = self.drop(out1)
        out2, h2 = self.gru2(out1, h2)
        out2 = self.drop(out2)
        
        noise_est = self.fc(out2)
        confidence = 1.0 / (1.0 + torch.std(noise_est, dim=-1, keepdim=True))
        
        return noise_est, (h1, h2), confidence


def berouti(SNR):
    """Berouti's algorithm for computing over-subtraction factor"""
    a = np.ones_like(SNR)
    a[(SNR >= -5.0) & (SNR <= 20)] = 4 - SNR[(SNR >= -5.0) & (SNR <= 20)] * 3 / 20
    a[SNR < -5.0] = 4.75
    return a

def noiseupdt_neural(x_magsm, n_spect, cmmnlen, nframes, model, mel_transform, device, 
                     fftl, n_bands=32, update_weight=0.5):
    """Neural-guided noise update with optimized mel inversion."""
    
    SPEECH, SILENCE = 1, 0
    state = np.zeros(nframes * cmmnlen, dtype=int)
    n_stft_bins = fftl // 2 + 1

    # Maybe use cell 5 to get the correct format for input 
    
    # === PRECOMPUTE MEL INVERSE (ONCE) ===
    mel_fb = mel_transform.fb.cpu().numpy()
    mel_fb_pinv = np.linalg.pinv(mel_fb.T, rcond=1e-3)
    print(f"[DEBUG] Mel inverse: {mel_fb_pinv.shape}, condition number: {np.linalg.cond(mel_fb.T):.2f}")
    
    # === COLLECT MEL FEATURES ===
    log_mel_all = []
    model.eval()
    
    with torch.no_grad():
        for i in range(nframes):
            x_mag_frame = x_magsm[:n_stft_bins, i]
            x_tensor = torch.tensor(x_mag_frame, dtype=torch.float32, device=device).unsqueeze(1)
            mel_spec = mel_transform(x_tensor).clamp_min(1e-8)
            log_mel = torch.log(mel_spec.squeeze() + 1e-8)
            log_mel_all.append(log_mel)
    
    # === PROCESS FRAMES ===
    silence_count = 0
    
    with torch.no_grad():
        for i in range(nframes):
            # Compute delta
            delta = log_mel_all[i] - log_mel_all[i-1] if i > 0 else torch.zeros_like(log_mel_all[0])
            features = torch.cat([log_mel_all[i], delta]).unsqueeze(0).unsqueeze(0)
            
            # Get neural prediction
            log_noise_mel, _, confidence = model(features)
            linear_noise_mel = model.to_linear_power(log_noise_mel).squeeze().cpu().numpy()
            
            # Mel → FFT inversion (FAST!)
            noise_power_fft = mel_fb_pinv @ linear_noise_mel
            noise_power_fft = np.clip(noise_power_fft, 1e-10, 1e3)
            neural_noise_fft = np.sqrt(noise_power_fft)
            
            # Reconstruct full FFT
            neural_noise_full = np.zeros(fftl)
            neural_noise_full[:n_stft_bins] = neural_noise_fft
            if n_stft_bins < fftl:
                neural_noise_full[n_stft_bins:] = np.flipud(neural_noise_fft[1:fftl - n_stft_bins + 1])
            
            if i == 0:
                print(f"[DEBUG] Neural mel: [{linear_noise_mel.min():.6f}, {linear_noise_mel.max():.6f}]")
                print(f"[DEBUG] Neural FFT: [{noise_power_fft.min():.6f}, {noise_power_fft.max():.6f}]")
                print(f"[DEBUG] Clipped bins: {np.sum(noise_power_fft < 0)}/{len(noise_power_fft)}")


            # Traditional update
            vad_update = n_spect[:, 0] if i == 0 else np.sqrt(0.9 * n_spect[:, i-1]**2 + 0.1 * x_magsm[:, i]**2)
            
            # Blend
            n_spect[:, i] = (1 - update_weight) * vad_update + update_weight * neural_noise_full
            
            # VAD decision
            x_var = x_magsm[:, i] ** 2
            n_var = (n_spect[:, 0] ** 2 if i == 0 else n_spect[:, i-1] ** 2)
            n_var = np.maximum(n_var, 1e-10)
            x_var = np.maximum(x_var, 1e-10)
            
            rti = x_var / n_var - np.log10(x_var / n_var) - 1
            judgevalue = np.mean(rti)
            
            if judgevalue > 0.30:  # Speech
                state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH
                if i > 0:
                    n_spect[:, i] = n_spect[:, i-1]
            else:  # Silence
                state[i*cmmnlen:(i+1)*cmmnlen] = SILENCE
                silence_count += 1
    
    return n_spect, state, silence_count


def estfilt1(nChannels, Srate):
    """Estimate filter bank for logarithmic spacing"""
    FS = Srate / 2
    UpperFreq = FS
    LowFreq = 1
    range_log = np.log10(UpperFreq / LowFreq)
    interval = range_log / nChannels
    
    center = np.zeros(nChannels)
    upper1 = np.zeros(nChannels)
    lower1 = np.zeros(nChannels)
    
    for i in range(nChannels):
        upper1[i] = LowFreq * 10**(interval * (i + 1))
        lower1[i] = LowFreq * 10**(interval * i)
        center[i] = 0.5 * (upper1[i] + lower1[i])
    
    return lower1, center, upper1


def mel(N, low, high):
    """Compute Mel-spaced filter bank edges"""
    ac = 1100
    fc = 800
    LOW = ac * np.log(1 + low / fc)
    HIGH = ac * np.log(1 + high / fc)
    N1 = N + 1
    
    fmel = LOW + np.arange(1, N1 + 1) * (HIGH - LOW) / N1
    cen2 = fc * (np.exp(fmel / ac) - 1)
    lower = cen2[:N]
    upper = cen2[1:N+1]
    center = 0.5 * (lower + upper)
    
    return lower, center, upper


def calculate_delta_factors(lobin, hibin, fs, Nband, fftl):
    """Calculate frequency-dependent delta factors based on Loizou's rule"""
    hibin = np.array(hibin) 
    upper_freq_hz = hibin * fs / (2 * fftl)

    delta_factors = np.where(
        upper_freq_hz <= 1000,
        1.0,
        np.where(upper_freq_hz <= (fs / 2 - 1000), 2.5, 1.5)
    )
    return delta_factors


def mband_neural(
        noisy_audio: torch.Tensor,
        fs: int,
        model_path: Union[str, Path],
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
        VAD: int = 1,
        N_MEL_BANDS: int = 32,
        NEURAL_WEIGHT: float = 0.5,
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Multi-band spectral subtraction with NEURAL noise estimation integrated into VAD.
    
    Uses trained TinyNoiseGru model to guide noise spectrum updates during silence frames,
    blended with traditional VAD-based updates.
    
    Args:
        noisy_audio: Noisy input signal
        fs: Sampling frequency
        model_path: Path to trained TinyNoiseGru checkpoint
        output_dir: Output directory for enhanced audio
        output_file: Output filename
        input_name: Name for metadata
        Nband: Number of frequency bands
        Freq_spacing: 'linear', 'log', or 'mel'
        FRMSZ: Frame length in ms
        OVLP: Overlap percentage
        AVRGING: Apply smoothing (1) or not (0)
        Noisefr: Number of noise frames (for compatibility)
        FLOOR: Spectral floor
        VAD: Use VAD (1) or not (0)
        N_MEL_BANDS: Number of mel bands (must match model)
        NEURAL_WEIGHT: Blend weight for neural predictions (0-1, default 0.5)
    
    Returns:
        Tuple of (enhanced_audio, sample_rate)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load neural model
    print(f"[INFO] Loading neural model from: {model_path}")
    model = TinyNoiseGru(input_dim=N_MEL_BANDS, hidden_dim=24, dropout=0.0)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully")

        # ADD THIS:
    # Patch missing method for old models
    if not hasattr(model, 'to_linear_power'):
        def to_linear_power(self, log_noise):
            return torch.exp(log_noise)
        model.to_linear_power = to_linear_power.__get__(model, TinyNoiseGru)
    print("[INFO] Patched model with to_linear_power() method")
    
    # Handle tensor input
    if noisy_audio.dim() > 1 and noisy_audio.shape[0] > 1:
        noisy_speech = torch.mean(noisy_audio, dim=0)
    else:
        noisy_speech = noisy_audio.squeeze()
    
    noisy_speech = noisy_speech.to(device)
    input_name = input_name if input_name is not None else "neural_vad"

    frmelen = int(np.floor(FRMSZ * fs / 1000))
    ovlplen = int(np.floor(frmelen * OVLP / 100))
    cmmnlen = frmelen - ovlplen
    
    # fftl = 2
    # while fftl < frmelen:
    #     fftl *= 2
    fftl = 256 # Force 256-point FFT for consistency with model training

    # Band setup
    if Freq_spacing.lower() == 'linear':
        bandsz = [int(np.floor(fftl / (2 * Nband)))] * Nband
        lobin = [i * bandsz[0] for i in range(Nband)]
        hibin = [l + bandsz[0] - 1 for l in lobin]
    elif Freq_spacing.lower() == 'log':
        lof, midf, hif = estfilt1(Nband, fs)
        lobin = np.round(lof * fftl / fs).astype(int)
        hibin = np.round(hif * fftl / fs).astype(int)
        bandsz = hibin - lobin + 1
    elif Freq_spacing.lower() == 'mel':
        lof, midf, hif = mel(Nband, 0, fs / 2)
        lobin = np.round(lof * fftl / fs).astype(int)
        hibin = np.round(hif * fftl / fs).astype(int)
        lobin[0] = 0
        hibin[-1] = fftl // 2
        bandsz = hibin - lobin + 1
    else:
        raise ValueError('Error in selecting frequency spacing')

    win = np.sqrt(np.hanning(frmelen))
    U = np.sum(win**2) / frmelen

    # Frame extraction
    x_mag_frames = []
    x_ph_frames = []
    sample_pos = 0
    
    while sample_pos + frmelen <= len(noisy_speech):
        current_frame = noisy_speech[sample_pos:sample_pos + frmelen]
        windowed_frame = current_frame * win
        frame_fft = fft(windowed_frame.cpu().numpy(), fftl)
        frame_mag = np.abs(frame_fft)
        frame_ph = np.angle(frame_fft)
        x_mag_frames.append(frame_mag)
        x_ph_frames.append(frame_ph)
        sample_pos += cmmnlen

    if x_mag_frames:
        x_mag = np.array(x_mag_frames).T
        x_ph = np.array(x_ph_frames).T
        nframes = len(x_mag_frames)
    else:
        print("[ERROR] No frames generated")
        return None

    # Smooth input spectrum
    if AVRGING:
        filtb = [0.9, 0.1]
        x_magsm = np.zeros_like(x_mag)
        x_magsm[:, 0] = scipy.signal.lfilter(filtb, [1], x_mag[:, 0])

        for i in range(1, nframes):
            x_tmp1 = np.concatenate([x_mag[frmelen - ovlplen:, i - 1], x_mag[:, i]])
            x_tmp2 = scipy.signal.lfilter(filtb, [1], x_tmp1)
            x_magsm[:, i] = x_tmp2[-x_mag.shape[0]:]

        Wn2, Wn1, Wn0 = 0.09, 0.25, 0.66
        if nframes > 1:
            x_magsm[:, 1] = Wn1 * x_magsm[:, 0] + Wn0 * x_magsm[:, 1]
            for i in range(2, nframes):
                x_magsm[:, i] = (Wn2 * x_magsm[:, i - 2] + 
                                 Wn1 * x_magsm[:, i - 1] + 
                                 Wn0 * x_mag[:, i])
    else:
        x_magsm = x_mag

    # Initial noise estimate from first frame
    noise_pow = np.zeros(fftl)
    n_fft = fft(noisy_speech[:frmelen].cpu().numpy() * win, fftl)
    n_mag = np.abs(n_fft)
    noise_pow = n_mag ** 2
    n_spect = np.sqrt(noise_pow).reshape(-1, 1)

    # NEURAL-GUIDED VAD noise update
    if VAD:
        print("[INFO] Running neural-guided VAD...")
        n_stft_bins = fftl // 2 + 1
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=N_MEL_BANDS, 
            sample_rate=fs, 
            n_stft=n_stft_bins
        ).to(device)
        
        n_spect_expanded = np.tile(n_spect, (1, nframes))
        n_spect, state, silence_count = noiseupdt_neural(
            x_magsm, n_spect_expanded, cmmnlen, nframes, 
            model, mel_transform, device, fftl, N_MEL_BANDS, 
            update_weight=NEURAL_WEIGHT
        )
        
        # Debug: check VAD statistics
        speech_count = (state == 1).sum()
        silence_count_total = (state == 0).sum()
        print(f"[DEBUG] VAD: {speech_count} speech samples, {silence_count_total} silence samples ({100*silence_count_total/(speech_count+silence_count_total):.1f}%)")
        print(f"[DEBUG] Neural GRU called {silence_count} times (once per silence frame)")
        if silence_count == 0:
            print(f"[WARNING] Neural model was NEVER called! VAD is marking everything as speech.")
            print(f"[WARNING] Try lowering VAD threshold or disabling VAD (VAD=0)")
    else:
        n_spect = np.repeat(n_spect, nframes, axis=1)
        print("[INFO] VAD disabled - using constant noise estimate")

    print(f"[INFO] Neural weight: {NEURAL_WEIGHT}")

    # Calculate segmental SNR
    SNR_x = np.zeros((Nband, nframes))
    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1 if i < Nband - 1 else fftl // 2 + 1

        for j in range(nframes):
            signal_power = (np.linalg.norm(x_magsm[start:stop, j], 2) ** 2) / (frmelen * U)
            noise_power = (np.linalg.norm(n_spect[start:stop, j], 2) ** 2) / (frmelen * U)
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

    # Reconstruct spectrum
    enhanced_mag = np.sqrt(np.maximum(sub_speech_x, 0))
    enhanced_spectrum = np.zeros((fftl, nframes), dtype=np.complex128)
    enhanced_spectrum[:fftl // 2 + 1, :] = enhanced_mag * np.exp(1j * x_ph[:fftl // 2 + 1, :])
    enhanced_spectrum[fftl // 2 + 1:, :] = np.conj(np.flipud(enhanced_spectrum[1:fftl // 2, :]))

    y1_ifft = ifft(enhanced_spectrum, axis=0)
    y1_r = np.real(y1_ifft)

    # Overlap-add
    out = np.zeros((nframes - 1) * cmmnlen + frmelen)
    win_sum = np.zeros_like(out)

    for i in range(nframes):
        start = i * cmmnlen
        out[start:start + frmelen] += y1_r[:frmelen, i] * win
        win_sum[start:start + frmelen] += win**2

    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    out = out[:len(noisy_speech)]

    # Normalize
    max_amplitude = np.max(np.abs(out))
    if max_amplitude > 1.0:
        out = out / max_amplitude
        print(f"[INFO] Output normalized by factor {max_amplitude:.3f}")

    enhanced_tensor = torch.tensor(out, dtype=torch.float32)

    # Save to file
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_parts = [
            f"BANDS{Nband}",
            f"SPACING{Freq_spacing.upper()}",
            f"FRAME{FRMSZ}ms",
            f"NEURAL_W{NEURAL_WEIGHT:.1f}"
        ]

        base_name = output_file.replace('.wav', '') if output_file.endswith('.wav') else output_file
        output_filename = f"{base_name}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        torchaudio.save(str(full_output_path), enhanced_tensor.unsqueeze(0), fs)
        print(f"[INFO] Enhanced audio saved to: {full_output_path}")

    return enhanced_tensor, fs


# ==================== Example usage ====================
if __name__ == "__main__":
    TARGET_SR = 16000
    TARGET_SNR_DB = 5
    OUTPUT_DIR = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP2\\spectral\\NOISE_ESTIMATION"

    torch.manual_seed(42)

    clean_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\EARS_DATASET\\p092\\emo_adoration_freeform.wav")
    noise_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\NOIZEUS_NOISE_DATASET\\Noise Recordings\\cafeteria_babble.wav")

    # Audio preparation function
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

    print("Preparing audio data pair...")
    noisy_tensor, clean_tensor, noise_tensor, fs = prepare_audio_data(clean_path, noise_path, TARGET_SR, TARGET_SNR_DB)

    print(f"Data pair created at {TARGET_SNR_DB} dB SNR.")
    print("-" * 30)

    # neural_weight = [0.1, 0.3, 0.5, 0.7, 0.9]
    neural_weight = [0.1]

    for weight in neural_weight:
        print(f"\nRunning with NEURAL-GUIDED VAD ({weight} blend)...")
        enhanced, fs = mband_neural(
            noisy_audio=noisy_tensor,
            fs=fs,
            model_path="models/GRU_NoiseEst/tiny_noise_gru_best.pth",
            output_dir=OUTPUT_DIR,
            output_file="mband_neural_vad.wav",
            input_name="neural_guided",
            Nband=4,
            Freq_spacing='linear',
            FRMSZ=8,
            OVLP=50,
            AVRGING=1,
            VAD=1,
            N_MEL_BANDS=32,
            NEURAL_WEIGHT=weight,
        )

    print("\n[SUCCESS] Neural-guided VAD enhancement complete!")
