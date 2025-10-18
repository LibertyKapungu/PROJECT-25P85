import numpy as np
import scipy.io.wavfile as wavfile
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

def noiseupdt(x_magsm, n_spect, cmmnlen, nframes, n_spect_actual=None):
    """Voice Activity Detection and noise spectrum update
    Voice Activity Detection and noise spectrum update.

    This function can operate in two modes:
    1. Estimation Mode (if n_spect_actual is None): It estimates the noise
       spectrum frame by frame based on the VAD decision.
    2. Ground Truth Mode (if n_spect_actual is provided): It uses the
       actual noise spectrum to make a perfect VAD decision.
    """
    SPEECH = 1
    SILENCE = 0
    state = np.zeros(nframes * cmmnlen, dtype=int)

    # Mode 1: Using Ground Truth Noise Spectrum

    if n_spect_actual is not None:
        for i in range(nframes):
            x_var = x_magsm[:,i]**2
            n_var = n_spect_actual[:,i] **2 

            epsilon = 1e-10
            n_var[n_var<epsilon] = epsilon
            x_var[x_var<epsilon] = epsilon

            rti = x_var/n_var - np.log10(x_var/n_var) -1
            judgevalue = np.mean(rti)
            if judgevalue > 0.45:
                state[i*cmmnlen:(i+1)*cmmnlen]= SPEECH
            else:
                state[i*cmmnlen:(i+1)*cmmnlen] = SILENCE
        return n_spect_actual, state
    
    # MODE 2: Estimating noise spectrum (original logic)
    else:

        judgevalue1 = np.zeros(nframes * cmmnlen)

        # Process first frame
        i = 0  
        x_var = x_magsm[:, i] ** 2  # The power of the current audio frame
        n_var = n_spect[:, i] ** 2  # The power of the estimated background noise
        rti = x_var / n_var - np.log10(x_var / n_var) - 1  # x_var / n_var: This is the Signal-to-Noise Ratio (SNR). A high SNR means the signal is much louder than the noise, which usually indicates speech.
        judgevalue = np.mean(rti)  # If the signal power is very close to the noise power (i.e., silence), this formula results in a value near 0. If the signal power is much larger than the noise power (i.e., speech), it results in a large positive value.
        judgevalue1[i*cmmnlen:(i+1)*cmmnlen] = judgevalue
        
        if judgevalue > 0.4:  # Different threshold for first frame
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
            
            if judgevalue > 0.45:  # Different threshold for subsequent frames
                state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH
                n_spect[:, i] = n_spect[:, i-1]  # Keep previous noise estimate
            else:
                state[i*cmmnlen:(i+1)*cmmnlen] = SILENCE
                n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i-1]**2 + (1-0.9) * x_magsm[:, i]**2)
        
        return n_spect, state

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
    upper_freq_hz = hibin * fs / (2 * fftl) # Convert FFT bin indices to frequency in Hz using Nyquist scaling

    # Apply Loizou's rule:
    # - 1.0 for f <= 1000 Hz
    # - 2.5 for 1000 < f <= (fs/2 - 1000)
    # - 1.5 for f > (fs/2 - 1000)
    delta_factors = np.where(
        upper_freq_hz <= 1000,
        1.0,
        np.where(upper_freq_hz <= (fs / 2 - 1000), 2.5, 1.5)
    )
    return delta_factors

def mband(
        noisy_audio: torch.Tensor,
        fs: int,
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
        actual_noise_audio: Optional[np.ndarray] = None,
) -> Optional[Tuple[torch.Tensor, int]]:
    """ Implements the multi-band spectral subtraction algorithm for speech enhancement.
    This function implements an advanced spectral subtraction method that divides the frequency
    spectrum into multiple bands to better handle colored noise. The algorithm is particularly
    effective for speech corrupted by non-stationary or colored noise, as it applies different
    subtraction parameters in each frequency band based on the local SNR.

    Algorithm Overview:
    1. Divides the FFT spectrum into Nband frequency bands (linear, log, or mel spacing)
    2. Estimates initial noise spectrum from first few frames
    3. For each frame:
        - Computes the segmental SNR in each frequency band
        - Determines over-subtraction factor using Berouti's rule based on SNR
        - Applies band-specific spectral subtraction with floor constraint
    4. Features voice activity detection (VAD) for noise update
    5. Uses weighted overlap-add (WOLA) with Hanning windows for synthesis
    6. Optionally saves enhanced output to specified directory/file

    Args:
         noisy_audio - noisy speech file in .wav format
         fs - sampling frequency of the audio file
         output_dir - directory to save enhanced output file (if None, no file is saved)
         output_file - enhanced output file in .wav format
         Nband - Number of frequency bands (recommended 4-8)
         Freq_spacing - Type of frequency spacing for the bands, choices:
                        'linear', 'log' and 'mel', default='linear'
         FRMSZ - Frame length in milli-seconds, default=8. 
         OVLP - Window overlap in percent of frame size, default=50
         AVRGING - Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
         Noisefr - Number of noise frames at beginning of file for noise spectrum estimate, default=1. 
         FLOOR - Spectral floor, default=0.002
         VAD - Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise

         Returns:
            Optional[Tuple[torch.Tensor, int]]: Tuple containing:
                - Enhanced speech signal as torch.Tensor
                - Sampling frequency
                Returns None if output_dir and output_file are provided (saves to file instead)

   Example call:
            enhanced_audio, fs = mband(noisy_audio, fs, output_dir='output/', output_file='enhanced.wav', Nband=4, Freq_spacing='linear', FRMSZ=8, OVLP=50, AVRGING=1, Noisefr=1, FLOOR=0.002, VAD=1)

    References:
    [1] Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
        method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
        Conf. Acoust.,Speech, Signal Processing
    
    Authors: Sunil Kamath and Philipos C. Loizou
    Copyright (c) 2006 by Philipos C. Loizou
    $Revision: 0.0 $  $Date: 10/09/2006 $
    -----------------------------------------------
    """   

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Handle tensor input and keep on device
    if noisy_audio.dim() > 1 and noisy_audio.shape[0] > 1:
        noisy_speech = torch.mean(noisy_audio, dim=0)
    else:
        noisy_speech = noisy_audio.squeeze()
    
    noisy_speech = noisy_speech.to(device)
    input_name = input_name if input_name is not None else "spectral"

    frmelen = int(np.floor(FRMSZ * fs / 1000))  # Frame size in samples
    ovlplen = int(np.floor(frmelen * OVLP / 100)) # Number of overlap samples
    cmmnlen = frmelen - ovlplen  # Number of common samples between adjacent frames
    
    # Determine FFT length 
    fftl = 2  
    while fftl < frmelen:
        fftl *= 2       # set to the smallest power of two greater than or equal to the frame length in sample

    # Band setup 
    if Freq_spacing.lower() == 'linear':
        bandsz = [int(np.floor(fftl / (2 * Nband)))] * Nband   #  the code divides the FFT spectrum evenly into Nband bands. It calculates the size of each band (bandsz) by dividing half the FFT length (since the FFT of a real signal is symmetric) by the number of bands.
        lobin = [i * bandsz[0] for i in range(Nband)]  # The lower (lobin) and upper (hibin) bin indices for each band are then determined so that each band covers an equal range of frequency bin
        hibin = [l + bandsz[0] - 1 for l in lobin]
    elif Freq_spacing.lower() == 'log':
        lof, midf, hif = estfilt1(Nband, fs)  # the code uses the estfilt1 function to determine the lower, center, and upper frequency edges of each band on a logarithmic scale.
        lobin = np.round(lof * fftl / fs).astype(int)  # These frequency values are then converted to FFT bin indices (lobin and hibin) by scaling them according to the FFT length and sampling rate. 
        hibin = np.round(hif * fftl / fs).astype(int)
        bandsz = hibin - lobin + 1  # The size of each band is calculated as the difference between the upper and lower bin indices plus one.
    elif Freq_spacing.lower() == 'mel':
        lof, midf, hif = mel(Nband, 0, fs / 2)
        lobin = np.round(lof * fftl / fs).astype(int)
        hibin = np.round(hif * fftl / fs).astype(int)
        lobin[0] = 0
        hibin[-1] = fftl // 2
        bandsz = hibin - lobin + 1
    else:
        raise ValueError('Error in selecting frequency spacing')

    # Calculate Hanning window
    win = np.sqrt(np.hanning(frmelen))
    U = np.sum(win**2) / frmelen  # Normalization factor for overlap-add

    # Estimate noise magnitude for first 'Noisefr' frames
    noise_pow = np.zeros(fftl)
    j = 0
    for k in range(Noisefr):
        n_fft = fft(noisy_speech[j:j + frmelen]* win, fftl)
        n_mag = np.abs(n_fft)
        n_ph = np.angle(n_fft)
        n_magsq = n_mag ** 2
        noise_pow += n_magsq
        j += frmelen
    n_spect = np.sqrt(noise_pow / Noisefr).reshape(-1, 1)

    # Initialize frame-by-frame processing
    x_mag_frames = []  # Store magnitude spectra for each frame
    x_ph_frames = []   # Store phase spectra for each frame
    frame_count = 0
    
    # Process audio frame by frame with overlap
    sample_pos = 0
    while sample_pos + frmelen <= len(noisy_speech):
        current_frame = noisy_speech[sample_pos:sample_pos + frmelen]
        windowed_frame = current_frame * win
        frame_fft = fft(windowed_frame, fftl)
        frame_mag = np.abs(frame_fft)
        frame_ph = np.angle(frame_fft)
        x_mag_frames.append(frame_mag)
        x_ph_frames.append(frame_ph)
        # Advance by hop size (cmmnlen) for proper overlap
        sample_pos += cmmnlen 
        frame_count += 1

    # Convert lists to matrices
    if x_mag_frames:
        x_mag = np.array(x_mag_frames).T 
        x_ph = np.array(x_ph_frames).T   
        nframes = len(x_mag_frames)
    else:
        print("Warning: No frames generated - audio too short")

    # ADDED CODE 
    n_spect_actual = None
    if actual_noise_audio is not None:
        print("INFO: Processing ground truth noise signal into spectral frames.")
        noise_speech = actual_noise_audio.to(device).squeeze()

        n_mag_frames_actual = []
        sample_pos = 0
        while sample_pos + frmelen <= len(noise_speech):
            current_frame = noise_speech[sample_pos:sample_pos+frmelen]
            windowed_frame = current_frame*win
            frame_fft = fft(windowed_frame,fftl)
            n_mag_frames_actual.append(np.abs(frame_fft))
            sample_pos += cmmnlen
        if n_mag_frames_actual:
            n_spect_actual = np.array(n_mag_frames_actual).T
            if n_spect_actual.shape[1] > nframes:
                n_spect_actual = n_spect_actual[:, :nframes]

    # Smooth the input spectrum
    if AVRGING:             
        filtb = [0.9, 0.1]  # This defines the coefficients of a first-order IIR low-pass filter used for temporal smoothing of the magnitude spectrum. This filter smooths the spectrum by blending the current and previous values: 0.9 weight on the previous value 0.1 weight on the current value
        x_magsm = np.zeros_like(x_mag)
        x_magsm[:, 0] = scipy.signal.lfilter(filtb, [1], x_mag[:, 0])

        for i in range(1, nframes):
            x_tmp1 = np.concatenate([x_mag[frmelen - ovlplen:, i - 1], x_mag[:, i]])
            x_tmp2 = scipy.signal.lfilter(filtb, [1], x_tmp1)
            x_magsm[:, i] = x_tmp2[-x_mag.shape[0]:]

        # Weighted spectral estimate 
        Wn2, Wn1, Wn0 = 0.09, 0.25, 0.66  # Sum = 1.0
        if nframes > 1:
            x_magsm[:, 1] = Wn1 * x_magsm[:, 0] +Wn0 * x_magsm[:, 1]
            for i in range(2, nframes):
                x_magsm[:, i] = (Wn2 * x_magsm[:, i - 2] + Wn1 * x_magsm[:, i - 1] + Wn0 * x_mag[:, i])
    else:
        x_magsm = x_mag   

    # Noise update during silence frames    
    if VAD:
        n_spect_expanded = np.tile(n_spect, (1, nframes))
        n_spect, state = noiseupdt(x_magsm, n_spect_expanded, cmmnlen, nframes, n_spect_actual)
    else:
        # Replicate noise spectrum for all frames (no VAD)   
        n_spect = np.repeat(n_spect, nframes, axis=1)
   
    # Calculate segmental SNR in each band
    SNR_x = np.zeros((Nband, nframes))
    for i in range(Nband):
        if i < Nband - 1:
            start = lobin[i]
            stop = hibin[i] + 1
        else:
            start = lobin[i]
            stop = fftl // 2 + 1 

        for j in range(nframes):
            signal_power = (np.linalg.norm(x_magsm[start:stop, j], 2) ** 2)/(frmelen * U)
            noise_power = (np.linalg.norm(n_spect[start:stop, j], 2) ** 2)/(frmelen * U)
            SNR_x[i, j] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    beta_x = berouti(SNR_x)
        
    # ---------- START SUBTRACTION PROCEDURE --------------------------
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
            if i < Nband-1:
                sub_speech = sub_speech + 0.05 * x_magsm[start:stop, j] ** 2
            else:
                sub_speech = sub_speech + 0.01 * x_magsm[start:stop, j] ** 2
            sub_speech_x[start:stop, j] += sub_speech

    # Reconstruct whole spectrum
    enhanced_mag = np.sqrt(np.maximum(sub_speech_x, 0))
    enhanced_spectrum = np.zeros((fftl, nframes), dtype=np.complex128)
    enhanced_spectrum[:fftl // 2 + 1, :] = enhanced_mag * np.exp(1j * x_ph[:fftl // 2 + 1, :])
    enhanced_spectrum[fftl // 2 + 1:, :] = np.conj(np.flipud(enhanced_spectrum[1:fftl // 2, :]))
    
    # IFFT to time domain
    y1_ifft = ifft(enhanced_spectrum, axis=0)
    y1_r = np.real(y1_ifft)

    # Weighted Overlap-Add (WOLA) synthesis
    out = np.zeros((nframes - 1) * cmmnlen + frmelen)
    win_sum = np.zeros_like(out)

    for i in range(nframes):
        start = i * cmmnlen
        out[start:start + frmelen] += y1_r[:frmelen, i]*win
        win_sum[start:start + frmelen] += win**2
   # Normalize by window energy
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]

    # Trim to original length
    out = out[:len(noisy_speech)]

    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(out))
    if max_amplitude > 1.0:
        out = out / max_amplitude
        print(f"Output normalized by factor {max_amplitude:.3f}")

    enhanced_tensor = torch.tensor(out, dtype=torch.float32)    # Convert to tensor

    # Save to file if output path provided
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with metadata
        metadata_parts = [
            f"BANDS{Nband}",
            f"SPACING{Freq_spacing.upper()}",
            f"FRAME{FRMSZ}ms"
        ]

        # Extract base name without extension
        base_name = output_file.replace('.wav', '') if output_file.endswith('.wav') else output_file
        output_filename = f"{base_name}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        torchaudio.save(full_output_path, enhanced_tensor.unsqueeze(0), fs)
        print(f"Enhanced audio saved to: {full_output_path}")

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

if __name__ == "__main__":

    TARGET_SR = 16000
    TARGET_SNR_DB = 5
    OUTPUT_DIR = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP2\\spectral\\NOISE_ESTIMATION"

    torch.manual_seed(42)
    
    clean_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\EARS_DATASET\\p092\\emo_adoration_freeform.wav")
    noise_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\NOIZEUS_NOISE_DATASET\\Noise Recordings\\cafeteria_babble.wav")

    print("Preparing audio data pair...")
    noisy_tensor, clean_tensor, noise_tensor, fs = prepare_audio_data(clean_path, noise_path, TARGET_SR, TARGET_SNR_DB)

    print(f"Data pair created at {TARGET_SNR_DB} dB SNR.")
    print("-" * 30)

    # Generate clean, noise, and noisy audio in output directory for reference
    torchaudio.save(Path(OUTPUT_DIR) / "clean_reference.wav", clean_tensor.unsqueeze(0), fs)
    torchaudio.save(Path(OUTPUT_DIR) / "noise_reference.wav", noise_tensor.unsqueeze(0), fs)
    torchaudio.save(Path(OUTPUT_DIR) / "noisy_input.wav", noisy_tensor.unsqueeze(0), fs)


    print("\nRunning in STANDARD MODE (estimating noise)...")

    mband(
        noisy_audio=noisy_tensor,
        fs=fs,
        output_dir=OUTPUT_DIR,
        output_file="enhanced_ESTIMATED_NOISE.wav",
        input_name="standard_mode",
        Nband=4,
        Freq_spacing='linear',
        FRMSZ=8,
        OVLP=50,
        AVRGING=1,
        Noisefr=1,
        FLOOR=0.002,
        VAD=1,
        actual_noise_audio=None
    )

    print("\nRunning in GROUND TRUTH MODE (using actual noise)...")
    mband(
        noisy_audio=noisy_tensor,
        fs=fs,
        output_dir=OUTPUT_DIR,
        output_file="enhanced_GROUND_TRUTH_NOISE.wav",
        input_name="ground_truth_mode",
        Nband=4,
        Freq_spacing='linear',
        FRMSZ=8,
        OVLP=50,
        AVRGING=1,
        Noisefr=1,
        FLOOR=0.002,
        VAD=1,
        actual_noise_audio=noise_tensor
    )

    print("Processing complete.")
    # Example usage
    # noisy_audio = torchaudio.load("C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\final_audio_check\\03_noisy_input_to_mband.wav")[0]
    # mband(noisy_audio, fs=16000, output_dir="C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\validation_dataset\\enhanced_speech", output_file="enhanced_sp21_station_sn5.wav", input_name="sp21_station_sn5", Nband=4, Freq_spacing='linear', FRMSZ=8, OVLP=50, AVRGING=1, Noisefr=1, FLOOR=0.002, VAD=1, n_spect_actual=None)
