import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import scipy.signal
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import torchaudio

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
    """Mel scale frequency mapping
    
       This function returns the lower, center and upper freqs
       of the filters equally spaced in mel-scale
       Input: N - number of filters
 	   low - (left-edge) 3dB frequency of the first filter
	   high - (right-edge) 3dB frequency of the last filter

       # The mel scale is designed to approximate the human ear's perception of pitch.

       Copyright (c) 1996-97 by Philipos C. Loizou

    """
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

def mband(
        noisy_audio: torch.Tensor,
        fs: int,
        output_dir: Optional[Union[str, Path]] = None,
        output_file: Optional[str] = None,
        input_name: Optional[str] = None,
        Nband: int = 6,
        Freq_spacing: str = 'log',
        FRMSZ: int = 20, 
        OVLP: int = 50, 
        AVRGING: int = 1,
        Noisefr: int = 6,
        FLOOR: float = 0.002,
        VAD: int = 1
) -> Tuple[torch.Tensor, int]:
    """
    Implements the multi-band spectral subtraction algorithm [1]. 
    Usage:  mband(noisy_audio, outputfile,Nband,Freq_spacing)
           
         noisy_audio - noisy speech file in .wav format
         outputFile - enhanced output file in .wav format
         Nband - Number of frequency bands (recommended 4-8)
         Freq_spacing - Type of frequency spacing for the bands, choices:
                        'linear', 'log' and 'mel'
         AVRGING - Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
         FRMSZ - Frame length in milli-seconds, default=20. hop size of 16ms * (1 - 0.50) = 8ms 
         OVLP - Window overlap in percent of frame size, default=50
         Noisefr - Number of noise frames at beginning of file for noise spectrum estimate, default=6 .Matlab recommends 6 but doing 1 so less latency  
         FLOOR - Spectral floor, default=0.002
         VAD - Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise


    Example call:  mband('sp04_babble_sn10.wav','out_mband.wav',6,'linear');

    References:
    [1] Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
        method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
        Conf. Acoust.,Speech, Signal Processing
    
    Authors: Sunil Kamath and Philipos C. Loizou

    Copyright (c) 2006 by Philipos C. Loizou
    $Revision: 0.0 $  $Date: 10/09/2006 $

    -----------------------------------------------
    """   

    # Handle tensor input
    if noisy_audio.dim() > 1 and noisy_audio.shape[0] > 1:
        noisy_speech = torch.mean(noisy_audio, dim=0).numpy()
    else:
        noisy_speech = noisy_audio.squeeze().numpy()

    # Convert to double precision
    noisy_speech = noisy_speech.astype(np.float64)
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
        #  The resulting frequency edges are again converted to FFT bin indices. The first lower bin is set to zero and the last upper bin is set to half 
        # the FFT length to ensure the bands cover the entire spectrum. The band sizes are computed similarly to the logarithmic case.
    else:
        raise ValueError('Error in selecting frequency spacing')

    # Calculate Hanning window (better for WOLA at 50% overlap)
    win = np.sqrt(np.hanning(frmelen))  # np.sqrt() keeps it power-complementary

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


    # ============ STREAMING INITIALIZATION ============
    # State variables for causal processing
    prev_mag = None
    prev_prev_smoothed_mag = None
    prev_smoothed_mag = None
    n_current = n_spect.squeeze()  # Current noise estimate
    
    # Circular output buffers
    out_buffer = np.zeros(frmelen)
    win_sum_buffer = np.zeros(frmelen)
    output_samples = []  # Collect output chunks
    
    # IIR filter coefficients
    filtb = np.array([0.9, 0.1])
    
    # Frame counter
    frame_idx = 0
    
    # Process audio frame by frame with overlap
    sample_pos = 0
    while sample_pos + frmelen <= len(noisy_speech):
    # Extract and FFT current frame
        current_frame = noisy_speech[sample_pos:sample_pos + frmelen]
        windowed_frame = current_frame * win
        frame_fft = fft(windowed_frame, fftl)
        frame_mag = np.abs(frame_fft)
        frame_phase = np.angle(frame_fft)
         
        sample_pos += cmmnlen  # Advance by hop size (cmmnlen)  

        #  IIR SMOOTHING (CAUSAL)
        if AVRGING: 
            if frame_idx == 0:
                frame_smoothed = scipy.signal.lfilter(filtb, [1], frame_mag)
            else: 
                overlap_region = prev_mag[frmelen-ovlplen:]
                x_tmp = np.concatenate([overlap_region,frame_mag])
                x_tmp_filtered = scipy.signal.lfilter(filtb, [1], x_tmp)
                frame_smoothed = x_tmp_filtered[-len(frame_mag):]
        else:
            frame_smoothed = frame_mag.copy()  

        # VAD and Noise Update 
        if VAD:               
            x_var = frame_smoothed**2
            n_var = n_current**2
            rti = x_var/(n_var + 1e-10) - np.log10(x_var/(n_var+1e-10))-1
            judgevalue = np.mean(rti)

            threshold = 0.4 if frame_idx == 0 else 0.45

            if judgevalue > threshold:  # SPEECH
                n_updated = n_current.copy()
            else: # SILENCE
                n_updated = np.sqrt(0.9*n_current**2 + 0.1*frame_smoothed**2)
            n_current=n_updated
        else:
            n_updated = n_current.copy()

        # WEIGHTED AVERAGING (CAUSAL)
        if AVRGING and frame_idx >= 2:
            Wn2, Wn1, Wn0 = 0.09, 0.25, 0.66
            frame_final = Wn2 * prev_prev_smoothed_mag + Wn1 * prev_smoothed_mag + Wn0 * frame_smoothed
        elif AVRGING and frame_idx == 1:
            frame_final = 0.30 * prev_smoothed_mag + 0.70 * frame_smoothed # 2 tap filter
        else: 
            frame_final = frame_smoothed.copy()

        # SPECTRAL SUBTRACTION (PER-BAND)
        enhanced_mag_frame = np.zeros(fftl // 2+1)

        for band_idx in range(Nband):
            start = lobin[band_idx]
            stop = hibin[band_idx] + 1 if band_idx < Nband-1 else fftl // 2+1

            signal_power = np.linalg.norm(frame_final[start:stop], 2) ** 2
            noise_power = np.linalg.norm(n_updated[start:stop], 2) ** 2
            snr_band = 10 * np.log10(signal_power / (noise_power + 1e-10))

            # Beta (over-subtraction factor)
            if -5.0 <= snr_band <= 20.0:
                beta = 4.0 - snr_band * 3.0 / 20.0
            elif snr_band < -5.0:
                beta = 4.75
            else:
                beta = 1.0

            # Delta (frequency-dependent)
            if band_idx == 0:
                delta = 1.0
            elif band_idx == Nband - 1:
                delta = 1.5
            else:
                delta = 2.5
        
            sub_speech = frame_final[start:stop] ** 2 - beta* n_updated[start:stop]**2 * delta
            sub_speech = np.maximum(sub_speech, FLOOR*frame_final[start:stop]**2)

            # Residual 
            if band_idx < Nband-1:
                sub_speech += 0.05 * frame_final[start:stop] ** 2
            else:
                sub_speech += 0.01 * frame_final[start:stop] ** 2
            
            enhanced_mag_frame[start:stop] = np.sqrt(sub_speech)
           

        # ===== RECONSTRUCT SPECTRUM & IFFT =====
        enhanced_spectrum_frame = np.zeros(fftl, dtype=np.complex128)
        enhanced_spectrum_frame[:fftl // 2 + 1] = enhanced_mag_frame * np.exp(1j * frame_phase[:fftl // 2 + 1])
        enhanced_spectrum_frame[fftl // 2 + 1:] = np.conj(np.flipud(enhanced_spectrum_frame[1:fftl // 2]))
        
        y_frame = ifft(enhanced_spectrum_frame).real

        # After spectral subtraction, check for musical noise
    # if frame_idx % 50 == 0:
    #     spectral_variance = np.var(enhanced_mag_frame)
    #     print(f"Frame {frame_idx}: Spectral variance = {spectral_variance:.4f}")
    #     # High variance = musical noise present

        # ===== WOLA SYNTHESIS (CIRCULAR BUFFER) =====

        out_buffer += y_frame[:frmelen]*win
        win_sum_buffer += win**2

        # Output first 'cmmnlen' samples (ready for playback)
        output_chunk = out_buffer[:cmmnlen]/(win_sum_buffer[:cmmnlen]+1e-8)
        output_samples.extend(output_chunk)

        # Shift circular buffer
        out_buffer = np.roll(out_buffer,-cmmnlen)
        out_buffer[-cmmnlen:] = 0 # Zero out entire tail region

        win_sum_buffer = np.roll(win_sum_buffer, -cmmnlen) 
        win_sum_buffer[-cmmnlen:] = 0

        #  ===== UPDATE STATE FOR NEXT FRAME =====
        prev_mag = frame_mag.copy()
        prev_prev_smoothed_mag = prev_smoothed_mag
        prev_smoothed_mag = frame_smoothed.copy()
        frame_idx += 1

    # ============ FLUSH REMAINING SAMPLES ============
    if np.any(win_sum_buffer > 1e-8):
        final_chunk = out_buffer / (win_sum_buffer + 1e-8)
        valid_idx = np.where(win_sum_buffer > 1e-8)[0]
        if len(valid_idx) > 0:
            output_samples.extend(final_chunk[:valid_idx[-1] + 1])
    
    out = np.array(output_samples)

    # ===== VERIFY WOLA NORMALIZATION (DEBUG) =====
    # Uncomment to check if windows are power-complementary
    U = np.sum(win**2) / frmelen
    expected_energy = np.sum(noisy_speech**2)
    actual_energy = np.sum(out**2)
    print(f"Energy ratio: {actual_energy / expected_energy:.4f} (should be ~1.0)")

    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(out))
    if max_amplitude > 1.0:
        out = out / max_amplitude
        print(f"Output normalized by factor {max_amplitude:.3f}")

    # Convert to tensor
    enhanced_tensor = torch.tensor(out, dtype=torch.float32)

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

        # Save as 16-bit WAV
        # enhanced_speech_int = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        # wavfile.write(str(full_output_path), fs, enhanced_speech_int)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced_tensor, fs

if __name__ == "__main__":

    TARGET_SR = 16000
    TARGET_SNR_DB = 5
    OUTPUT_DIR = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP2\\spectral\\NOISE_ESTIMATION"

    torch.manual_seed(42)

    clean_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\EARS_DATASET\\p092\\emo_adoration_freeform.wav")
    #noise_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\sound_data\\raw\\NOIZEUS_NOISE_DATASET\\Noise Recordings\\cafeteria_babble.wav")
    #noise_path = Path(r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn_model\\test_wavs\\noisy_input.wav")
    noise_path = enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\noisy_speech\\sp21_station_sn5.wav'

    noisy_tensor, noisy_fs = torchaudio.load(noise_path)

    # print("Preparing audio data pair...")
    #noisy_tensor, clean_tensor, noise_tensor, fs = prepare_audio_data(clean_path, noise_path, TARGET_SR, TARGET_SNR_DB)

    print(f"Data pair created at {TARGET_SNR_DB} dB SNR.")
    print("-" * 30)

    mband(
        noisy_audio=noisy_tensor,
        fs=noisy_fs,
        output_dir=OUTPUT_DIR,
        output_file="mband_causal.wav",
        input_name="standard_mode",
        Nband=6,
        Freq_spacing='linear',
        FRMSZ=20,
        OVLP=50,
        AVRGING=1,
        Noisefr=6,
        FLOOR=0.002,
        VAD=1,
    )