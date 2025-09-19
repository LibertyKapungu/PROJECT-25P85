import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import scipy.signal

def berouti(SNR):
    """Berouti's algorithm for computing over-subtraction factor"""
    nbands, nframes = SNR.shape
    a = np.zeros((nbands, nframes))
    for i in range(nbands):
        for j in range(nframes):
            if SNR[i, j] >= -5.0 and SNR[i, j] <= 20:
                a[i, j] = 4 - SNR[i, j] * 3 / 20
            elif SNR[i, j] < -5.0:
                a[i, j] = 4.75
            else:
                a[i, j] = 1
    return a

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

# If Freq_spacing is 'mel', the code uses the mel 
# function to compute the band edges according to the Mel scale,
# which is designed to approximate the human ear's perception of pitch.

def mel(N, low, high):
    """Mel scale frequency mapping"""
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

def frame(sdata, window, frmshift, offset=0, trunc=0):
    """Frame signal into overlapping windows"""
    sdata = np.array(sdata).flatten()
    ndata = len(sdata)
    window = np.array(window).flatten()
    nwind = len(window)
    
    if frmshift <= 0:
        raise ValueError('frame: shift must be positive')
    
    # Apply offset
    if offset > 0:
        sdata = sdata[offset:max(offset, ndata)]
        ndata = len(sdata)
    elif offset < 0:
        sdata = np.concatenate([np.zeros(abs(offset)), sdata])
        ndata = len(sdata)
    
    # Determine number of frames
    if trunc:
        nframes = int(np.floor((ndata - nwind) / frmshift + 1))
    else:
        nframes = int(np.ceil(ndata / frmshift))
    
    # Frame the data
    tdata = np.zeros((nwind, nframes))
    ixstrt = 0
    
    for frm in range(nframes):
        ixend = min(ndata, ixstrt + nwind)
        ixlen = ixend - ixstrt
        tdata[:ixlen, frm] = sdata[ixstrt:ixstrt + ixlen]
        ixstrt += frmshift
    
    # Apply window
    fdata = tdata * window.reshape(-1, 1)
    return fdata

def noiseupdt(x_magsm, n_spect, cmmnlen, nframes):
    """Voice Activity Detection and noise spectrum update"""
    SPEECH = 1
    SILENCE = 0
    
    # Initialize arrays
    state = np.zeros(nframes * cmmnlen, dtype=int)
    judgevalue1 = np.zeros(nframes * cmmnlen)
    
    # Process first frame
    i = 0
    x_var = x_magsm[:, i] ** 2
    n_var = n_spect[:, i] ** 2
    rti = x_var / n_var - np.log10(x_var / n_var) - 1
    judgevalue = np.mean(rti)
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

def mband(filename, outfile, Nband, Freq_spacing):
    """
    Implements the multi-band spectral subtraction algorithm [1]. 
    Usage:  mband(filename, outputfile,Nband,Freq_spacing)
           
         filename - noisy speech file in .wav format
         outputFile - enhanced output file in .wav format
         Nband - Number of frequency bands (recommended 4-8)
         Freq_spacing - Type of frequency spacing for the bands, choices:
                        'linear', 'log' and 'mel'

    Example call:  mband('sp04_babble_sn10.wav','out_mband.wav',6,'linear');
    -----------------------------------------------
    """   

    if filename is None or outfile is None:
        print('Usage: mband(noisyfile.wav, outFile.wav,  Nband, Freq_spacing)\n')
        return

    fs, data = wavfile.read(filename)

    # Convert to double precision normalized samples (MATLAB wavread default behavior (see waveread docs  for details))
    # Based on MATLAB wavread documentation scaling rules
    if data.dtype == np.uint8: # 8-bit: 0 <= y <= 255 -> convert to -1.0 <= y < +1.0
        noisy_speech = (data.astype(np.float64) - 128.0) / 128.0
        nbits = 8

    elif data.dtype == np.int16: # 16-bit: -32768 <= y <= +32767 -> convert to -1.0 <= y < +1.0
        noisy_speech = data.astype(np.float64) / 32768.0
        nbits = 16

    elif data.dtype == np.int32:
        max_val = np.max(np.abs(data))

        if max_val <= 2**23: # 24-bit: -2^23 <= y <= 2^23-1 -> convert to -1.0 <= y < +1.0
            noisy_speech = data.astype(np.float64) / (2**23)
            nbits = 24

        else: # 32-bit: -2^31 <= y <= 2^31-1 -> convert to -1.0 <= y < +1.0
            noisy_speech = data.astype(np.float64) / (2**31)
            nbits = 32

    elif data.dtype == np.float32:
        noisy_speech = data.astype(np.float64) # 32-bit float: already in -1.0 <= y < +1.0 range
        nbits = 32

    elif data.dtype == np.float64: # 64-bit float: already in proper range
        noisy_speech = data
        nbits = 64

    else:
        # Default fallback - treat unknown types as 16-bit
        noisy_speech = data.astype(np.float64) / 32768.0
        nbits = 16

    # Parameters
    AVRGING = 1
    FRMSZ = 20
    OVLP = 50
    Noisefr = 6
    FLOOR = 0.002
    VAD = 1

    # AVRGING -> Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
    # FRMSZ -> Frame length in milli-seconds, default=20
    # OVLP -> Window overlap in percent of frame size, default=50
    # Noisefr -> Number of noise frames at beginning of file for noise spectrum estimate, default=6
    # FLOOR -> Spectral floor, default=0.002
    # VAD -> Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise

    frmelen = int(np.floor(FRMSZ * fs / 1000))  # Frame size in samples
    ovlplen = int(np.floor(frmelen * OVLP / 100)) # Number of overlap samples
    cmmnlen = frmelen - ovlplen  # Number of common samples between adjacent frames
    
    # Determine FFT length 
    fftl = 2  
    while fftl < frmelen:
        fftl *= 2       # set to the smallest power of two greater than or equal to the frame length in sample

    # Band setup

    # **REAL-TIME ISSUE #1: Band setup done offline**

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
        #  The resulting frequency edges are again converted to FFT bin indices. 
        # The first lower bin is set to zero and the last upper bin is set to half 
        # the FFT length to ensure the bands cover the entire spectrum. The band 
        # sizes are computed similarly to the logarithmic case.
    else:
        raise ValueError('Error in selecting frequency spacing')

    # Calculate Hamming window
    win = np.sqrt(hamming(frmelen))

    # **REAL-TIME ISSUE #2: Initial noise estimation requires buffering**
    # Estimate noise magnitude for first 'Noisefr' frames

    # Estimate noise magnitude for first Noisefr frames
    noise_pow = np.zeros(fftl)
    j = 0
    for k in range(Noisefr):
        if j + frmelen <= len(noisy_speech):
            frame_data = noisy_speech[j:j + frmelen]
        else:
            frame_data = np.pad(noisy_speech[j:], (0,j+frmelen - len(noisy_speech)))
        
        n_fft = fft(frame_data * win, fftl)
        n_mag = np.abs(n_fft)
        n_magsq = n_mag ** 2
        noise_pow += n_magsq
        j += frmelen

    n_spect = np.sqrt(noise_pow / Noisefr).reshape(-1, 1)

    # **REAL-TIME ISSUE #3: Batch framing of entire signal**
    # Frame the input signal
    # Framing
    framed_x = frame(noisy_speech, win, ovlplen, 0, 0)
    #nframes = framed_x.shape[1]
    tmp, nframes = framed_x.shape

    # **REAL-TIME ISSUE #4: Batch FFT processing**
    # FFT processing
    # FFT
    x_fft = fft(framed_x, fftl, axis=0)
    x_mag = np.abs(x_fft)
    x_ph = np.angle(x_fft)

    # Smoothing
    if AVRGING:
        filtb = [0.9, 0.1]
        x_magsm = np.zeros_like(x_mag)
        # First frame
        x_magsm[:, 0] = scipy.signal.lfilter(filtb, [1], x_mag[:, 0])
        # Remaining frames
        for i in range(1, nframes):
            # Concatenate tail of previous frame and current frame
            x_tmp1 = np.concatenate([x_mag[frmelen - ovlplen:, i - 1], x_mag[:, i]])
            x_tmp2 = scipy.signal.lfilter(filtb, [1], x_tmp1)
            x_magsm[:, i] = x_tmp2[1:1 + len(x_mag[:,i])]

        # Weighted spectral estimate (temporal smoothing across frames)
        Wn2, Wn1, W0 = 0.09, 0.25, 0.32
        total_causal = Wn2 + Wn1 + W0
           # Frame 2: special boundary case
        if nframes > 1:
            temp_total_2 = Wn1 + W0
            x_magsm[:, 1] = (Wn1/temp_total_2 * x_magsm[:, 0] + 
                            W0/temp_total_2 * x_magsm[:, 1])
        
        # Frames 3 onwards: full 3-tap causal filter
        for i in range(2, nframes):
            x_magsm[:, i] = (Wn2/total_causal * x_magsm[:, i-2] + 
                            Wn1/total_causal * x_magsm[:, i-1] + 
                            W0/total_causal * x_magsm[:, i])
    else:
        x_magsm = x_mag

    # Noise update during silence frames
    if VAD:
        n_spect_full = np.tile(n_spect, (1, nframes))
        n_spect, state = noiseupdt(x_magsm, n_spect_full, cmmnlen, nframes)
    else:
        n_spect = np.tile(n_spect, (1, nframes))

    # Segmental SNR in each band
        
    # Calculate segmental SNR in each band
    SNR_x = np.zeros((Nband, nframes))
    for i in range(Nband-1):
        start = lobin[i]
        stop = hibin[i] + 1
        for j in range(nframes):
            signal_power = np.linalg.norm(x_magsm[start:stop, j], 2)**2
            noise_power = np.linalg.norm(n_spect[start:stop, j], 2)**2
            SNR_x[i, j] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Last band (special handling)
    start = lobin[Nband-1]
    stop = fftl//2 + 1
    for j in range(nframes):
        signal_power = np.linalg.norm(x_magsm[start:stop, j], 2)**2
        noise_power = np.linalg.norm(n_spect[start:stop, j], 2)**2
        SNR_x[Nband-1, j] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Compute over-subtraction factors
    beta_x = berouti(SNR_x)
        
        # Subtraction
    sub_speech_x = np.zeros((fftl // 2 + 1, nframes))
    k = 1
    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1
        
        for j in range(nframes):
            n_spec_sq = n_spect[start:stop, j] ** 2
            if i == 0:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq
            elif i == Nband - 1:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * 1.5
            else:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * 2.5
            z = np.where(sub_speech < 0)[0]
            if z.size > 0:
                sub_speech[z] = FLOOR * x_magsm[start:stop, j][z] ** 2
            if i == 0:
                sub_speech = sub_speech + 0.05 * x_magsm[start:stop, j] ** 2
            elif i == Nband - 1:
                sub_speech = sub_speech + 0.01 * x_magsm[start:stop, j] ** 2
            sub_speech_x[start:stop, j] += sub_speech

    # Reconstruct spectrum with Hermitian symmetry
    enhanced_mag = np.sqrt(np.maximum(sub_speech_x, 0))
    enhanced_spectrum = np.zeros((fftl, nframes), dtype=np.complex128)
    enhanced_spectrum[:fftl // 2 + 1, :] = enhanced_mag * np.exp(1j * x_ph[:fftl // 2 + 1, :])
    enhanced_spectrum[fftl // 2 + 1:, :] = np.conj(np.flipud(enhanced_spectrum[1:fftl // 2, :]))
    y1_ifft = ifft(enhanced_spectrum, axis=0)
    y1_r = np.real(y1_ifft)

    # Overlap-add (standard)
    out = np.zeros((nframes - 1) * cmmnlen + frmelen)
    win_sum = np.zeros_like(out)
    for i in range(nframes):
        start = i * cmmnlen
        out[start:start + frmelen] += y1_r[:frmelen, i]
        win_sum[start:start + frmelen] += win
    out /= (win_sum + 1e-8)

    # Output normalization and type conversion
    out = out[:len(noisy_speech)]
    if nbits == 8:
        enhanced_speech_int = np.clip(out * 128.0 + 128.0, 0, 255).astype(np.uint8)
    elif nbits == 16:
        enhanced_speech_int = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
    elif nbits == 32:
        if data.dtype == np.float32:
            enhanced_speech_int = np.clip(out, -1.0, 1.0).astype(np.float32)
        else:
            max_val = np.iinfo(np.int32).max
            enhanced_speech_int = np.clip(out * max_val, -max_val, max_val - 1).astype(np.int32)
    else:
        enhanced_speech_int = np.clip(out, -1.0, 1.0)

    wavfile.write(outfile, fs, enhanced_speech_int)

# Example usage:
mband('C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS/noisy_speech/sp21_station_sn0.wav', 'C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS/noisy_speech/out_mband.wav', 6, 'linear')


