import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import scipy.signal

def berouti(SNR):
    """Berouti's algorithm for computing over-subtraction factor"""
    # Should probs avoid the for loop 
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

def frame(sdata, window, frmshift, offset=0, trunc=0):
    """Frame signal into overlapping windows
    
    This function places sampled data in vector sdata into a matrix of
    frame data.  The input sampled data sdata must be a vector.  The
    window is a windowing vector (eg, hamming) applied to each frame of
    sampled data and must be specified, because it defines the length
    of each frame in samples.  The optional frmshift parameter
    specifies the number of samples to shift between frames, and if not
    specified defaults to the window size (which implies no overlap).
    The optional offset specifies the offset from the first sample to
    be used for processing.  If not specified, it is set to 0, which
    means that the first sample of the sdata is the first sample of the
    frame data.  The value of offset can be negative, in which case
    initial padding of 0 samples is done.  The optional argument trunc
    is a flag that specifies that sample data at the end should be 
    truncated so that the last frame contains only valid data from the
    samples and no zero padding is done at the end of the sample data
    to fill a frame.  This means some sample data at the end will be
    lost.  The default is not to truncate, but to pad with zero
    samples until all sample data is represented in a frame at the end.
    
    """

    if sdata.ndim != 1:
        raise ValueError("frame: sdata must be a 1D vector")
    if window.ndim != 1:
        raise ValueError("frame: window must be a 1D vector")

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
    dowind = not np.all(window == 1)
    ixstrt = 0
    fstart = []

    for frm in range(nframes):
        ixend = min(ndata, ixstrt + nwind)
        ixlen = ixend - ixstrt
        tdata[:ixlen, frm] = sdata[ixstrt:ixstrt + ixlen]
        fstart.append(ixstrt)
        ixstrt += frmshift

    if offset != 0:
        fstart = [i + offset for i in fstart]
    
    # Apply window
    if dowind:
        fdata = tdata * window.reshape(-1, 1)
    else:
        fdata = tdata

    return fdata, fstart

# Not part of matlab code
def calculate_delta_factors(lobin, hibin, fs, Nband, fftl):
    """Calculate frequency-dependent delta factors from Loizou eq 5.62"""
    delta_factors = np.zeros(Nband)
    
    for i in range(Nband):
        # Convert bin index to frequency
        upper_freq_hz = hibin[i] * fs / (2 * fftl)  # Nyquist scaling
        
        # Apply Loizou's rules
        if upper_freq_hz <= 1000:  # f <= 1 kHz
            delta_factors[i] = 1.0
        elif 1000 < upper_freq_hz <= (fs/2 - 1000):  # 1 kHz < f <= Fs/2 - 1 kHz  
            delta_factors[i] = 2.5
        else:  # f > Fs/2 - 1 kHz
            delta_factors[i] = 1.5
            
    return delta_factors

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

    References:
    [1] Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
        method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
        Conf. Acoust.,Speech, Signal Processing
    
    Authors: Sunil Kamath and Philipos C. Loizou

    Copyright (c) 2006 by Philipos C. Loizou
    $Revision: 0.0 $  $Date: 10/09/2006 $

    -----------------------------------------------
    """   

    # Parameters
    AVRGING = 0
    FRMSZ = 20
    OVLP = 50
    Noisefr = 1  # Matlab recommends 6 but doing 1 so less latency  
    FLOOR = 0.002
    VAD = 1

    # AVRGING -> Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
    # FRMSZ -> Frame length in milli-seconds, default=20
    # OVLP -> Window overlap in percent of frame size, default=50
    # Noisefr -> Number of noise frames at beginning of file for noise spectrum estimate, default=6
    # FLOOR -> Spectral floor, default=0.002
    # VAD -> Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise

    if filename is None or outfile is None:
        print('Usage: mband(noisyfile.wav, outFile.wav,  Nband, Freq_spacing)\n')
        return

    fs, data = wavfile.read(filename)

    # Convert to double precision normalized samples
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
        #  The resulting frequency edges are again converted to FFT bin indices. The first lower bin is set to zero and the last upper bin is set to half 
        # the FFT length to ensure the bands cover the entire spectrum. The band sizes are computed similarly to the logarithmic case.
    else:
        raise ValueError('Error in selecting frequency spacing')

    # Calculate Hamming window
    win = np.sqrt(hamming(frmelen))

    # **REAL-TIME ISSUE #2: Initial noise estimation requires buffering**  (can fix with noisefr = 1?)
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
        # Step 1: Extract current frame
        current_frame = noisy_speech[sample_pos:sample_pos + frmelen]
        
        # Step 2: Apply window function
        windowed_frame = current_frame * win
        
        # Step 3: Single-frame FFT (instead of batch FFT)
        frame_fft = fft(windowed_frame, fftl)
        frame_mag = np.abs(frame_fft)
        frame_ph = np.angle(frame_fft)
        
        # Step 4: Store results
        x_mag_frames.append(frame_mag)
        x_ph_frames.append(frame_ph)
        
        # Step 5: Advance by hop size (cmmnlen) for proper overlap
        sample_pos += cmmnlen  # NOT frmelen - this creates overlap
        frame_count += 1
        
    # Convert lists to matrices (same format as original batch method)
    if x_mag_frames:
        x_mag = np.array(x_mag_frames).T  # Shape: (fftl, nframes)
        x_ph = np.array(x_ph_frames).T    # Shape: (fftl, nframes)
        nframes = len(x_mag_frames)
    else:
        # Handle edge case of very short audio
        x_mag = np.array([]).reshape(fftl, 0)
        x_ph = np.array([]).reshape(fftl, 0)
        nframes = 0
        print("Warning: No frames generated - audio too short")

    # ==========Start Processing =======

    if AVRGING:
            # Smooth the input spectrum
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
        # Expand n_spect to match number of frames BEFORE calling noiseupdt
        n_spect_expanded = np.tile(n_spect, (1, nframes))
        n_spect, state = noiseupdt(x_magsm, n_spect_expanded, cmmnlen, nframes)
    else:
        # Replicate noise spectrum for all frames (no VAD)   
        n_spect = np.repeat(n_spect, nframes, axis=1)
   
    # Calculate segmental SNR in each band
    # Good to parallelize here as they are independent of each other
    
    SNR_x = np.zeros((Nband, nframes))

    for i in range(Nband):
        if i < Nband - 1:
            start = lobin[i]
            stop = hibin[i] + 1
        else:
            start = lobin[i]
            stop = fftl // 2 + 1  # Nyquist bin

        for j in range(nframes):
            signal_power = np.linalg.norm(x_magsm[start:stop, j], 2) ** 2
            noise_power = np.linalg.norm(n_spect[start:stop, j], 2) ** 2
            SNR_x[i, j] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    beta_x = berouti(SNR_x)
        
    # ---------- START SUBTRACTION PROCEDURE --------------------------
    sub_speech_x = np.zeros((fftl // 2 + 1, nframes))

  
    #delta_factors = calculate_delta_factors(lobin, hibin, fs, Nband, fftl) 
   

    for i in range(Nband):
        start = lobin[i]
        stop = hibin[i] + 1
        
        for j in range(nframes):
            n_spec_sq = n_spect[start:stop, j] ** 2
            #sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * delta_factors[i]
            if i == 0:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq
            elif i == Nband - 1:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * 1.5
            else:
                sub_speech = x_magsm[start:stop, j] ** 2 - beta_x[i, j] * n_spec_sq * 2.5
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
#mband('C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS/noisy_speech/sp21_station_sn5.wav', 'C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS/noisy_speech/out_mband.wav', 4, 'log')

