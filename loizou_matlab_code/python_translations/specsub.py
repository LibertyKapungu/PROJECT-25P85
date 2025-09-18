import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hann
from scipy.fft import fft, ifft

def nextpow2(n):
    return int(np.ceil(np.log2(n)))

def berouti(SNR):
    if SNR >= -5.0 and SNR <= 20:
        return 4-SNR*3/20
    elif SNR < -5.0:
        return 5
    else:  # SNR > 20
        return 1

def berouti1(SNR):
    if SNR >= -5.0 and SNR <= 20:
        return 3 - SNR * 2 / 20
    elif SNR < -5.0:
        return 4
    else:  # SNR > 20
        return 1

def specsub(filename: str, outfile: str):
    """
    Implements the basic power spectral subtraction algorithm [1].

    Usage:  specsub(noisyFile, outputFile)

            filename - noisy speech file in .wav format
            outputFile - enhanced output file in .wav format

    Example call:  specsub('sp04_babble_sn10.wav', 'out_specsub.wav')
    --------------------------------------------------------------------------------
    """

    if filename is None or outfile is None:
        print('Usage: specsub(noisyfile.wav, outFile.wav)\n')
        return

    # Read the noisy speech file
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

    # --- Initialize variables ---
    frame_dur = 20  # Frame duration in milliseconds
    len_ = int(np.floor(frame_dur * fs / 1000))  # Frame length in samples
    if len_ % 2 != 0: # Ensure frame length is even
        len_ += 1
    
    PERC = 50 # window overlap in percent of frame size
    len1 = int(np.floor(len_ * PERC / 100)) # Number of overlapping samples
    len2 = len_ - len1 # Number of new samples per frame

    Thres = 3 # VAD threshold in dB SNRseg 
    alpha = 2.0 # power exponent
    FLOOR = 0.002
    G = 0.9

    win = hann(len_) # Hanning window
    winGain = len2 / np.sum(win) # normalization gain for overlap+add with 50% overlap

    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * 2 ** nextpow2(len_) # nFFT is the FFT length (next power of 2 from frame length)

    # --- Initial Noise Power Spectrum Estimation ---
    # Estimate the average noise power spectrum from the first few frames.
    noise_mean = np.zeros(nFFT)
    j = 0
    for k in range(5):
        noise_frame = noisy_speech[j:j+len_]
        if len(noise_frame) < len_:
            noise_frame = np.pad(noise_frame, (0, len_ - len(noise_frame)), 'constant') 
        noise_mean += np.abs(fft(win * noise_frame, nFFT))
        j += len_
    noise_mu = noise_mean / 5

    # --- Main Processing Loop ---
    # Process the entire signal frame by frame
    # it is pre-allocating memory for the output signal (xfinal) and initializing 
    # the necessary buffers (x_old) and counters (k) before starting the computationally
    #  intensive main processing loop.
    k = 0
    x_old = np.zeros(len1) # Buffer for overlap-add
    Nframes = int(np.floor(len(noisy_speech) / len2)) - 1
    xfinal = np.zeros(Nframes * len2)

    # Also use welch method for fiar comparison???

    # ============= Start Processing ============

    for n in range(Nframes):
        frame = noisy_speech[k:k+len_]  
        if len(frame) < len_:
            frame = np.pad(frame, (0, len_ - len(frame)), 'constant')
        insign = win * frame   # Windowing
        spec = fft(insign, nFFT)  #compute fourier transform of a frame
        sig = np.abs(spec) # compute the magnitude

        #save the noisy phase information 
        theta = np.angle(spec)

        SNRseg = 10*np.log10(np.linalg.norm(sig, 2)**2 / np.linalg.norm(noise_mu, 2)**2 + 1e-8) #1e-8 prevents log(0) error

        if alpha == 1.0:
            beta = berouti1(SNRseg)
        else:
            beta = berouti(SNRseg)

        sub_speech = sig ** alpha - beta * noise_mu ** alpha
        diffw = sub_speech - FLOOR * noise_mu ** alpha

        # Floor negative components
        z = np.where(diffw < 0)[0]
        if z.size > 0:
            sub_speech[z] = FLOOR * noise_mu[z] ** alpha

        # --- implement a simple VAD detector --------------
        if SNRseg < Thres:      #Update noise spectrum
            noise_temp = G*noise_mu**alpha + (1-G)*sig**alpha
            noise_mu = noise_temp ** (1 / alpha) # new noise spectrum

        # Conjugate symmetry for real IFFT
        sub_speech[nFFT//2+1:nFFT] = np.flipud(sub_speech[1:nFFT//2]) #to ensure conjugate symmetry for real reconstruction

        x_phase = (sub_speech ** (1 / alpha)) * np.exp(1j * theta)

        # take the IFFT 
        xi = np.real(ifft(x_phase, nFFT))

        # --- Overlap and add ---------------
        xfinal[k:k+len2] = x_old + xi[:len1]
        x_old = xi[len1:len_]

        k += len2

    # Output normalization
    xfinal = winGain * xfinal

    # --- De-normalization to original bit depth ---
    # Convert the processed float signal back to its original integer format for saving.
    if nbits == 8:
        enhanced_speech_int = np.clip(xfinal * 128.0 + 128.0, 0, 255).astype(np.uint8)
    
    elif nbits == 16:
        enhanced_speech_int = np.clip(xfinal * 32768.0, -32768, 32767).astype(np.int16)
    
    elif nbits == 24: # Convert back to -2^23 to 2^23-1 range, stored as int32
        enhanced_speech_int = np.clip(xfinal * (2**23), -(2**23), 2**23 - 1).astype(np.int32)
    
    elif nbits == 32:
        if data.dtype == np.float32: # Keep as float32 for 32-bit float data
            enhanced_speech_int = np.clip(xfinal, -1.0, 1.0).astype(np.float32)
        
        else: # Convert back to -2^31 to 2^31-1 range for int32
            enhanced_speech_int = np.clip(xfinal * (2**31), -(2**31), 2**31 - 1).astype(np.int32)
    
    elif nbits == 64: # Keep as float64
        enhanced_speech_int = np.clip(xfinal, -1.0, 1.0)
    
    else: # Raise error for unsupported bit depths
        raise ValueError(f"Unsupported bit depth: {nbits}. Supported values are 8, 16, 24, 32, 64.")
    
    # Save the enhanced speech to the output file
    wavfile.write(outfile, fs, enhanced_speech_int)

# Example usage:
# specsub('sp04_babble_sn10.wav', 'out_specsub.wav')

