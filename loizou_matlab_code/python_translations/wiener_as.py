import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import sys

############################################################
# Wiener_AS Algorithm (Python Translation)
#
# This implementation is translated from the MATLAB code by
# Philipos C. Loizou, provided on the CD that accompanies his book:
#
#   "Speech Enhancement: Theory and Practice, 2nd Edition"
#   ['https://www.routledge.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781466504219']
#
# The original MATLAB code:
#   Authors: Yi Hu and Philipos C. Loizou
#   Copyright (c) 2006 by Philipos C. Loizou
#   $Revision: 0.0 $   $Date: 10/09/2006 $
#
# References:
#   [1] Scalart, P. and Filho, J. (1996).
#       "Speech enhancement based on a priori signal to noise estimation."
#       Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing, 629–632.
#
# Notes:
#   - This Python version follows the original algorithm structure.
#   - Some parameter values may need tuning for optimal performance.
############################################################


def wiener_as(filename: str, outfile: str):
    """
    Implements the Wiener filtering algorithm based on a priori SNR estimation [1].
    
    Usage:  wiener_as(noisyFile, outputFile)
            
          filename - noisy speech file in .wav format
          outfile - enhanced output file in .wav format
          
    Example call:  wiener_as('sp04_babble_sn10.wav','out_wien_as.wav')
    --------------------------------------------------------------------------------
    """

    if filename is None or outfile is None:
        print('Usage: wiener_as(noisyfile.wav, outFile.wav)\n')
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
    
    # column vector noisy_speech

    # set parameter values
    mu = 0.98  # smoothing factor in noise spectrum update
    a_dd = 0.98  # smoothing factor in priori update
    eta = 0.15  # VAD threshold
    frame_dur = 20  # frame duration 
    L = int(frame_dur * fs / 1000)  # L is frame length (160 for 8k sampling rate)
    hamming_win = hamming(L)  # hamming window
    U = np.dot(hamming_win, hamming_win) / L  # normalization factor
    
    # first 120 ms is noise only
    len_120ms = int(fs / 1000 * 120)
    first_120ms = noisy_speech[0:len_120ms]
    
    # =============now use Welch's method to estimate power spectrum with
    # Hamming window and 50% overlap
    nsubframes = int(np.floor(len_120ms / (L / 2))) - 1  # 50% overlap
    noise_ps = np.zeros(L)
    n_start = 0  # Python uses 0-based indexing
    
    for j in range(nsubframes):
        noise = first_120ms[n_start:n_start + L]
        noise = noise * hamming_win
        noise_fft = fft(noise, L)
        noise_ps = noise_ps + (np.abs(noise_fft) ** 2) / (L * U)
        n_start = n_start + int(L / 2)
    
    noise_ps = noise_ps / nsubframes
    
    # ==============
    # number of noisy speech frames 
    len1 = int(L / 2)  # with 50% overlap
    nframes = int(np.floor(len(noisy_speech) / len1)) - 1
    n_start = 0
    
    # Initialize arrays
    enhanced_speech = np.zeros(len(noisy_speech))
    vad = np.zeros(len(noisy_speech))
    vad_decision = np.zeros(nframes)
    
    for j in range(nframes):
        noisy = noisy_speech[n_start:n_start + L]
        noisy = noisy * hamming_win
        noisy_fft = fft(noisy, L)
        noisy_ps = (np.abs(noisy_fft) ** 2) / (L * U)
        
        # ============ voice activity detection ============
        if j == 0:  # initialize posteri
            posteri = noisy_ps / noise_ps
            posteri_prime = posteri - 1
            posteri_prime[posteri_prime < 0] = 0
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / noise_ps
            posteri_prime = posteri - 1
            posteri_prime[posteri_prime < 0] = 0
            priori = a_dd * (G_prev ** 2) * posteri_prev + (1 - a_dd) * posteri_prime
        
        log_sigma_k = posteri * priori / (1 + priori) - np.log(1 + priori)
        vad_decision[j] = np.sum(log_sigma_k) / L
        
        if vad_decision[j] < eta:
            # noise only frame found
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
            vad[n_start:n_start + L] = 0
        else:
            vad[n_start:n_start + L] = 1
        
        # ============ end of vad ============
        
        G = np.sqrt(priori / (1 + priori))  # gain function
        
        enhanced = ifft(noisy_fft * G, L)
        enhanced = np.real(enhanced)  # Take real part after IFFT
        
        if j == 0:
            enhanced_speech[n_start:n_start + int(L/2)] = enhanced[0:int(L/2)]
        else:
            enhanced_speech[n_start:n_start + int(L/2)] = overlap + enhanced[0:int(L/2)]
        
        overlap = enhanced[int(L/2):L]
        n_start = n_start + int(L / 2)
        
        G_prev = G
        posteri_prev = posteri
    
    enhanced_speech[n_start:n_start + int(L/2)] = overlap
    
    # Convert back to original format for saving - mimicking MATLAB's wavwrite behavior (see waveread docs  for details)
    if nbits == 8: # Convert back to 0-255 range for uint8
        enhanced_speech_int = np.clip(enhanced_speech * 128.0 + 128.0, 0, 255).astype(np.uint8)
    
    elif nbits == 16: # Convert back to -32768 to +32767 range for int16
        enhanced_speech_int = np.clip(enhanced_speech * 32768.0, -32768, 32767).astype(np.int16)
    
    elif nbits == 24: # Convert back to -2^23 to 2^23-1 range, stored as int32
        enhanced_speech_int = np.clip(enhanced_speech * (2**23), -(2**23), 2**23 - 1).astype(np.int32)
    
    elif nbits == 32:
        if data.dtype == np.float32: # Keep as float32 for 32-bit float data
            enhanced_speech_int = np.clip(enhanced_speech, -1.0, 1.0).astype(np.float32)
        
        else: # Convert back to -2^31 to 2^31-1 range for int32
            enhanced_speech_int = np.clip(enhanced_speech * (2**31), -(2**31), 2**31 - 1).astype(np.int32)
    
    elif nbits == 64: # Keep as float64
        enhanced_speech_int = np.clip(enhanced_speech, -1.0, 1.0)
    
    else: # Raise error for unsupported bit depths
        raise ValueError(f"Unsupported bit depth: {nbits}. Supported values are 8, 16, 24, 32, 64.")
    
    wavfile.write(outfile, fs, enhanced_speech_int)

# Example usage:
# wiener_as('sp04_babble_sn10.wav', 'out_wien_as.wav')