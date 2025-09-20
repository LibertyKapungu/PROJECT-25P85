import torch
import torchaudio
from scipy.signal.windows import hamming

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
#   - It uses torchaudio for audio I/O and PyTorch tensors for processing.
############################################################

def wiener_as_torch(filename: str, outfile: str=None, device='cpu'):
    """
    Wiener filtering algorithm using torchaudio and PyTorch tensors.
    
    filename - noisy speech file in .wav format
    outfile  - enhanced output file in .wav format
    device   - 'cpu' or 'cuda'
    """
    # Load waveform using torchaudio
    waveform, fs = torchaudio.load(filename)  # shape: [channels, samples]
    waveform = waveform.mean(dim=0)  # convert to mono if stereo
    waveform = waveform.to(device)
    
    # Parameters
    mu = 0.98
    a_dd = 0.98
    eta = 0.15
    frame_dur = 20
    L = int(frame_dur * fs / 1000)
    hamming_win = torch.tensor(hamming(L), dtype=torch.float32, device=device)
    U = (hamming_win @ hamming_win) / L
    
    # First 120ms assumed noise-only
    len_120ms = int(fs / 1000 * 120)
    first_120ms = waveform[0:len_120ms]
    
    # Noise power spectrum estimation using Welch's method
    nsubframes = int(torch.floor(torch.tensor(len_120ms / (L / 2)))) - 1  # 50% overlap
    noise_ps = torch.zeros(L, device=device)
    n_start = 0
    
    for _ in range(nsubframes):
        noise = first_120ms[n_start:n_start + L] * hamming_win
        noise_fft = torch.fft.fft(noise, n=L)
        noise_ps += (torch.abs(noise_fft) ** 2) / (L * U)
        n_start += L // 2
    
    noise_ps = noise_ps / nsubframes
    
    # Frame processing
    len1 = L // 2
    nframes = int(torch.floor(torch.tensor(len(waveform) / len1))) - 1
    n_start = 0
    
    enhanced_speech = torch.zeros_like(waveform, device=device)
    vad = torch.zeros_like(waveform, device=device)
    vad_decision = torch.zeros(nframes, device=device)
    
    for j in range(nframes):
        noisy = waveform[n_start:n_start + L] * hamming_win
        noisy_fft = torch.fft.fft(noisy, n=L)
        noisy_ps = (torch.abs(noisy_fft) ** 2) / (L * U)
        
        # Voice activity detection
        if j == 0:
            posteri = noisy_ps / noise_ps
            posteri_prime = torch.clamp(posteri - 1, min=0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / noise_ps
            posteri_prime = torch.clamp(posteri - 1, min=0)
            priori = a_dd * (G_prev ** 2) * posteri_prev + (1 - a_dd) * posteri_prime
        
        log_sigma_k = posteri * priori / (1 + priori) - torch.log1p(priori)
        vad_decision[j] = torch.sum(log_sigma_k) / L
        
        if vad_decision[j] < eta:
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
            vad[n_start:n_start + L] = 0
        else:
            vad[n_start:n_start + L] = 1
        
        G = torch.sqrt(priori / (1 + priori))
        enhanced = torch.fft.ifft(noisy_fft * G, n=L).real
        
        if j == 0:
            enhanced_speech[n_start:n_start + L//2] = enhanced[:L//2]
        else:
            enhanced_speech[n_start:n_start + L//2] = overlap + enhanced[:L//2]
        
        overlap = enhanced[L//2:]
        n_start += L // 2
        
        G_prev = G
        posteri_prev = posteri
    
    enhanced_speech[n_start:n_start + L//2] = overlap
    
    # Save the enhanced speech
    enhanced_speech = enhanced_speech.clamp(-1.0, 1.0)  # torchaudio expects float32 -1 to 1
    
    if outfile is not None:
        torchaudio.save(outfile, enhanced_speech.unsqueeze(0).cpu(), fs)
    else:
        return enhanced_speech

# Example usage
# wiener_as_torch('sp04_babble_sn10.wav', 'out_wien_as_torch.wav')
