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

def wiener_as_torch(input_dir: str, input_file: str, output_dir: str=None, output_file: str=None):
    """
    Wiener filtering algorithm using torchaudio.

    Parameters:
        input_dir  - directory of the noisy input speech file
        input_file - name of the noisy input speech file (e.g., 'noisy.wav')
        output_dir - directory where the enhanced speech will be saved
        output_file - name of the output enhanced speech file (e.g., 'enhanced.wav')
    
    Returns:
        enhanced_speech tensor if output_dir/output_file not provided
    """
    # Automatically select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build full input path
    input_path = os.path.join(input_dir, input_file)

    # Load waveform
    waveform, fs = torchaudio.load(input_path)
    waveform = waveform.mean(dim=0)  # convert to mono
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
    nsubframes = int(torch.floor(torch.tensor(len_120ms / (L / 2)))) - 1
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

        # ================= Voice activity detection =================
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
            enhanced_speech[n_start:n_start + L // 2] = enhanced[:L // 2]
        else:
            enhanced_speech[n_start:n_start + L // 2] = overlap + enhanced[:L // 2]

        overlap = enhanced[L // 2:]
        n_start += L // 2

        G_prev = G
        posteri_prev = posteri

    enhanced_speech[n_start:n_start + L // 2] = overlap
    enhanced_speech = enhanced_speech.clamp(-1.0, 1.0)

    if output_dir is not None and output_file is not None:
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(input_file)[0] 
        output_filename = f"{os.path.splitext(output_file)[0]}_{base_name}.wav"

        output_path = os.path.join(output_dir, output_filename)
        torchaudio.save(output_path, enhanced_speech.unsqueeze(0).cpu(), fs)
        print(f"Enhanced file saved to: {output_path}")
    else:
        return enhanced_speech, fs

# Example usage
wiener_as_torch('audio_stuff/', 'sp21_station_sn0.wav', 'wiener_filter_priori', 'wiener')
