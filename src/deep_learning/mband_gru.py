import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft
import scipy.signal
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
import torchaudio

import torch.nn as nn
import torch.nn.functional as F

"""
Integration of TinyGRUVAD (from WF_gru_vad.ipynb) with Spectral Filter.

This module provides a lightweight VAD integration using the ~2K parameter
TinyGRUVAD model that works with mel-spectrogram features.
"""

class TinyGRUVAD(nn.Module):
    """Light GRU-based VAD, causal, hearing-aid friendly (~2 k params)."""
    def __init__(self, input_dim=32, hidden_dim=16, dropout=0.1):
        super().__init__()
        # use no right-padding in Conv1d; we'll apply left (causal) padding in forward
        self.pre = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=0, groups=input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h=None):
        # x: (B,T,F)
        x = x.transpose(1,2)              # (B,F,T)
        # causal pad: pad (kernel_size-1) frames on the left only so conv doesn't see future frames
        k = self.pre.kernel_size[0] if isinstance(self.pre.kernel_size, (list, tuple)) else self.pre.kernel_size
        pad_left = k - 1
        x = F.pad(x, (pad_left, 0))       # pad on time dimension (left, right)
        x = self.pre(x).transpose(1,2)    # local causal conv
        x = self.norm(x)
        out, h = self.gru(x, h)
        out = self.drop(out)
        # return raw logits (B,T,1); use BCEWithLogitsLoss for stability
        logits = self.fc(out)
        return logits, h


class TinyVADProcessor:
    """Wrapper for using TinyGRUVAD model in Spectral filter integration."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        n_mels: int = 16,
    ):
        """Initialize TinyGRUVAD processor.
        
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            device: Device to run model on (defaults to CUDA if available)
            threshold: Probability threshold for VAD decision (default: 0.5)
            n_mels: Number of mel bands (default: 16)
        """
        self.model_path = Path(model_path)
        self.device = device if device is not None else \
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.n_mels = n_mels
        
        # Load model
        self.model = TinyGRUVAD(input_dim=n_mels*2, hidden_dim=16).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        
        print(f"TinyGRUVAD loaded from {self.model_path}")
        print(f"Device: {self.device}")
    
    def extract_features(
        self, 
        audio: torch.Tensor, 
        fs: int,
        frame_len_ms: float = 8.0,
        hop_len_ms: float = 4.0,
    ) -> torch.Tensor:
        """Extract log-mel + delta features from audio.
        
        Args:
            audio: Audio waveform (samples,)
            fs: Sample rate
            frame_len_ms: Frame length in milliseconds
            hop_len_ms: Hop length in milliseconds
            
        Returns:
            Features tensor of shape (1, T, n_mels*2)
        """
        n_fft = int(fs * frame_len_ms / 1000)
        hop = int(fs * hop_len_ms / 1000)
        win = torch.hann_window(n_fft, device=audio.device)
        
        # STFT with center=False for causal processing
        spec = torch.stft(
            audio, 
            n_fft, 
            hop, 
            window=win, 
            center=False, 
            return_complex=True
        )
        pspec = spec.abs() ** 2
        
        # Mel filterbank
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=self.n_mels, 
            sample_rate=fs, 
            n_stft=n_fft//2+1
        ).to(audio.device)
        
        mel = mel_transform(pspec).clamp_min(1e-8)
        log_mel = torch.log(mel.T + 1e-8)  # (T, n_mels)
        
        # Delta features (first-order difference)
        delta = torch.zeros_like(log_mel)
        delta[1:] = log_mel[1:] - log_mel[:-1]
        
        # Concatenate: (T, n_mels*2)
        features = torch.cat([log_mel, delta], dim=1).unsqueeze(0)  # (1, T, n_mels*2)
        
        return features
    
    def predict_audio(
        self,
        audio: torch.Tensor,
        fs: int,
        frame_len_ms: float = 8.0,
        hop_len_ms: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict VAD for entire audio.
        
        Args:
            audio: Audio waveform (samples,)
            fs: Sample rate
            frame_len_ms: Frame length in ms
            hop_len_ms: Hop length in ms
            
        Returns:
            Tuple of (decisions, probabilities):
                - decisions: Binary array (num_frames,)
                - probabilities: Probability array (num_frames,)
        """
        audio = audio.to(self.device)
        
        # Extract features
        features = self.extract_features(audio, fs, frame_len_ms, hop_len_ms)
        
        # Predict
        with torch.no_grad():
            logits, _ = self.model(features)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        decisions = (probs >= self.threshold).astype(bool)
        
        return decisions, probs
    
    def predict_frame_features(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, float, torch.Tensor]:
        """Predict VAD from pre-computed features (single frame or sequence).
        
        Args:
            features: Features tensor (1, T, n_mels*2)
            hidden: Hidden state from previous prediction
            
        Returns:
            Tuple of (decision, probability, new_hidden)
        """
        with torch.no_grad():
            features = features.to(self.device)
            logits, hidden = self.model(features, hidden)
            prob = torch.sigmoid(logits).squeeze().cpu().item()
            decision = prob >= self.threshold
        
        return decision, prob, hidden

def berouti(SNR):
    """Berouti's algorithm for computing over-subtraction factor"""
    a = np.ones_like(SNR)
    a[(SNR >= -5.0) & (SNR <= 20)] = 4 - SNR[(SNR >= -5.0) & (SNR <= 20)] * 3 / 20
    a[SNR < -5.0] = 4.75
    return a

def noiseupdt(x_magsm, n_spect, cmmnlen, nframes):
    """Voice Activity Detection and noise spectrum update"""
    SPEECH = 1
    SILENCE = 0
    
    # Initialize arrays
    state = np.zeros(nframes * cmmnlen, dtype=int)
    judgevalue1 = np.zeros(nframes * cmmnlen)

    i = 0   # Process first frame
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

def noiseupdt_with_gru_vad(
    x_magsm: np.ndarray,
    n_spect: np.ndarray,
    cmmnlen: int,
    nframes: int,
    vad_processor,  # TinyVADProcessor instance
    noisy_speech: torch.Tensor,
    fs: int,
    frmelen: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voice Activity Detection using TinyGRUVAD and noise spectrum update.
    
    This function replaces the LRT-based VAD in the original noiseupdt() 
    with a GRU-based VAD that uses mel-spectrogram features.
    
    Args:
        x_magsm: Smoothed magnitude spectrum (fftl, nframes)
        n_spect: Noise spectrum estimate (fftl, nframes)
        cmmnlen: Common samples between frames (hop size)
        nframes: Number of frames
        vad_processor: TinyVADProcessor instance with loaded model
        noisy_speech: Original noisy audio waveform
        fs: Sampling frequency
        frmelen: Frame length in samples
        device: torch device
        
    Returns:
        Tuple of (updated_noise_spectrum, state_array)
        - updated_noise_spectrum: (fftl, nframes) noise estimates
        - state_array: (nframes * cmmnlen,) binary speech/silence flags
    """
    SPEECH = 1
    SILENCE = 0
    
    # Initialize arrays
    state = np.zeros(nframes * cmmnlen, dtype=int)
    
    # Extract features for entire audio using TinyGRUVAD's feature extraction
    # This uses the same mel-spectrogram approach as the trained model
    audio_tensor = noisy_speech.to(device)
    
    # Calculate hop length for VAD feature extraction
    # Match the frame timing used in spectral processing
    frame_len_ms = (frmelen / fs) * 1000
    hop_len_ms = (cmmnlen / fs) * 1000
    
    # Get VAD decisions for all frames
    # This extracts mel features at the correct frame rate
    vad_decisions, vad_probs = vad_processor.predict_audio(
        audio_tensor,
        fs,
        frame_len_ms=frame_len_ms,
        hop_len_ms=hop_len_ms
    )
    
    # Handle potential frame count mismatch (due to STFT vs mel extraction differences)
    if len(vad_decisions) != nframes:
        print(f"Warning: VAD frame count ({len(vad_decisions)}) != spectral frames ({nframes})")
        # Truncate or pad to match
        if len(vad_decisions) > nframes:
            vad_decisions = vad_decisions[:nframes]
            vad_probs = vad_probs[:nframes]
        else:
            # Pad with last decision
            pad_len = nframes - len(vad_decisions)
            vad_decisions = np.pad(vad_decisions, (0, pad_len), mode='edge')
            vad_probs = np.pad(vad_probs, (0, pad_len), mode='edge')
    
    # Process each frame with GRU-VAD decisions
    for i in range(nframes):
        # Get VAD decision for this frame
        is_speech = vad_decisions[i]
        
        # Update state array
        state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH if is_speech else SILENCE
        
        # Update noise spectrum based on VAD decision
        if i == 0:
            # First frame
            if not is_speech:  # SILENCE
                # Update noise estimate with exponential averaging
                n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i]**2 + 0.1 * x_magsm[:, i]**2)
            # else: keep initial noise estimate (from first Noisefr frames)
        else:
            # Subsequent frames
            if not is_speech:  # SILENCE
                # Update noise estimate with exponential averaging
                n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i-1]**2 + 0.1 * x_magsm[:, i]**2)
            else:  # SPEECH
                # Keep previous noise estimate (no update during speech)
                n_spect[:, i] = n_spect[:, i-1]
    
    return n_spect, state


def noiseupdt_with_gru_vad_causal(
    x_magsm: np.ndarray,
    n_spect: np.ndarray,
    cmmnlen: int,
    nframes: int,
    vad_processor,  # TinyVADProcessor instance
    noisy_speech: torch.Tensor,
    fs: int,
    frmelen: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TRUE FRAME-BY-FRAME causal VAD for real-time compatibility.
    
    This version maintains GRU hidden state across frames and processes
    each frame sequentially, making it suitable for real-time/streaming.
    
    Args:
        Same as noiseupdt_with_gru_vad
        
    Returns:
        Same as noiseupdt_with_gru_vad
    """
    SPEECH = 1
    SILENCE = 0
    
    # Initialize arrays
    state = np.zeros(nframes * cmmnlen, dtype=int)
    
    # Setup for frame-by-frame processing
    n_fft = frmelen
    win = torch.hann_window(frmelen, device=device)
    
    # Mel transform for VAD features
    mel_transform = torch.nn.Sequential(
        torchaudio.transforms.MelScale(
            n_mels=vad_processor.n_mels,
            sample_rate=fs,
            n_stft=n_fft // 2 + 1
        )
    ).to(device)
    
    # Initialize GRU hidden state
    vad_hidden = None
    prev_log_mel = None
    
    # Process each frame causally
    for i in range(nframes):
        # Get current frame from audio
        frame_start = i * cmmnlen
        frame_end = frame_start + frmelen
        
        if frame_end <= len(noisy_speech):
            frame = noisy_speech[frame_start:frame_end].to(device)
        else:
            # Pad last frame if needed
            frame = torch.nn.functional.pad(
                noisy_speech[frame_start:].to(device),
                (0, frmelen - (len(noisy_speech) - frame_start))
            )
        
        # Compute STFT for this frame
        windowed = frame * win
        X = torch.fft.fft(windowed, n=n_fft)
        pspec = X.abs() ** 2
        
        # Extract mel features for VAD
        pspec_for_mel = pspec[:n_fft//2+1].unsqueeze(1)  # (n_fft//2+1, 1)
        mel_frame = mel_transform[0](pspec_for_mel).clamp_min(1e-8)  # (n_mels, 1)
        log_mel_frame = torch.log(mel_frame.squeeze(1) + 1e-8)  # (n_mels,)
        
        # Compute delta features (causal)
        if prev_log_mel is None:
            delta_frame = torch.zeros_like(log_mel_frame)
        else:
            delta_frame = log_mel_frame - prev_log_mel
        prev_log_mel = log_mel_frame.clone()
        
        # Prepare VAD input: (1, 1, n_mels*2)
        vad_features = torch.cat([log_mel_frame, delta_frame]).unsqueeze(0).unsqueeze(0)
        
        # Get VAD decision from GRU
        is_speech, vad_prob, vad_hidden = vad_processor.predict_frame_features(
            vad_features,
            vad_hidden
        )
        
        # Update state array
        state[i*cmmnlen:(i+1)*cmmnlen] = SPEECH if is_speech else SILENCE
        
        # Update noise spectrum based on VAD decision
        if i == 0:
            if not is_speech:
                n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i]**2 + 0.1 * x_magsm[:, i]**2)
        else:
            if not is_speech:  # SILENCE - update noise
                n_spect[:, i] = np.sqrt(0.9 * n_spect[:, i-1]**2 + 0.1 * x_magsm[:, i]**2)
            else:  # SPEECH - keep previous estimate
                n_spect[:, i] = n_spect[:, i-1]
    
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
        vad_model_path: Union[str, Path] = None,
        vad_threshold: float = 0.5,
        use_causal_vad: bool = False,
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

     # Load TinyGRUVAD
    vad_processor = TinyVADProcessor(
        model_path=vad_model_path,
        device=device,
        threshold=vad_threshold
    )
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
        # Handle edge case of very short audio
        x_mag = np.array([]).reshape(fftl, 0)
        x_ph = np.array([]).reshape(fftl, 0)
        nframes = 0
        print("Warning: No frames generated - audio too short")

    if AVRGING:             # Smooth the input spectrum
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

    # # Noise update during silence frames    
    # if VAD:
    #     n_spect_expanded = np.tile(n_spect, (1, nframes))
    #     n_spect, state = noiseupdt(x_magsm, n_spect_expanded, cmmnlen, nframes)
    # else:
    #     # Replicate noise spectrum for all frames (no VAD)   
    #     n_spect = np.repeat(n_spect, nframes, axis=1)
   
    if VAD:
        n_spect_expanded = np.tile(n_spect, (1, nframes))
        
        if use_causal_vad:
            # Frame-by-frame causal processing (real-time compatible)
            n_spect, state = noiseupdt_with_gru_vad_causal(
                x_magsm, 
                n_spect_expanded, 
                cmmnlen, 
                nframes,
                vad_processor,
                noisy_speech,
                fs,
                frmelen,
                device
            )
        else:
            # Batch processing (offline, faster)
            n_spect, state = noiseupdt_with_gru_vad(
                x_magsm, 
                n_spect_expanded, 
                cmmnlen, 
                nframes,
                vad_processor,
                noisy_speech,
                fs,
                frmelen,
                device
            )
    else:
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
            signal_power = np.linalg.norm(x_magsm[start:stop, j], 2) ** 2
            noise_power = np.linalg.norm(n_spect[start:stop, j], 2) ** 2
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

# Example usage
# noisy_audio = torchaudio.load("C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\validation_dataset\\noisy_speech\\sp21_station_sn5.wav")[0]
# mband(noisy_audio, fs=8000, output_dir="C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\validation_dataset\\enhanced_speech", output_file="enhanced_sp21_station_sn5.wav", input_name="sp21_station_sn5", Nband=4, Freq_spacing='linear', FRMSZ=8, OVLP=50, AVRGING=1, Noisefr=1, FLOOR=0.002, VAD=1)


    
# # Load audio
#     #noisy_audio, fs = torchaudio.load("noisy_speech.wav")
# noisy_audio = torchaudio.load("C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\validation_dataset\\noisy_speech\\sp21_station_sn5.wav")[0]

# # Path to your trained TinyGRUVAD model
# vad_model_path = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\models\\GRU_VAD\\tiny_vad_best.pth"
    
#     # Process with GRU-VAD
# enhanced, fs = mband(
#     noisy_audio=noisy_audio,
#     fs=8000,
#     vad_model_path=vad_model_path,
#     output_dir="output/",
#     output_file="enhanced.wav",
#     Nband=4,
#     Freq_spacing='linear',
#     FRMSZ=8,
#     OVLP=50,
#     AVRGING=1,  # KEEP THIS!
#     VAD=1,
#     vad_threshold=0.1,
#     use_causal_vad=False  # False = faster batch, True = real-time compatible
# )