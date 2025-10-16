"""
Integration of TinyGRUVAD (from WF_gru_vad.ipynb) with Wiener Filter.

This module provides a lightweight VAD integration using the ~2K parameter
TinyGRUVAD model that works with mel-spectrogram features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple


class TinyGRUVAD(nn.Module):
    """Light GRU-based VAD, causal, hearing-aid friendly (~3.5k params with 24 mel bands)."""
    def __init__(self, input_dim=48, hidden_dim=16, dropout=0.1):
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
    """Wrapper for using TinyGRUVAD model in Wiener filter integration."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        n_mels: int = 24,
    ):
        """Initialize TinyGRUVAD processor.
        
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            device: Device to run model on (defaults to CUDA if available)
            threshold: Probability threshold for VAD decision (default: 0.5)
            n_mels: Number of mel bands (default: 24, matching training)
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


def wiener_filter_with_tiny_vad(
    noisy_audio: torch.Tensor,
    fs: int,
    vad_model_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    vad_threshold: float = 0.5,
    frame_dur_ms: int = 8,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Wiener filter with TinyGRUVAD frame-by-frame VAD supplement.
    
    This version processes each frame causally, computing VAD decision on-the-fly
    using mel-spectrogram features while performing Wiener filtering in the STFT domain.
    The GRU hidden state is maintained across frames for true streaming processing.
    
    Args:
        noisy_audio: Input noisy speech signal (mono, 1D tensor)
        fs: Sampling frequency in Hz
        vad_model_path: Path to trained TinyGRUVAD model checkpoint
        output_dir: Directory to save enhanced audio
        output_file: Output filename prefix
        input_name: Input filename for metadata
        mu: Noise power update parameter (0 < mu < 1)
        a_dd: Decision-directed a priori SNR smoothing (0 < a_dd < 1)
        vad_threshold: VAD probability threshold
        frame_dur_ms: Frame duration in milliseconds
        
    Returns:
        Tuple of (enhanced_signal, sample_rate)
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    waveform = noisy_audio.clone().to(device)
    input_name = input_name if input_name is not None else "WF_TinyVAD_"
    
    # Load TinyGRUVAD
    vad_processor = TinyVADProcessor(
        model_path=vad_model_path,
        device=device,
        threshold=vad_threshold,
    )
    
    # --- Wiener Filter Processing ---
    print("\nApplying frame-by-frame Wiener filter with TinyGRUVAD...")
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2
    
    # Hann windows
    hann = torch.hann_window(frame_samples, periodic=False, device=device)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()
    U = (analysis_win @ analysis_win) / frame_samples
    
    # Setup for mel-spectrogram VAD features
    n_fft = frame_samples
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=vad_processor.n_mels,
        sample_rate=fs,
        n_stft=n_fft // 2 + 1
    ).to(device)
    
    # --- Initial noise PSD estimate ---
    len_120ms = int(fs * 0.120)
    init_seg = waveform[:len_120ms]
    nsubframes = max(1, (len(init_seg) - frame_samples) // hop + 1)
    
    noise_ps = torch.zeros(frame_samples, device=device)
    for j in range(nsubframes):
        seg = init_seg[j * hop:j * hop + frame_samples]
        if seg.numel() < frame_samples:
            seg = torch.nn.functional.pad(seg, (0, frame_samples - seg.numel()))
        wseg = seg * analysis_win
        X = torch.fft.fft(wseg, n=frame_samples)
        noise_ps += (X.abs() ** 2) / (frame_samples * U)
    noise_ps /= nsubframes
    
    # --- Prepare output ---
    n_frames = (len(waveform) - frame_samples) // hop + 1
    out_len = (n_frames - 1) * hop + frame_samples
    enhanced = torch.zeros(out_len, device=device)
    norm = torch.zeros(out_len, device=device)
    
    # --- State variables ---
    G_prev = torch.ones(frame_samples, device=device)
    posteri_prev = torch.ones(frame_samples, device=device)
    
    # VAD state variables
    vad_hidden = None  # GRU hidden state
    prev_log_mel = None  # For computing delta features
    speech_frame_count = 0
    
    # --- Process each frame (TRUE FRAME-BY-FRAME) ---
    for j in range(n_frames):
        n_start = j * hop
        frame = waveform[n_start:n_start + frame_samples]
        if frame.numel() < frame_samples:
            frame = torch.nn.functional.pad(frame, (0, frame_samples - frame.numel()))
        
        # Apply window and FFT for Wiener filtering
        win_frame = frame * analysis_win
        X = torch.fft.fft(win_frame, n=frame_samples)
        noisy_ps = (X.abs() ** 2) / (frame_samples * U)
        
        # --- Frame-by-frame VAD decision using TinyGRUVAD ---
        # Compute mel features from current frame's STFT
        pspec_frame = X.abs() ** 2  # Power spectrum (frame_samples,)
        # MelScale expects (n_freqs, n_frames), so reshape: (n_fft//2+1, 1)
        pspec_for_mel = pspec_frame[:n_fft//2+1].unsqueeze(1)  # (n_fft//2+1, 1)
        mel_frame = mel_transform(pspec_for_mel).clamp_min(1e-8)  # (n_mels, 1)
        log_mel_frame = torch.log(mel_frame.squeeze(1) + 1e-8)  # (n_mels,)
        
        # Compute delta feature (causal: use previous frame)
        if prev_log_mel is None:
            delta_frame = torch.zeros_like(log_mel_frame)
        else:
            delta_frame = log_mel_frame - prev_log_mel
        prev_log_mel = log_mel_frame.clone()
        
        # Concatenate log-mel + delta for VAD input
        vad_features = torch.cat([log_mel_frame, delta_frame]).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels*2)
        
        # Get VAD decision for this single frame
        vad_decision, vad_prob, vad_hidden = vad_processor.predict_frame_features(
            vad_features, 
            vad_hidden
        )
        
        if vad_decision:
            speech_frame_count += 1
        
        # --- SNR estimation (Wiener filter) ---
        if j == 0:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev**2) * posteri_prev + (1 - a_dd) * posteri_prime
        
        # Update noise estimate using frame-by-frame TinyGRUVAD decision
        if not vad_decision:  # Non-speech frame
            noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
        
        # Wiener gain
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))
        
        # Apply gain + IFFT
        Y = X * G
        y_ifft = torch.fft.ifft(Y).real
        
        # WOLA synthesis
        synth_seg = y_ifft * synth_win
        enhanced[n_start:n_start + frame_samples] += synth_seg
        norm[n_start:n_start + frame_samples] += synth_win**2
        
        # Update states
        G_prev = G
        posteri_prev = posteri
    
    # Normalize WOLA overlap
    mask = norm > 1e-8
    enhanced[mask] /= norm[mask]
    
    # Trim to original length
    enhanced = enhanced[:len(waveform)]
    
    # Print VAD statistics
    print(f"VAD: {speech_frame_count}/{n_frames} frames detected as speech "
          f"({speech_frame_count/n_frames*100:.1f}%)")
    
    # Save if requested
    if output_dir is not None and output_file is not None:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_parts = [
            f"FRAME{frame_dur_ms}ms",
            f"MU{mu:.3f}".replace('.', 'p'),
            f"ADD{a_dd:.3f}".replace('.', 'p'),
            f"TinyVAD{vad_threshold:.2f}".replace('.', 'p')
        ]
        
        output_file = output_file.replace(".wav", "")
        input_name = input_name.replace(".wav", "")
        
        output_filename = f"{output_file}_{input_name}_{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        
        torchaudio.save(full_output_path, enhanced.unsqueeze(0).cpu(), fs)
        print(f"\nEnhanced audio saved to: {full_output_path}")
    
    return enhanced.cpu(), fs


if __name__ == "__main__":
    # Example usage
    print("TinyGRUVAD Integration Test")
    print("=" * 60)
    
    # Test loading a model (update path to your trained model)
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tiny_vad_best.pth'
    
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Please train the model using WF_gru_vad.ipynb first")
    else:
        # Load processor
        processor = TinyVADProcessor(model_path)
        
        # Test with random audio
        test_audio = torch.randn(16000)  # 1 second at 16kHz
        
        print("\nProcessing test audio...")
        decisions, probs = processor.predict_audio(test_audio, 16000)
        
        print(f"Processed {len(decisions)} frames")
        print(f"Speech frames: {decisions.sum()} ({decisions.mean()*100:.1f}%)")
        print(f"Average speech probability: {probs.mean():.3f}")
