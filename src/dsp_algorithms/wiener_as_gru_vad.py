############################################################
# Wiener_AS Algorithm + GRU-VAD Integration (Loizou 2e Adapted)
#
# Original algorithm by Yi Hu & Philipos C. Loizou, 2006
# GRU-VAD integration inspired by:
#   Xia & Stern (2018). A Gated Recurrent Unit Based Robust Voice Activity Detection.
#
# This hybrid system uses a lightweight GRU VAD model to adaptively control
# the noise update in the Wiener filter, improving robustness under nonstationary
# noise while remaining computationally efficient for low-power devices.
############################################################

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple

############################################################
# === Tiny GRU-VAD Model Definition ===
############################################################
class TinyVADGRU(torch.nn.Module):
    """Lightweight GRU-based VAD for speech-presence estimation."""
    def __init__(self, input_dim=32, hidden_dim=16):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h_prev=None):
        out, h = self.gru(x, h_prev)
        p = self.sigmoid(self.fc(out))
        return p, h


############################################################
# === Helper for VAD Feature Computation ===
############################################################
def compute_vad_feature_from_X(X_fft, mel_fb, prev_logE=None):
    """
    Compute 32-dim VAD feature vector (log-mel + delta) from FFT magnitude.
    X_fft: complex spectrum of one frame
    mel_fb: mel filterbank (n_mels x n_freqs)
    prev_logE: previous frame log-mel vector (for delta)
    Returns: feat (2*n_mels,), new prev_logE
    """
    npos = X_fft.shape[0] // 2 + 1
    mag2 = (X_fft.abs()[:npos] ** 2).to(mel_fb.device)
    melE = torch.matmul(mel_fb.T, mag2).clamp_min(1e-8)
    logE = torch.log(melE)
    if prev_logE is None:
        delta = torch.zeros_like(logE)
    else:
        delta = logE - prev_logE
    feat = torch.cat([logE, delta], dim=0)
    return feat, logE.detach()


############################################################
# === Wiener Filter with GRU-VAD ===
############################################################
def wiener_filter_gru(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.15,
    frame_dur_ms: int = 32,    # match GRU-VAD training
) -> Optional[Tuple[torch.Tensor, int]]:

    """Wiener filter with GRU-VAD guided noise update."""

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    waveform = noisy_audio.clone().to(device)
    input_name = input_name or "WF_"

    # --- Parameter checks ---
    if not 0 < mu < 1: raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1: raise ValueError("a_dd must be between 0 and 1")
    if eta <= 0: raise ValueError("eta must be positive")

    ########################################################
    # Frame setup
    ########################################################
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2

    hann = torch.hann_window(frame_samples, periodic=False, device=device)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()
    U = (analysis_win @ analysis_win) / frame_samples

    ########################################################
    # === Load GRU-VAD Model ===
    ########################################################
    vad_model_path = Path("models") / "tiny_vad_gru_best.pth"
    n_bands = 16
    input_dim = 2 * n_bands
    vad_device = torch.device("cpu")  # run VAD on CPU (quantized-safe)

    try:
        mel_fb = torchaudio.functional.create_fb_matrix(
            n_mels=n_bands,
            n_freqs=frame_samples // 2 + 1,
            f_min=0.0,
            f_max=fs / 2,
        ).to(vad_device)
    except AttributeError:
        # fallback for older torchaudio versions (e.g., 2.8.0)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=frame_samples // 2 + 1,
            f_min=0.0,
            f_max=fs / 2,
            n_mels=n_bands,
            sample_rate=fs,
            norm="slaney",
            mel_scale="htk",
        ).to(vad_device)

    # Load model
    vad = TinyVADGRU(input_dim=input_dim, hidden_dim=16).to(vad_device)
    if vad_model_path.exists():
        vad.load_state_dict(torch.load(vad_model_path, map_location=vad_device))
        vad.eval()
        print(f"[INFO] Loaded VAD model from {vad_model_path}")
    else:
        print(f"[WARN] VAD model not found at {vad_model_path}, fallback to heuristic VAD")
        vad = None

    # GRU hidden and feature states
    h_prev = None
    prev_logE = None
    p_smooth = 0.0
    alpha_smooth = 0.6  # smoothing factor

    ########################################################
    # === Initial noise PSD estimate (first 120 ms) ===
    ########################################################
    len_120ms = int(fs * 0.120)
    init_seg = waveform[:len_120ms]
    nsubframes = max(1, (len(init_seg) - frame_samples) // hop + 1)
    noise_ps = torch.zeros(frame_samples, device=device)

    for j in range(nsubframes):
        seg = init_seg[j * hop:j * hop + frame_samples]
        if seg.numel() < frame_samples:
            seg = torch.nn.functional.pad(seg, (0, frame_samples - seg.numel()))
        X = torch.fft.fft(seg * analysis_win, n=frame_samples)
        noise_ps += (X.abs() ** 2) / (frame_samples * U)
    noise_ps /= nsubframes

    ########################################################
    # === Prepare output buffers ===
    ########################################################
    n_frames = (len(waveform) - frame_samples) // hop + 1
    out_len = (n_frames - 1) * hop + frame_samples
    enhanced = torch.zeros(out_len, device=device)
    norm = torch.zeros(out_len, device=device)

    G_prev = torch.ones(frame_samples, device=device)
    posteri_prev = torch.ones(frame_samples, device=device)

    ########################################################
    # === Main Processing Loop ===
    ########################################################
    print(f"[INFO] Processing {n_frames} frames (~{len(waveform)/fs:.2f}s)")

    for j in range(n_frames):
        n_start = j * hop
        frame = waveform[n_start:n_start + frame_samples]
        if frame.numel() < frame_samples:
            frame = torch.nn.functional.pad(frame, (0, frame_samples - frame.numel()))

        # FFT
        X = torch.fft.fft(frame * analysis_win, n=frame_samples)
        noisy_ps = (X.abs() ** 2) / (frame_samples * U)

        # posteriori/priori SNR
        if j == 0:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev**2) * posteri_prev + (1 - a_dd) * posteri_prime

        ####################################################
        # === GRU-VAD Noise Update ===
        ####################################################
        if vad is None:
            # Fallback heuristic VAD
            log_sigma_k = posteri * priori / (1 + priori) - torch.log1p(priori)
            vad_decision = log_sigma_k.mean()
            if vad_decision < eta:
                noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
        else:
            # Compute VAD feature
            feat, prev_logE = compute_vad_feature_from_X(X.cpu(), mel_fb, prev_logE)
            vad_in = feat.unsqueeze(0).unsqueeze(0).to(vad_device)
            with torch.no_grad():
                p_frame, h_prev = vad(vad_in, h_prev)
            p = float(p_frame.squeeze().cpu().item())
            p_smooth = alpha_smooth * p_smooth + (1 - alpha_smooth) * p

            # adaptive noise PSD update
            noise_ps = p_smooth * noise_ps + (1 - p_smooth) * noisy_ps

        ####################################################
        # === Wiener Gain + Reconstruction ===
        ####################################################
        G = torch.sqrt(priori / (1.0 + priori + 1e-16))
        Y = X * G
        y_ifft = torch.fft.ifft(Y).real

        # Overlap-add
        synth_seg = y_ifft * synth_win
        enhanced[n_start:n_start + frame_samples] += synth_seg
        norm[n_start:n_start + frame_samples] += synth_win**2

        G_prev = G
        posteri_prev = posteri

    ########################################################
    # Normalize and output
    ########################################################
    mask = norm > 1e-8
    enhanced[mask] /= norm[mask]
    enhanced = enhanced[:len(waveform)]

    # Optionally save
    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_name = f"{Path(output_file).stem}_GRUVAD_FD{frame_dur_ms}ms.wav"
        full_out = output_path / out_name
        torchaudio.save(full_out, enhanced.unsqueeze(0).cpu(), fs)
        print(f"[INFO] Enhanced audio saved to {full_out}")

    return enhanced, fs


############################################################
# === Standalone test entry ===
############################################################
if __name__ == "__main__":
    # Quick test using torchaudio
    import time
    test_path = Path("data/test_noisy.wav")
    if not test_path.exists():
        print(f"[WARN] No test file at {test_path}")
    else:
        x, fs = torchaudio.load(str(test_path))
        x = x.squeeze(0)
        t0 = time.perf_counter()
        y, fs = wiener_filter_gru(x, fs)
        print(f"[INFO] Done. Duration: {time.perf_counter()-t0:.2f}s for {len(x)/fs:.2f}s audio.")
        torchaudio.save("enhanced_gru.wav", y.unsqueeze(0).cpu(), fs)
