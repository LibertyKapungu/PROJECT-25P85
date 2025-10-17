"""
Integration of Marblenet VAD with Wiener Filter.

This module provides a lightweight VAD integration using the 20ms frame based
Marblenet model.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Any

try:
    import nemo.collections.asr as nemo_asr
except ImportError:  # pragma: no cover - optional dependency
    nemo_asr = None


_MARBLENET_VAD_MODEL: Optional[Any] = None


def _load_marblenet_vad(device: torch.device) -> Any:
    """Load and cache the NVIDIA Marblenet VAD model on the requested device."""

    global _MARBLENET_VAD_MODEL
    if nemo_asr is None:
        raise ImportError(
            "nemo-toolkit is required for Marblenet VAD. Install `nemo_toolkit[asr]`."
        )

    if _MARBLENET_VAD_MODEL is None:
        model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
            model_name="nvidia/frame_vad_multilingual_marblenet_v2.0",
            strict=False,
        )
        model.eval()
        _MARBLENET_VAD_MODEL = model

    current_device = next(_MARBLENET_VAD_MODEL.parameters()).device
    if current_device != device:
        _MARBLENET_VAD_MODEL = _MARBLENET_VAD_MODEL.to(device)

    return _MARBLENET_VAD_MODEL


def _marblenet_vad_probabilities(
    waveform: torch.Tensor,
    fs: int,
    device: torch.device,
    vad_model: Any,
) -> Tuple[torch.Tensor, float]:
    """Run Marblenet VAD and return frame probabilities with their hop in seconds."""

    target_sr = 16000
    if fs != target_sr:
        waveform_16k = torchaudio.functional.resample(
            waveform.detach().cpu(), orig_freq=fs, new_freq=target_sr
        )
        waveform_16k = waveform_16k.to(device)
    else:
        waveform_16k = waveform

    signal = waveform_16k.unsqueeze(0)
    if signal.numel() == 0:
        raise ValueError("Input waveform is empty; cannot run VAD.")
    signal_len = torch.tensor([signal.shape[1]], device=device)

    with torch.no_grad():
        logits = vad_model(
            input_signal=signal.to(device),
            input_signal_length=signal_len,
        ).detach().cpu().squeeze(0)

    if logits.ndim == 2:
        # If logits has shape [num_frames, num_classes], take the speech class (index 1)
        if logits.shape[-1] == 2:
            logits = logits[:, 1]  # Take speech probability (class 1)
        else:
            logits = logits.squeeze(-1)

    probs = torch.sigmoid(logits)
    if probs.numel() == 0:
        raise RuntimeError("VAD model returned no frame probabilities.")

    samples_per_frame = waveform_16k.numel() / float(probs.numel())
    hop_seconds = max(samples_per_frame / target_sr, 1e-4)
    return probs, hop_seconds

def wiener_filter(
    noisy_audio: torch.Tensor,
    fs: int,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    input_name: Optional[str] = None,
    mu: float = 0.98,
    a_dd: float = 0.98,
    eta: float = 0.5,
    frame_dur_ms: int = 20,
    vad_model: Optional[Any] = None,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Implements the a-priori SNR-based Wiener filter for speech enhancement.

    This function implements the Wiener filtering algorithm based on a-priori SNR 
    estimation as described by Scalart and Filho (1996). The implementation uses
    weighted overlap-add (WOLA) processing with Hann windows and 50% overlap.

    Args:
        noisy_audio (torch.Tensor): Input noisy speech signal (mono, 1D tensor)
        fs (int): Sampling frequency in Hz
        output_dir (Optional[Union[str, Path]], optional): Directory to save enhanced audio. Defaults to None.
        output_file (Optional[str], optional): Output filename prefix. Defaults to None.
        input_name (Optional[str], optional): Input filename for metadata. Defaults to None.
        mu (float, optional): Noise power update parameter. Defaults to 0.98.
        a_dd (float, optional): Decision-directed a priori SNR smoothing. Defaults to 0.98.
        eta (float, optional): Marblenet speech probability threshold in (0, 1). Defaults to 0.5.
        frame_dur_ms (int, optional): Frame duration in milliseconds. Defaults to 20.
        vad_model (Optional[Any], optional): Preloaded Marblenet VAD model. Defaults to None.

    Returns:
        Optional[Tuple[torch.Tensor, int]]: Tuple containing:
            - Enhanced speech signal as torch.Tensor
            - Sampling frequency
            Returns None if output_dir and output_file are provided (saves to file instead)

    Raises:
        ValueError: If mu, a_dd not in (0,1), if eta not in (0,1), or if frame_dur_ms <= 0

    Notes:
        - Initial noise estimate uses first 120ms of signal
        - Uses CUDA if available, falls back to CPU
        - Implements VAD-based noise updating
        - Uses power complementary windows for perfect reconstruction
    """

    # --- device + basic setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model or _load_marblenet_vad(device)

    waveform = noisy_audio.clone()
    input_name = input_name if input_name is not None else "WF_"

    waveform = waveform.to(device)

    if not 0 < mu < 1:
        raise ValueError("mu must be between 0 and 1")
    if not 0 < a_dd < 1:
        raise ValueError("a_dd must be between 0 and 1")
    if not 0 < eta < 1:
        raise ValueError("eta must be a probability between 0 and 1")
    if frame_dur_ms <= 0:
        raise ValueError("frame_dur must be positive")

    vad_probs, vad_hop_sec = _marblenet_vad_probabilities(waveform, fs, device, vad_model)
    vad_len = vad_probs.numel()

    # --- Frame / window setup ---
    frame_samples = int(frame_dur_ms * fs / 1000)
    if frame_samples % 2 != 0:
        frame_samples += 1
    hop = frame_samples // 2

    # squared-root Hann analysis/synthesis windows
    hann = torch.hann_window(frame_samples, periodic=False, device=device)
    analysis_win = hann.sqrt()
    synth_win = analysis_win.clone()

    # normalization constant
    U = (analysis_win @ analysis_win) / frame_samples

    # --- Initial noise PSD estimate (first 120 ms) ---
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

    # --- State variables for decision-directed prior ---
    G_prev = torch.ones(frame_samples, device=device)
    posteri_prev = torch.ones(frame_samples, device=device)

    # --- Process each frame (causal WOLA loop) ---
    for j in range(n_frames):
        n_start = j * hop
        frame = waveform[n_start:n_start + frame_samples]
        if frame.numel() < frame_samples:
            frame = torch.nn.functional.pad(frame, (0, frame_samples - frame.numel()))

        win_frame = frame * analysis_win
        X = torch.fft.fft(win_frame, n=frame_samples)
        noisy_ps = (X.abs() ** 2) / (frame_samples * U)

        # posteriori & priori SNR
        if j == 0:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd + (1 - a_dd) * posteri_prime
        else:
            posteri = noisy_ps / (noise_ps + 1e-16)
            posteri_prime = torch.clamp(posteri - 1.0, min=0.0)
            priori = a_dd * (G_prev**2) * posteri_prev + (1 - a_dd) * posteri_prime

        # VAD / noise update
        frame_center_time = (n_start + frame_samples / 2) / fs
        vad_index = min(int(frame_center_time / vad_hop_sec), vad_len - 1)
        speech_prob = vad_probs[vad_index].item()
        if speech_prob < eta:
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

        # update states
        G_prev = G
        posteri_prev = posteri

    # normalize WOLA overlap
    mask = norm > 1e-8
    enhanced[mask] /= norm[mask]

    # trim to original length
    enhanced = enhanced[:len(waveform)]

    if output_dir is not None and output_file is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_parts = [
            f"FRAME{frame_dur_ms}ms",
            f"MU{mu:.3f}".replace('.', 'p'),
            f"ADD{a_dd:.3f}".replace('.', 'p'),
            f"ETA{eta:.3f}".replace('.', 'p')
        ]

        output_file = output_file.replace(".wav", "")
        input_name = input_name.replace(".wav", "")

        output_filename = f"{output_file}_{input_name}{'_'.join(metadata_parts)}.wav"
        full_output_path = output_path / output_filename
        torchaudio.save(full_output_path, enhanced.unsqueeze(0).cpu(), fs)
        print(f"Enhanced audio saved to: {full_output_path}")

    return enhanced, fs