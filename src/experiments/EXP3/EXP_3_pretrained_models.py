import torch
import torchaudio
import numpy as np
import os
import sys
from pathlib import Path

# ------------------------------
# Paths
# ------------------------------

current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

print(f"Source directory: {src_dir}\n")

pytorch_model_path = src_dir.parent / "models" / "pretrained" / "PYTORCH" / "denoiser_LSTM_Valetini.pt"
print(f"PyTorch model path: {pytorch_model_path}\n")

noisy_wav_path = src_dir.parent / "sound_data" / "processed" / "wiener_processed_outputs" / "EXP1_output" / "wiener_filter_priori_P107_S23_F10_SNR5_wiener_as__FRAME10ms_MU0p980_ADD0p950_ETA0p150_NONCAUSAL.wav"
print(f"Noisy WAV path: {noisy_wav_path}\n")

print("=== STARTING BENCHMARK ===\n")
# ------------------------------
# Load PyTorch ONNX-wrapper model
# ------------------------------
print(f"Loading PyTorch model from: {pytorch_model_path}")
model = torch.load(pytorch_model_path, map_location="cpu")
model.eval()
print(" Model loaded successfully.")

# ------------------------------
# Load and preprocess audio
# ------------------------------
waveform, sr = torchaudio.load(noisy_wav_path)
print(f"Original audio: shape={waveform.shape}, sr={sr}")

# Convert to mono
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample if needed
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sr = 16000

print(f"Preprocessed audio: shape={waveform.shape}, sr={sr}")

# ------------------------------
# STFT Configuration - MUST MATCH MODEL SPECS
# ------------------------------
sr = 16000
frame_duration = 0.025  # 25ms
hop_duration = 0.00625  # 6.25ms (25% of frame)

win_length = int(frame_duration * sr)  # 400 samples
hop_length = int(hop_duration * sr)    # 100 samples
n_fft = 512  # Next power of 2 above win_length

window = torch.hann_window(win_length)
eps = 1e-8

stft = torch.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    return_complex=True,
    center=True,
    normalized=False,
    onesided=True
)

magnitude = torch.abs(stft)
phase = torch.angle(stft)

# ------------------------------
# Input preprocessing - MODEL EXPECTS RAW MAGNITUDE
# ------------------------------
model_input = magnitude.unsqueeze(0) if magnitude.ndim == 2 else magnitude  # add batch dim
print(f"Input format: raw magnitude, shape: {model_input.shape}")

# ------------------------------
# Run PyTorch model (ONNX-wrapper)
# ------------------------------
print("Processing frames via PyTorch model...")
with torch.no_grad():
    gain_masks = model(model_input)  # [batch, freq_bins, frames]

# Ensure numpy for processing
if isinstance(gain_masks, torch.Tensor):
    gain_masks = gain_masks.numpy()
elif isinstance(gain_masks, list):
    gain_masks = np.stack([g.numpy() for g in gain_masks], axis=-1)

print(f"Gain masks shape: {gain_masks.shape}")
print(f"Gain masks range: [{gain_masks.min():.6f}, {gain_masks.max():.6f}]")

# ------------------------------
# Apply gain masks to get denoised magnitude
# ------------------------------
denoised_magnitude = magnitude.numpy() * gain_masks
denoised_stft = denoised_magnitude * np.exp(1j * phase.numpy())

# ------------------------------
# ISTFT back to waveform
# ------------------------------
denoised_waveform = torch.istft(
    torch.from_numpy(denoised_stft),
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    length=waveform.shape[1],
    center=True,
    normalized=False,
    onesided=True
)

# ------------------------------
# Normalize and save
# ------------------------------
denoised_tensor = denoised_waveform.unsqueeze(0) if denoised_waveform.ndim == 1 else denoised_waveform
max_val = torch.max(torch.abs(denoised_tensor))
if max_val > 0.95:
    denoised_tensor *= 0.95 / max_val
    print(f"Normalized audio by factor {0.95/max_val:.3f}")

output_dir = os.path.dirname(noisy_wav_path)
output_path = os.path.join(output_dir, "corrected_denoised.wav")
torchaudio.save(output_path, denoised_tensor, sr)
print(f" Corrected denoised file saved at: {output_path}")

# Save original for comparison
orig_path = os.path.join(output_dir, "original_mono_16k.wav")
torchaudio.save(orig_path, waveform, sr)
print(f" Original mono file saved at: {orig_path}")

# ------------------------------
# Quality metrics
# ------------------------------
print(f"\n=== QUALITY METRICS ===")
print(f"Original RMS: {torch.sqrt(torch.mean(waveform**2)):.4f}")
print(f"Denoised RMS: {torch.sqrt(torch.mean(denoised_tensor**2)):.4f}")
snr_improvement = 20 * torch.log10(torch.std(denoised_tensor) / torch.std(waveform - denoised_tensor))
print(f"SNR improvement estimate: {snr_improvement:.2f} dB")
