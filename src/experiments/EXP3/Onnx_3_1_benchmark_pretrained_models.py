import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import os
import sys
from pathlib import Path
from onnx2pytorch import ConvertModel
import onnx

# ------------------------------
# Paths
# ------------------------------

current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir/ "src"))

pytorch_pretrained_models_dir = (src_dir / "models" / "pretrained" / "ONNX" / "FullRankLSTM_Stateful.onnx")

# Define noisy wav path (relative to repo root)
noisy_wav_path = (
    src_dir / "sound_data" / "processed" / "wiener_processed_outputs" /
    "EXP1_output" / "wiener_filter_priori_P107_S23_F10_SNR100_wiener_as__FRAME10ms_"
    "MU0p980_ADD0p950_ETA0p150_NONCAUSAL.wav"
)

# ------------------------------
# Debug: Check model I/O specs
# ------------------------------
# session = ort.InferenceSession(pytorch_pretrained_models_dir, providers=["CPUExecutionProvider"])

# print("=== MODEL DEBUG INFO ===")
# for i, input_meta in enumerate(session.get_inputs()):
#     print(f"Input {i}: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")

# for i, output_meta in enumerate(session.get_outputs()):
#     print(f"Output {i}: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")

# input_name = session.get_inputs()[0].name
if not noisy_wav_path.exists():
    raise FileNotFoundError(f"File not found: {noisy_wav_path}")

print(f"Resolved WAV path: {noisy_wav_path.resolve()}")

# ------------------------------
# Convert ONNX model to PyTorch
# ------------------------------
print("Converting ONNX model to PyTorch...")
onnx_model = onnx.load(str(pytorch_pretrained_models_dir))
torch_model = ConvertModel(onnx_model)
torch_model.eval()
print("Conversion complete.")

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
# Model specs: 25ms frame, 6.25ms hop, 16kHz sampling, Hanning window
sr = 16000
frame_duration = 0.025  # 25ms
hop_duration = 0.00625  # 6.25ms (25% of frame)

win_length = int(frame_duration * sr)  # 400 samples
hop_length = int(hop_duration * sr)    # 100 samples
n_fft = 512  # Next power of 2 above win_length for 257 freq bins

print(f"STFT params: win_length={win_length}, hop_length={hop_length}, n_fft={n_fft}")
print(f"Expected freq bins: {n_fft//2 + 1}")

window = torch.hann_window(win_length)

# Add small epsilon to prevent log(0) issues
eps = 1e-8

stft = torch.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    return_complex=True,
    center=True,  # Important for reconstruction
    normalized=False,
    onesided=True
)

print(f"STFT shape: {stft.shape}")

magnitude = torch.abs(stft)
phase = torch.angle(stft)

# ------------------------------
# Input preprocessing - MODEL EXPECTS RAW MAGNITUDE
# ------------------------------
# The model takes STFT magnitude values and outputs gain mask [0,1]
model_input = magnitude.numpy()
print(f"Input format: raw magnitude, shape: {model_input.shape}")
print(f"Input range: [{model_input.min():.6f}, {model_input.max():.6f}]")

# ------------------------------
# Run ONNX model - GAIN MASK PREDICTION
# ------------------------------
# print("Processing frames...")
# gain_masks = []

# for i in range(model_input.shape[2]):
#     frame = model_input[:, :, i]  # [1, 257]
    
#     # Model expects [batch, freq_bins, 1] format
#     frame_input = frame[:, :, np.newaxis].astype(np.float32)  # [1, 257, 1]
    
#     # Debug first frame
#     if i == 0:
#         print(f"Frame input shape: {frame_input.shape}")
#         print(f"Frame input range: [{frame_input.min():.6f}, {frame_input.max():.6f}]")
    
#     try:
#         # Model outputs gain mask [0, 1]
#         gain_mask = session.run(None, {input_name: frame_input})[0]
        
#         if i == 0:
#             print(f"Gain mask shape: {gain_mask.shape}")
#             print(f"Gain mask range: [{gain_mask.min():.6f}, {gain_mask.max():.6f}]")
        
#         # Remove last dimension if present
#         if gain_mask.ndim == 3:
#             gain_mask = gain_mask.squeeze(-1)  # [1, 257]
        
#         gain_masks.append(gain_mask)
        
#     except Exception as e:
#         print(f"Error processing frame {i}: {e}")
#         break

# gain_masks = np.stack(gain_masks, axis=2)  # [1, 257, frames]
# print(f"All gain masks shape: {gain_masks.shape}")
# print(f"Gain masks range: [{gain_masks.min():.6f}, {gain_masks.max():.6f}]")

print("Processing frames with PyTorch model...")
gain_masks = []

for i in range(model_input.shape[2]):
    frame = model_input[:, :, i]  # [1, 257]
    frame_input = torch.tensor(frame[:, :, np.newaxis], dtype=torch.float32)  # [1, 257, 1]

    with torch.no_grad():
        gain_mask = torch_model(frame_input)
            # If gain_mask is a list, extract the first item
        if isinstance(gain_mask, list):
            gain_mask = gain_mask[0]

    if i == 0:
        print(f"Gain mask shape: {gain_mask.shape}")
        print(f"Gain mask range: [{gain_mask.min():.6f}, {gain_mask.max():.6f}]")

    gain_mask = gain_mask.squeeze(-1).numpy()  # [1, 257]
    gain_masks.append(gain_mask)
        
gain_masks = np.stack(gain_masks, axis=0)  # [frames, 1, 257]
gain_masks = np.transpose(gain_masks, (1, 2, 0))  # [1, 257, frames]



# ------------------------------
# Apply gain masks to get denoised magnitude
# ------------------------------
# The model outputs gain masks [0,1], multiply with original magnitude
original_magnitude = magnitude.numpy()
denoised_magnitude = original_magnitude * gain_masks

print(f"Original magnitude range: [{original_magnitude.min():.6f}, {original_magnitude.max():.6f}]")
print(f"Denoised magnitude range: [{denoised_magnitude.min():.6f}, {denoised_magnitude.max():.6f}]")

print(f"Final magnitude range: [{denoised_magnitude.min():.6f}, {denoised_magnitude.max():.6f}]")

# ------------------------------
# Reconstruct audio
# ------------------------------
# Reconstruct complex STFT
denoised_stft = denoised_magnitude * np.exp(1j * phase.numpy())

# ISTFT back to waveform
denoised_waveform = torch.istft(
    torch.from_numpy(denoised_stft),
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    length=waveform.shape[1],  # match original length
    center=True,  # Must match STFT settings
    normalized=False,
    onesided=True
)

# ------------------------------
# Post-processing and normalization
# ------------------------------
# Ensure correct shape
if denoised_waveform.ndim == 1:
    denoised_tensor = denoised_waveform.unsqueeze(0)
elif denoised_waveform.ndim == 2:
    denoised_tensor = denoised_waveform
else:
    raise ValueError(f"Unexpected waveform shape: {denoised_waveform.shape}")

# Normalize to prevent clipping
max_val = torch.max(torch.abs(denoised_tensor))
if max_val > 0.95:
    denoised_tensor = denoised_tensor * 0.95 / max_val
    print(f"Normalized audio by factor {0.95/max_val:.3f}")


# ------------------------------
# Save and compare
# ------------------------------
output_dir = os.path.dirname(noisy_wav_path)
output_path = os.path.join(output_dir, "corrected_denoised.wav")

torchaudio.save(output_path, denoised_tensor, sr)
print(f"✅ Corrected denoised file saved at: {output_path}")

# Save original for comparison
orig_path = os.path.join(output_dir, "original_mono_16k.wav")
torchaudio.save(orig_path, waveform, sr)
print(f"📁 Original mono file saved at: {orig_path}")

# ------------------------------
# Quality metrics
# ------------------------------
print(f"\n=== QUALITY METRICS ===")
print(f"Original RMS: {torch.sqrt(torch.mean(waveform**2)):.4f}")
print(f"Denoised RMS: {torch.sqrt(torch.mean(denoised_tensor**2)):.4f}")
print(f"SNR improvement estimate: {20*torch.log10(torch.std(denoised_tensor)/torch.std(waveform-denoised_tensor)):.2f} dB")