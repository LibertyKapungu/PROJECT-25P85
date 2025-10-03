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
# noisy_wav_path = (
#     src_dir / "sound_data" / "processed" / "wiener_processed_outputs" /
#     "EXP1_output" / "wiener_filter_priori_P107_S23_F10_SNR100_wiener_as__FRAME10ms_"
#     "MU0p980_ADD0p950_ETA0p150_NONCAUSAL.wav"
# )

noisy_wav_path = (
    src_dir / "sound_data" / "processed" / "spectral_processed_outputs" /
    "EXP1_output" / "python_mband_sp21_station_sn0.wav"
)

# ------------------------------
# Debug: Check model I/O specs
# ------------------------------
session = ort.InferenceSession(pytorch_pretrained_models_dir, providers=["CPUExecutionProvider"])

print("=== MODEL DEBUG INFO ===")
for i, input_meta in enumerate(session.get_inputs()):
    print(f"Input {i}: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")

for i, output_meta in enumerate(session.get_outputs()):
    print(f"Output {i}: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")

input_name = session.get_inputs()[0].name

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
# Save as pytorch model for future use
torch_model_path = pytorch_pretrained_models_dir.with_suffix('.pt')
torch.save(torch_model.state_dict(), torch_model_path)
print(f"PyTorch model saved at: {torch_model_path}")


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

# print("Testing ONNX model directly...")
# # Test with a single frame first
test_frame = model_input[:, :, 0:1]  # [1, 257, 1]
test_frame_input = test_frame.astype(np.float32)

# print(f"Test frame shape: {test_frame_input.shape}")
# print(f"Test frame range: [{test_frame_input.min():.6f}, {test_frame_input.max():.6f}]")

# # Run ONNX model directly
# onnx_result = session.run(None, {input_name: test_frame_input})[0]
# print(f"ONNX result shape: {onnx_result.shape}")
# print(f"ONNX result range: [{onnx_result.min():.6f}, {onnx_result.max():.6f}]")

# # If ONNX result is all 1.0, the model itself has issues
# if np.allclose(onnx_result, 1.0):
#     print("⚠️ ONNX model is outputting all 1.0 - model may not be trained or has issues")
# else:
#     print("✅ ONNX model is working - proceeding with PyTorch conversion")

print("Testing ONNX model directly...")
# Initialize hidden states for stateful LSTM
batch_size = 1
# From debug info: hidden states are [1, 1, 256]
hidden_size = 256

# Initialize all hidden states to zero with correct shape [1, 1, 256]
initial_states = {
    input_name: test_frame_input,
    'rnn1_h_state_in': np.zeros((1, 1, 256), dtype=np.float32),
    'rnn1_c_state_in': np.zeros((1, 1, 256), dtype=np.float32),
    'rnn2_h_state_in': np.zeros((1, 1, 256), dtype=np.float32),
    'rnn2_c_state_in': np.zeros((1, 1, 256), dtype=np.float32)
}

print(f"Test frame shape: {test_frame_input.shape}")
print(f"Test frame range: [{test_frame_input.min():.6f}, {test_frame_input.max():.6f}]")
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
# The PyTorch converted model should handle states internally
# But since conversion might not work properly for stateful models, 
# let's process with ONNX directly

print("Using ONNX model directly for stateful processing...")
gain_masks = []

# Initialize states with correct shape [1, 1, 256]
h1_state = np.zeros((1, 1, 256), dtype=np.float32)
c1_state = np.zeros((1, 1, 256), dtype=np.float32)
h2_state = np.zeros((1, 1, 256), dtype=np.float32)
c2_state = np.zeros((1, 1, 256), dtype=np.float32)

for i in range(model_input.shape[2]):
    frame = model_input[:, :, i:i+1]  # [1, 257, 1]
    frame_input = frame.astype(np.float32)
    
    # Prepare inputs with current states
    inputs = {
        input_name: frame_input,
        'rnn1_h_state_in': h1_state,
        'rnn1_c_state_in': c1_state,
        'rnn2_h_state_in': h2_state,
        'rnn2_c_state_in': c2_state
    }
    
    # Run model
    outputs = session.run(None, inputs)
    
    # Extract outputs (gain mask + updated states)
    gain_mask = outputs[0]  # [1, 257, 1]
    h1_state = outputs[1]   # Updated hidden state 1
    c1_state = outputs[2]   # Updated cell state 1  
    h2_state = outputs[3]   # Updated hidden state 2
    c2_state = outputs[4]   # Updated cell state 2
    
    if i == 0:
        print(f"Gain mask shape: {gain_mask.shape}")
        print(f"Gain mask range: [{gain_mask.min():.6f}, {gain_mask.max():.6f}]")
    
    # Store gain mask
    gain_masks.append(gain_mask.squeeze(-1))  # [1, 257]

gain_masks = np.stack(gain_masks, axis=2)  # [1, 257, frames]
print(f"All gain masks shape: {gain_masks.shape}")
print(f"Gain masks range: [{gain_masks.min():.6f}, {gain_masks.max():.6f}]")

# print("Processing frames with PyTorch model...")
# gain_masks = []

# for i in range(model_input.shape[2]):
#     frame = model_input[:, :, i]  # [1, 257]
#     frame_input = torch.tensor(frame[:, :, np.newaxis], dtype=torch.float32)  # [1, 257, 1]

#     with torch.no_grad():
#         gain_mask = torch_model(frame_input)
#             # If gain_mask is a list, extract the first item
#         if isinstance(gain_mask, list):
#             gain_mask = gain_mask[0]

#     if i == 0:
#         print(f"Gain mask shape: {gain_mask.shape}")
#         print(f"Gain mask range: [{gain_mask.min():.6f}, {gain_mask.max():.6f}]")

#     gain_mask = gain_mask.squeeze(-1).numpy()  # [1, 257]
#     gain_masks.append(gain_mask)
        
# gain_masks = np.stack(gain_masks, axis=0)  # [frames, 1, 257]
# gain_masks = np.transpose(gain_masks, (1, 2, 0))  # [1, 257, frames]



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