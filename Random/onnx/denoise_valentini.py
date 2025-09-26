import torch
import torch.nn as nn
import torchaudio
import onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import torchmetrics

# ... (Optional imports and the TinyDenoiser class definition are unchanged) ...
# Handle optional imports for PESQ and STOI
try:
    from pesq import pesq
    from stoi import stoi
    PESQ_STOI_AVAILABLE = True
except ImportError:
    PESQ_STOI_AVAILABLE = False
    print("Warning: 'pesq' and 'pystoi' not found. Run 'pip install pesq pystoi' for advanced metrics.")

class TinyDenoiser(nn.Module):
    def __init__(self, input_size=257, hidden_size=256):
        super().__init__()
        self.fc_in = nn.Linear(input_size, input_size)
        self.bn_in = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_out1 = nn.Linear(hidden_size, input_size)
        self.bn_out = nn.BatchNorm1d(input_size)
        self.fc_out2 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_in(x)
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc_out1(x)
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.fc_out2(x)
        x = self.sigmoid(x)
        return x

# ------------------------------
# 2. CONFIGURE YOUR PATHS HERE
# ------------------------------
onnx_model_path = Path(r"C:\Users\E7440\Documents\Uni2025\Investigation\PROJECT-25P85\models\pretrained\ONNX\denoiser_LSTM_Valetini.onnx")
noisy_wav_path = Path(r"C:\Users\E7440\Documents\Uni2025\Investigation\PROJECT-25P85\sound_data\processed\spectral_processed_outputs\EXP1_output\python_mband_sp21_station_sn0.wav")
clean_wav_path = Path(r"C:\Users\E7440\Documents\Uni2025\Investigation\PROJECT-25P85\Random\Matlab2025Files\SS\validation_dataset\clean_speech\S_56_02.wav")

# ... (Model creation and weight loading are unchanged) ...
# ------------------------------
# 3. Manually Create PyTorch Model and Load Weights
# ------------------------------
print("Manually creating PyTorch model with correct architecture and loading weights...")
torch_model = TinyDenoiser()
torch_model.to(torch.float64)

onnx_model = onnx.load(str(onnx_model_path))
weights = {w.name: torch.from_numpy(numpy_helper.to_array(w).copy()).to(torch.float64) for w in onnx_model.graph.initializer}

try:
    print("Assigning weights to PyTorch model...")
    # Input Stage
    torch_model.fc_in.weight.data = weights['fc0.weight'].squeeze().T
    torch_model.fc_in.bias.data = weights['fc0.bias']
    torch_model.bn_in.weight.data = weights['norm0.weight']
    torch_model.bn_in.bias.data = weights['norm0.bias']
    torch_model.bn_in.running_mean.data = weights['norm0.running_mean']
    torch_model.bn_in.running_var.data = weights['norm0.running_var']

    # LSTMs
    torch_model.lstm1.weight_ih_l0.data = weights['enhance.weight_ih_l0']
    torch_model.lstm1.weight_hh_l0.data = weights['enhance.weight_hh_l0']
    torch_model.lstm1.bias_ih_l0.data = weights['enhance.bias_ih_l0']
    torch_model.lstm1.bias_hh_l0.data = weights['enhance.bias_hh_l0']
    torch_model.lstm2.weight_ih_l0.data = weights['enhance.weight_ih_l1']
    torch_model.lstm2.weight_hh_l0.data = weights['enhance.weight_hh_l1']
    torch_model.lstm2.bias_ih_l0.data = weights['enhance.bias_ih_l1']
    torch_model.lstm2.bias_hh_l0.data = weights['enhance.bias_hh_l1']

    # Output Stage
    torch_model.fc_out1.weight.data = weights['fc1.weight'].squeeze()[:,:256]
    torch_model.fc_out1.bias.data = weights['fc1.bias']
    torch_model.bn_out.weight.data = weights['norm1.weight']
    torch_model.bn_out.bias.data = weights['norm1.bias']
    torch_model.bn_out.running_mean.data = weights['norm1.running_mean']
    torch_model.bn_out.running_var.data = weights['norm1.running_var']
    
    torch_model.fc_out2.weight.data = weights['fc2.weight'].squeeze().T
    torch_model.fc_out2.bias.data = weights['fc2.bias']

    torch_model.eval()
    print("✅ Weights loaded successfully into custom PyTorch model.")
except Exception as e:
    print(f"❌ An unexpected error occurred during weight loading: {e}")
    exit()
# ------------------------------
# 4. Load and Preprocess Audio
# ------------------------------
waveform, sr = torchaudio.load(noisy_wav_path)
if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sr = 16000
print(f"Preprocessed noisy audio: shape={waveform.shape}, sr={sr}")

clean_waveform, clean_sr = torchaudio.load(clean_wav_path)
if clean_waveform.shape[0] > 1: clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
if clean_sr != 16000:
    clean_waveform = torchaudio.transforms.Resample(clean_sr, 16000)(clean_waveform)
print(f"Preprocessed clean audio: shape={clean_waveform.shape}, sr={sr}")

# ------------------------------
# 5. Perform STFT
# ------------------------------
win_length, hop_length, n_fft = 400, 100, 512
stft = torch.stft(input=waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length), return_complex=True, center=True)
magnitude, phase = torch.abs(stft), torch.angle(stft)

# --- NEW: Normalize input magnitude as per your suggestion ---
magnitude_normalized = magnitude / (magnitude.max() + 1e-8)
print(f"Normalized input range: [{magnitude_normalized.min():.6f}, {magnitude_normalized.max():.6f}]")

model_input = torch.nan_to_num(magnitude_normalized).to(torch.float64)

# ------------------------------
# 6. Run Inference
# ------------------------------
print("Running denoising model...")
with torch.no_grad():
    magnitude_transposed = model_input.permute(0, 2, 1)
    gain_masks_transposed = torch_model(magnitude_transposed)
    gain_masks = gain_masks_transposed.permute(0, 2, 1)
print("✅ Inference complete.")

# ------------------------------
# 7. Apply Mask and Reconstruct Audio
# ------------------------------
denoised_magnitude = magnitude.to(torch.float64) * gain_masks # Apply mask to original magnitude
denoised_stft_complex = torch.polar(denoised_magnitude.to(torch.float32), phase)
denoised_waveform = torch.istft(denoised_stft_complex, n_fft, hop_length, win_length, torch.hann_window(win_length), length=waveform.shape[1])

# ------------------------------
# 8. Post-processing and Save
# ------------------------------
max_val = torch.max(torch.abs(denoised_waveform))
if max_val > 0.95: denoised_waveform *= (0.95 / max_val)
output_path = noisy_wav_path.parent / f"denoised_normalized_input_{noisy_wav_path.name}"
torchaudio.save(output_path, denoised_waveform.to(torch.float32), sr)
print(f"🎉 Denoised file saved at: {output_path}")

# ------------------------------
# 9. UPGRADED: Comprehensive Metrics
# ------------------------------
print("\n--- 📈 Comprehensive Quality Metrics ---")

min_len = min(waveform.shape[1], denoised_waveform.shape[1], clean_waveform.shape[1])
noisy_comp = waveform[:, :min_len]
denoised_comp = denoised_waveform[:, :min_len]
clean_comp = clean_waveform[:, :min_len]

# SI-SDR (vs Clean) - The most important metric
si_sdr = torchmetrics.functional.scale_invariant_signal_distortion_ratio(preds=denoised_comp, target=clean_comp)
print(f"Scale-Invariant SDR (vs Clean): {si_sdr.item():.2f} dB")

# Gain mask statistics (key indicator of model performance)
print("\n--- Gain Mask Diagnostics ---")
gain_masks_float = gain_masks.to(torch.float32)
print(f"  Mean: {gain_masks_float.mean():.4f}")
print(f"  Std Dev: {gain_masks_float.std():.4f}")
print(f"  Min: {gain_masks_float.min():.4f}")
print(f"  Max: {gain_masks_float.max():.4f}")

# Check if model is actually working
if gain_masks_float.std() < 0.05:
    print("⚠️ Warning: Very low gain mask variance. Model is likely not attenuating noise effectively.")
elif gain_masks_float.mean() > 0.95:
    print("⚠️ Warning: Gain mask mean is close to 1.0. Minimal denoising is being applied.")
else:
    print("✅ Gain masks show reasonable variation. Model is actively processing.")
print("---------------------------")

# ... (Visualization code is unchanged) ...