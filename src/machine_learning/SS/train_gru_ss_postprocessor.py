import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import sys

# Add repo root to path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root))

from models.spectral.gru_postprocessor import GRUPostProcessor
from src.utils.checkpoint import save_checkpoint, load_checkpoint, count_parameters
from src.utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics

# Load training data
def load_training_data(data_dir):
    dsp_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_dsp.npy')])
    clean_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_clean.npy')])
    inputs, targets = [], []
    for dsp_file, clean_file in zip(dsp_files, clean_files):
        dsp = np.load(os.path.join(data_dir, dsp_file)).T  # shape: (time, bands)
        clean = np.load(os.path.join(data_dir, clean_file)).T
        inputs.append(dsp)
        targets.append(clean)
    inputs = torch.FloatTensor(np.concatenate(inputs, axis=0)).unsqueeze(0)  # (1, time, bands)
    targets = torch.FloatTensor(np.concatenate(targets, axis=0)).unsqueeze(0)
    return TensorDataset(inputs, targets)

# Configuration
data_dir = "training_data/dsp_clean_pairs"
checkpoint_dir = "experiments/EXP2/SS/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

input_size = 257
hidden_size = 64
num_layers = 1
batch_size = 1
num_epochs = 20
learning_rate = 0.001
resume = False
checkpoint_path = os.path.join(checkpoint_dir, "gru_postprocessor.pth")
metrics_csv_path = os.path.join(checkpoint_dir, "evaluation_metrics.csv")

# Initialize model, optimizer, loss
model = GRUPostProcessor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Resume training if needed
start_epoch = 0
if resume and os.path.exists(checkpoint_path):
    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

# Load data
dataset = load_training_data(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Prepare CSV for metrics
metrics_log = []

# Training loop
model.train()
for epoch in range(start_epoch, num_epochs):
    running_loss = 0.0
    for batch in dataloader:
        dsp_input, clean_target = batch
        output = model(dsp_input)
        loss = criterion(output, clean_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    save_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_path)

    # Evaluation on small test set (first batch)
    with torch.no_grad():
        test_input, test_target = dataset[0]
        test_input = test_input.unsqueeze(0)
        test_target = test_target.unsqueeze(0)
        test_output = model(test_input)

        # Compute metrics
        metrics = compute_and_save_speech_metrics(
            clean_tensor=test_target.squeeze(0),
            enhanced_tensor=test_output.squeeze(0),
            fs=16000,
            clean_name="test_clean",
            enhanced_name=f"epoch_{epoch+1}_output",
            csv_dir=None,
            csv_filename=None
        )

        print(f"  ✓ PESQ: {metrics['PESQ']:.3f} | STOI: {metrics['STOI']:.3f} | "
              f"SI-SDR: {metrics['SI_SDR']:.2f} dB | DNSMOS: {metrics['DNSMOS_mos_ovr']:.3f}")

        metrics_row = {
            'Epoch': epoch+1,
            'Loss': avg_loss,
            'PESQ': metrics['PESQ'],
            'STOI': metrics['STOI'],
            'SI_SDR': metrics['SI_SDR'],
            'DNSMOS_mos_ovr': metrics['DNSMOS_mos_ovr'],
            'DNSMOS_mos_sig': metrics['DNSMOS_mos_sig'],
            'DNSMOS_mos_bak': metrics['DNSMOS_mos_bak']
        }
        metrics_log.append(metrics_row)

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_log)
metrics_df.to_csv(metrics_csv_path, index=False)

# Report model size
param_count = count_parameters(model)
print(f"Model has {param_count} trainable parameters.")
print(f"Evaluation metrics saved to: {metrics_csv_path}")