"""
Fine-tune GTCRN as a post-processor for spectral subtraction
Need to fix this. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import numpy as np
from pathlib import Path
from gtcrn import GTCRN
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsp_algorithms.mband import mband

class GTCRNPostProcessor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.gtcrn = GTCRN()
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            self.gtcrn.load_state_dict(ckpt['model'])
            print(f"Loaded pretrained model from {pretrained_path}")
    
    def forward(self, spec):
        return self.gtcrn(spec)

def compute_stft(signal, n_fft=512, hop_length=256, win_length=512):
    """Compute STFT with consistent parameters"""
    window = torch.hann_window(win_length).pow(0.5)
    stft_complex = torch.stft(
        torch.from_numpy(signal.astype(np.float32)),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    return torch.view_as_real(stft_complex)  # Convert to real representation

def compute_istft(spec, n_fft=512, hop_length=256, win_length=512):
    """Compute iSTFT with consistent parameters"""
    window = torch.hann_window(win_length).pow(0.5)
    spec_complex = torch.view_as_complex(spec)
    signal = torch.istft(
        spec_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    return signal

class ComplexMSELoss(nn.Module):
    """Complex-valued MSE loss for spectral learning"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Compute MSE for real and imaginary parts
        real_loss = torch.mean((pred[..., 0] - target[..., 0])**2)
        imag_loss = torch.mean((pred[..., 1] - target[..., 1])**2)
        return real_loss + imag_loss

def train_gtcrn_post_processor(
    clean_files,
    noisy_files,
    model_path,
    output_dir,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Setup model and optimizer
    model = GTCRNPostProcessor(model_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ComplexMSELoss()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(zip(clean_files, noisy_files), total=len(clean_files))
        
        for clean_path, noisy_path in progress_bar:
            # Load audio files
            clean, fs = sf.read(clean_path)
            noisy, _ = sf.read(noisy_path)
            
            # Apply spectral subtraction first
            ss_enhanced = mband(
                noisy,
                fs,
                frame_length=512,
                frame_step=256,
                alpha=2,
                beta=0.01
            )
            
            # Compute STFTs
            clean_spec = compute_stft(clean)
            ss_spec = compute_stft(ss_enhanced)
            
            # Move to device
            clean_spec = clean_spec.to(device)
            ss_spec = ss_spec.to(device)
            
            # Forward pass
            enhanced_spec = model(ss_spec)
            
            # Compute loss
            loss = criterion(enhanced_spec, clean_spec)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(clean_files)
        print(f"\nEpoch {epoch+1}/{num_epochs} Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, output_dir / f'gtcrn_post_processor_epoch_{epoch+1}.pth')

def main():
    # Setup paths
    clean_dir = Path("path/to/clean/speech/files")
    noisy_dir = Path("path/to/noisy/speech/files")
    model_path = Path("checkpoints/model_trained_on_dns3.tar")
    output_dir = Path("checkpoints/fine_tuned")
    
    # Get file lists
    clean_files = sorted(list(clean_dir.glob("*.wav")))
    noisy_files = sorted(list(noisy_dir.glob("*.wav")))
    
    assert len(clean_files) == len(noisy_files), "Number of clean and noisy files must match"
    
    # Train the model
    train_gtcrn_post_processor(
        clean_files=clean_files,
        noisy_files=noisy_files,
        model_path=str(model_path),
        output_dir=str(output_dir),
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4
    )

if __name__ == "__main__":
    main()