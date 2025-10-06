import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import sys

# Import your modules
from gtcrn_adapter import GTCRNAdapter, CompressedLoss, prepare_training_pair


class DSPDataset(Dataset):
    """
    Loads pre-computed DSP outputs.
    """
    
    def __init__(self, data_dir, dsp_algorithm='mband'):
        self.data_dir = Path(data_dir)
        self.dsp_algorithm = dsp_algorithm
        
        # Find files for specific DSP algorithm
        if dsp_algorithm == 'generic':
            # Load all DSP outputs
            self.dsp_files = sorted(self.data_dir.glob("*_dsp_enhanced.npy"))
        else:
            # Load specific DSP (e.g., only mband)
            self.dsp_files = sorted(self.data_dir.glob(f"*_{dsp_algorithm}_*.npy"))
        
        self.noisy_files = sorted(self.data_dir.glob("*_noisy_mag.npy"))
        self.clean_files = sorted(self.data_dir.glob("*_clean_mag.npy"))
        
        max_pairs = 20
        self.dsp_files = self.dsp_files[:max_pairs]
        self.noisy_files = self.noisy_files[:max_pairs]
        self.clean_files = self.clean_files[:max_pairs]
        
        print(f"Loaded {len(self.dsp_files)} pairs for {dsp_algorithm}")
    
    def __len__(self):
        return len(self.dsp_files)
    
    def __getitem__(self, idx):
        # Load spectrograms (257 freq bins, variable time)
        dsp = np.load(self.dsp_files[idx])      # (257, T)
        noisy = np.load(self.noisy_files[idx])  # (257, T)
        clean = np.load(self.clean_files[idx])  # (257, T)
        
        # Convert to GTCRN format (129 freq bins, same time)
        dsp_down, noisy_down, clean_down = prepare_training_pair(
            dsp, noisy, clean
        )
        
        # Normalize by clean max for stable training
        max_val = clean_down.max() + 1e-8
        dsp_down = dsp_down / max_val
        noisy_down = noisy_down / max_val
        clean_down = clean_down / max_val
        
        return dsp_down, noisy_down, clean_down


def train_model(
    data_dir="training_data/dsp_clean_pairs_v2",
    dsp_algorithm='mband',  # or 'wiener' 
    num_epochs=10,
    batch_size=2,
    learning_rate=1e-3,
    device='cpu'  # Start with CPU for debugging
):
    """
    Training loop - optimized for quick feedback.
    """
    
    print("="*70)
    print(f"TRAINING: GTCRN Post-Processor for {dsp_algorithm.upper()}")
    print("="*70)
    
    # Setup
    checkpoint_dir = Path(f"experiments/gtcrn_mvp/{dsp_algorithm}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    print(f"\nUsing device: {device}")
    
    # Load data
    dataset = DSPDataset(data_dir, dsp_algorithm=dsp_algorithm)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Model
    model = GTCRNAdapter(
        n_fft=256,
        hop_length=128,
        hidden_channels=8  # Small for MVP speed
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Loss & optimizer
    criterion = CompressedLoss(alpha=0.3, residual_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best model
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*70}")
    print("TRAINING START")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # ========== TRAIN ==========
        model.train()
        epoch_loss = 0
        epoch_mag_loss = 0
        epoch_res_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for dsp, noisy, clean in pbar:
                dsp = dsp.to(device)
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                # Forward
                refined, residual = model(dsp, noisy)
                loss, mag_loss, res_loss = criterion(refined, clean, residual)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Track
                epoch_loss += loss.item()
                epoch_mag_loss += mag_loss.item()
                epoch_res_loss += res_loss.item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mag': f"{mag_loss.item():.4f}",
                    'res': f"{res_loss.item():.4f}"
                })
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ========== VALIDATE ==========
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for dsp, noisy, clean in val_loader:
                dsp = dsp.to(device)
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                refined, residual = model(dsp, noisy)
                loss, _, _ = criterion(refined, clean, residual)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)
        
        # ========== LOGGING ==========
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed/60:.1f}m elapsed)")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Mag: {epoch_mag_loss/len(train_loader):.4f}, "
              f"Res: {epoch_res_loss/len(train_loader):.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'dsp_algorithm': dsp_algorithm
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    
    # ========== SUMMARY ==========
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {total_time/60:.1f} minutes ({total_time/num_epochs:.0f}s/epoch)")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    
    return model, train_losses, val_losses


def test_inference_speed(model, device='cpu'):
    """
    Test real-time capability.
    Target: < 10ms per frame for hearing aid use.
    """
    print(f"\n{'='*70}")
    print("INFERENCE SPEED TEST")
    print(f"{'='*70}")
    
    model.eval()
    model.to(device)
    
    # Test with 1 second of audio
    # 16kHz, 16ms frames, 8ms hop = 125 frames/sec
    batch = 1
    frames = 125
    freq = 129
    
    dsp_test = torch.randn(batch, frames, freq).to(device)
    noisy_test = torch.randn(batch, frames, freq).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(dsp_test, noisy_test)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    n_runs = 100
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dsp_test, noisy_test)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.time() - start
    
    ms_per_frame = (elapsed / n_runs) * 1000 / frames
    rtf = (elapsed / n_runs) / 1.0  # Real-time factor (1.0sec audio)
    
    print(f"Time per frame: {ms_per_frame:.2f} ms")
    print(f"Real-time factor: {rtf:.3f}x")
    
    if rtf < 0.1:  # 10x faster than real-time
        print("✓ Fast enough for real-time hearing aid use")
    elif rtf < 1.0:
        print("✓ Real-time capable, but tight margins")
    else:
        print("⚠ Too slow for real-time, need optimization")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='training_data/dsp_clean_pairs_v2')
    parser.add_argument('--dsp', default='mband', 
                       choices=['mband', 'wiener', 'generic'],
                       help='DSP algorithm to train for')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Train
    model, train_losses, val_losses = train_model(
        data_dir=args.data_dir,
        dsp_algorithm=args.dsp,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Test speed
    test_inference_speed(model, device=args.device)
    