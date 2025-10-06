import torch
import numpy as np
from pathlib import Path
from gtcrn_adapter import GTCRNAdapter, CompressedLoss, prepare_training_pair

# Create dummy data directory
test_data_dir = Path('training_data/dsp_clean_pairs_test')
test_data_dir.mkdir(parents=True, exist_ok=True)

def generate_dummy_spectrograms(num_pairs=5):
    """Generate dummy spectrograms to simulate DSP output"""
    for i in range(num_pairs):
        # Create dummy spectrograms (257 freq bins x 100 time frames)
        time_frames = 100
        freq_bins = 257
        
        # Generate some patterns in the spectrograms
        t = np.linspace(0, 10, time_frames)
        f = np.linspace(0, 1, freq_bins)
        T, F = np.meshgrid(t, f)
        
        # Clean speech: simulate harmonic structure
        clean = 0.5 * np.sin(2 * np.pi * 5 * T) * np.exp(-F)
        
        # Noisy: add some noise
        noise = 0.2 * np.random.randn(freq_bins, time_frames)
        noisy = clean + noise
        
        # DSP enhanced: somewhere between clean and noisy
        dsp = 0.7 * clean + 0.3 * noise
        
        # Save as numpy files
        np.save(test_data_dir / f"pair_{i:03d}_clean_mag.npy", clean.astype(np.float32))
        np.save(test_data_dir / f"pair_{i:03d}_noisy_mag.npy", noisy.astype(np.float32))
        np.save(test_data_dir / f"pair_{i:03d}_dsp.npy", dsp.astype(np.float32))

def test_gtcrn():
    """Test GTCRN model with dummy data"""
    print("Testing GTCRN implementation...")
    
    # Generate test data
    print("Generating dummy spectrograms...")
    generate_dummy_spectrograms(num_pairs=5)
    
    # Load one pair for testing
    dsp = np.load(test_data_dir / "pair_000_dsp.npy")
    noisy = np.load(test_data_dir / "pair_000_noisy_mag.npy")
    clean = np.load(test_data_dir / "pair_000_clean_mag.npy")
    
    # Prepare data
    print("\nPreparing training pair...")
    dsp_input, noisy_input, clean_target = prepare_training_pair(dsp, noisy, clean)
    
    # Add batch dimension
    dsp_input = dsp_input.unsqueeze(0)      # (1, T, F)
    noisy_input = noisy_input.unsqueeze(0)  # (1, T, F)
    clean_target = clean_target.unsqueeze(0) # (1, T, F)
    
    print(f"Input shapes:")
    print(f"DSP: {dsp_input.shape}")
    print(f"Noisy: {noisy_input.shape}")
    print(f"Clean: {clean_target.shape}")
    
    # Create model
    print("\nInitializing model...")
    model = GTCRNAdapter(n_fft=256, hop_length=128, hidden_channels=8)
    
    # Create loss function
    criterion = CompressedLoss(alpha=0.3, residual_weight=0.1)
    
    # Test forward pass
    print("Testing forward pass...")
    refined, residual = model(dsp_input, noisy_input)
    print(f"Output shapes:")
    print(f"Refined: {refined.shape}")
    print(f"Residual: {residual.shape}")
    
    # Test loss computation
    loss, mag_loss, res_loss = criterion(refined, clean_target, residual)
    print(f"\nLoss values:")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Magnitude loss: {mag_loss.item():.4f}")
    print(f"Residual loss: {res_loss.item():.4f}")
    
    # Test backprop
    print("\nTesting backpropagation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\nTest complete! Everything seems to be working.")
    print(f"Test data saved in: {test_data_dir}")

if __name__ == "__main__":
    test_gtcrn()