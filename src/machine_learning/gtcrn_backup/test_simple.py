import torch
import soundfile as sf
import numpy as np    
from pathlib import Path
from gtcrn import GTCRN

def test_gtcrn_simple():
    """
    Simple test of GTCRN on a noisy file
    """
    # Create directories
    test_dir = Path('test_wavs')
    test_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    # Initialize model with pretrained weights
    device = torch.device("cpu")
    model = GTCRN().eval()
    
    # Check for pretrained weights
    checkpoint_path = Path(r"C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/src/machine_learning/gtcrn/checkpoints/model_trained_on_dns3.tar")
    if not checkpoint_path.exists():
        print("\nWARNING: Pretrained model not found at:", checkpoint_path)
    else:
        print("\nLoading pretrained model from:", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    
    # Load real speech files
    clean_path = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\Random\Matlab2025Files\SS\validation_dataset\clean_speech\S_56_02.wav")
    #noisy_path = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\Random\Matlab2025Files\SS\validation_dataset\noisy_speech\sp21_station_sn5.wav")
    noisy_path = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\Random\Matlab2025Files\SS\mband_processed_output\mband_sp21_station_sn5.wav")
    # Load the audio files
    clean, fs = sf.read(clean_path)
    noisy, fs_noisy = sf.read(noisy_path)
    
    # Save copies to test directory
    sf.write(test_dir / 'clean.wav', clean, fs)
    sf.write(test_dir / 'noisy.wav', noisy, fs)
    
    print("\nProcessing files:")
    print(f"Clean speech: {clean_path.name}")
    print(f"Noisy speech: {noisy_path.name}")
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(clean)/fs:.2f} seconds")
    
    try:
        # GTCRN is designed for these specific parameters
        n_fft = 512  # Must be 512 for GTCRN's ERB filterbank
        hop_length = n_fft // 2
        win_length = n_fft
        window = torch.hann_window(win_length).pow(0.5)
        
        # Compute STFT (first get complex output)
        stft_complex = torch.stft(
            torch.from_numpy(noisy.astype(np.float32)),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        
        # Convert to real representation for model
        input_stft = torch.view_as_real(stft_complex)  # Convert to real format
        
        print("\nSTFT shapes:")
        print("Input shape:", input_stft.shape)
        
        # Process through model
        with torch.no_grad():
            output = model(input_stft[None])[0]  # Add and remove batch dimension
            output = output.contiguous()  # Ensure proper stride for view_as_complex
        
        print("Output shape:", output.shape)
        
        # Convert back to complex
        output_complex = torch.view_as_complex(output)
        
        # Calculate and print statistics
        print("\nSignal statistics:")
        print(f"Clean speech RMS: {np.sqrt(np.mean(clean**2)):.4f}")
        print(f"Noisy speech RMS: {np.sqrt(np.mean(noisy**2)):.4f}")
        print(f"SNR estimate: {10*np.log10(np.sum(clean**2)/(np.sum((noisy-clean)**2) + 1e-10)):.1f} dB")
    
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    
    # Convert back to time domain using complex input
    enhanced = torch.istft(
        output_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False
    )
    
    # Save enhanced and calculate enhancement statistics
    enhanced_np = enhanced.numpy()
    enhanced_np = enhanced_np / np.max(np.abs(enhanced_np))
    
    # Ensure arrays are the same length for comparison
    min_len = min(len(enhanced_np), len(clean))
    enhanced_np = enhanced_np[:min_len]
    clean_trimmed = clean[:min_len]
    
    # Save the trimmed enhanced signal
    sf.write(test_dir / 'enhanced.wav', enhanced_np, fs)
    
    print("\nEnhancement Results:")
    print(f"Enhanced speech RMS: {np.sqrt(np.mean(enhanced_np**2)):.4f}")
    print(f"Enhanced SNR estimate: {10*np.log10(np.sum(clean_trimmed**2)/(np.sum((enhanced_np-clean_trimmed)**2) + 1e-10)):.1f} dB")
    
    print(f"\nFiles saved in {test_dir}:")
    print("- clean.wav: Original clean signal")
    print("- noisy.wav: Signal with added noise")
    print("- enhanced.wav: GTCRN enhanced signal")

if __name__ == "__main__":
    test_gtcrn_simple()