import sys
from pathlib import Path
import numpy as np
import torchaudio

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.mband import mband  

# Output directory for training pairs
training_data_dir = repo_root / 'training_data' / 'dsp_clean_pairs'
training_data_dir.mkdir(parents=True, exist_ok=True)

# Load datasets 
ears_files = load_ears_dataset(repo_root, mode="train") 
noizeus_files = load_noizeus_dataset(repo_root)
paired_files = create_audio_pairs(noizeus_files, ears_files)

# Limit to manageable size for initial training
SNR_VALUES = [-5,0,5,10,15] 
MAX_PAIRS = 125  # Limit total pairs

pair_count = 0
for snr_dB in SNR_VALUES:
    print(f"\nProcessing SNR: {snr_dB} dB")
    
    for noise_path, clean_path in paired_files:
        if pair_count >= MAX_PAIRS:
            break
            
        # Preprocess using your existing utility
        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
            clean_speech=clean_path,
            noisy_audio=noise_path,
            snr_db=snr_dB
        )
        
        # Save noisy audio temporarily
        temp_noisy = training_data_dir / 'temp_noisy.wav'
        # Add channel dimension if needed (convert 1D to 2D tensor)
        if noisy_speech.dim() == 1:
            noisy_speech = noisy_speech.unsqueeze(0)  # Add channel dimension
        torchaudio.save(temp_noisy, noisy_speech, clean_sr)
        
        # Run DSP with return_spectrograms=True
        # Load the temporary noisy file as a tensor
        noisy_tensor, sample_rate = torchaudio.load(temp_noisy)
        
        dsp_result = mband(
            noisy_tensor,
            sample_rate,
            Nband=4,
            Freq_spacing='linear',
            FRMSZ=20,
            OVLP=50,
            AVRGING=1,
            Noisefr=1,
            FLOOR=0.002,
            VAD=1,
            return_spectrograms=True  # Get spectrograms
        )
        
        # Get clean spectrogram
        import scipy.signal as signal
        fftl = dsp_result['fftl']
        clean_spec = np.abs(signal.stft(
            clean_waveform.numpy()[0],
            fs=clean_sr,
            nperseg=fftl
        )[2])[:fftl // 2 + 1, :]
        
        # Save pair
        base_name = f"snr{snr_dB}_{pair_count:03d}"
        np.save(training_data_dir / f"{base_name}_dsp.npy", dsp_result['enhanced_mag'])
        np.save(training_data_dir / f"{base_name}_clean.npy", clean_spec)
        
        pair_count += 1
        print(f"  Generated pair {pair_count}/{MAX_PAIRS}")
        
        # Cleanup
        try:
            temp_noisy.unlink()
        except FileNotFoundError:
            pass
        
        # Close any open file handles
        import gc
        gc.collect()

print(f"\nGenerated {pair_count} training pairs in {training_data_dir}")