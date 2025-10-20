"""
Experiment GTCRN_SS_EXP3p1b: GTCRN + Spectral Subtraction Enhanced Speech Metrics

This experiment evaluates the combined GTCRN + Spectral Subtraction enhancement by:
1. Creating noisy speech mixtures (clean EARS speech + NOIZEUS noise at various SNR levels)
2. Enhancing the noisy mixtures using GTCRN
3. Post-processing GTCRN output with multi-band spectral subtraction (mband)
4. Computing quality metrics on the final enhanced speech

Processing: Clean speech + noise -> noisy mixture -> GTCRN enhancement -> mband spectral subtraction -> compute quality metrics

Datasets used:
- EARS dataset for clean speech
- NOIZEUS dataset for noise

Model: GTCRN (trained on DNS3)
Post-processor: Multi-band Spectral Subtraction (mband)
SNR levels tested: -5, 0, 5, 10, 15 dB
Metrics computed: PESQ, STOI, SI-SDR, DNSMOS
"""

import pandas as pd
from pathlib import Path
import sys
import numpy as np
import random
import torch
import torchaudio

# Set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Add GTCRN to path
gtcrn_path = repo_root / "src" / "deep_learning" / "gtcrn_model" 
sys.path.insert(0, str(gtcrn_path))

results_dir = repo_root / 'results' / 'EXP3' / 'EXP3p1b_wovad9'

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs
# from dsp_algorithms.mband import mband
from dsp_algorithms.mband_var import mband

# Import GTCRN model
from deep_learning.gtcrn_model.gtcrn import GTCRN

def enhance_with_gtcrn(noisy_waveform, model, device, target_sr=16000):
    """
    Enhance noisy speech using GTCRN model.
    
    Args:
        noisy_waveform: Noisy speech tensor [channels, samples] or [samples]
        model: Loaded GTCRN model
        device: Device model is on
        target_sr: Target sampling rate (GTCRN expects 16kHz)
    
    Returns:
        Enhanced speech tensor with same shape as input
    """
    original_shape = noisy_waveform.shape
    
    # Ensure waveform is on correct device and has correct shape
    if noisy_waveform.dim() == 1:
        mix = noisy_waveform.unsqueeze(0)  # Add channel dimension
    else:
        mix = noisy_waveform
    
    # Take first channel if stereo
    if mix.shape[0] > 1:
        mix = mix[0:1, :]
    
    mix = mix.to(device)
    
    # Convert to numpy for STFT (following original implementation)
    mix_np = mix.squeeze(0).cpu().numpy()
    
    # Compute STFT
    input_stft = torch.stft(
        torch.from_numpy(mix_np),
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=False
    ).to(device)
    
    # GTCRN inference
    with torch.no_grad():
        output = model(input_stft[None])[0]
    
    # Reconstruct complex output
    real = output[..., 0]
    imag = output[..., 1]
    complex_output = torch.complex(real, imag)
    
    # Inverse STFT
    enhanced = torch.istft(
        complex_output,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=False
    )
    
    # Restore original shape
    if len(original_shape) == 1:
        enhanced = enhanced.squeeze()
    else:
        enhanced = enhanced.unsqueeze(0)
    
    return enhanced

# Load GTCRN model
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cpu")
model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])

# Load test datasets
print("\nLoading EARS test dataset...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS files for test mode")

print("Loading NOIZEUS test dataset...")
noizeus_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noizeus_files)} NOIZEUS files for test mode")

# Create audio pairs
paired_files = create_audio_pairs(noizeus_files, ears_files)
print(f"Created {len(paired_files)} audio pairs for processing")

# snr_dB_range = [-5, 0, 5, 10, 15]
snr_dB_range = [5]

for snr_dB in snr_dB_range:

    print(f"\n{'='*100}")
    print(f"Processing SNR: {snr_dB} dB")
    print(f"{'='*100}")

    results_dir_snr = results_dir / f"{snr_dB}dB"
    results_dir_snr.mkdir(parents=True, exist_ok=True)

    for noise_path, clean_path in paired_files:

        participant = clean_path.parent.name
        print(f"\nNoise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant}")

        # Step 1: Create noisy mixture
        print("1. Creating noisy mixture...")
        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
            clean_speech=clean_path, 
            noisy_audio=noise_path, 
            snr_db=snr_dB
        )

        # Step 2: Resample to 16kHz if needed (GTCRN requirement)
        if clean_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)
            clean_waveform_16k = resampler(clean_waveform)
            noisy_speech_16k = resampler(noisy_speech)
            processing_sr = 16000
        else:
            clean_waveform_16k = clean_waveform
            noisy_speech_16k = noisy_speech
            processing_sr = clean_sr

        # Step 3: Enhance with GTCRN
        print("3. Enhancing speech with GTCRN...")
        gtcrn_enhanced = enhance_with_gtcrn(
            noisy_waveform=noisy_speech_16k,
            model=model,
            device=device,
            target_sr=processing_sr
        )

        # Step 4: Post-process with spectral subtraction
        print("4. Applying multi-band spectral subtraction post-processing...")

        final_enhanced_speech, final_fs = mband(
            noisy_audio=gtcrn_enhanced,
            fs=processing_sr,
            Nband=4,
            Freq_spacing='linear',
            FRMSZ=20,
            OVLP=75,
            AVRGING=1,
            Noisefr=1,
            FLOOR=0.8,
            VAD=0,
        )

        # Step 5: Compute and save metrics
        clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
        noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
        
        print("5. Computing speech quality metrics...")
        metrics = compute_and_save_speech_metrics(
            clean_tensor=clean_waveform_16k,
            enhanced_tensor=final_enhanced_speech,
            fs=final_fs,
            clean_name=clean_filename,
            enhanced_name=f"GTCRN+SS_{clean_filename}_{noise_filename}_SNR[{snr_dB}]dB",
            csv_dir=str(results_dir_snr),
            csv_filename='GTCRN_SS_NOIZEUS_EARS_metrics.csv'
        )
        
        # Print summary
        print(f"\n{'='*100}")
        print(f"Completed GTCRN + Spectral Subtraction enhancement for:")
        print(f"  Noise: {noise_path.name}")
        print(f"  EARS: {clean_path.name}")
        print(f"  Participant: {participant}")
        print(f"  SNR: {snr_dB} dB")
        print(f"{'='*100}")
        print(f"Results saved to: {results_dir_snr}")
        
        # Handle potential NaN values in output
        import math
        pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN (No utterances detected)"
        stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
        si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
        dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"
        
        print(f"\nMetrics:")
        print(f"  PESQ: {pesq_str}")
        print(f"  STOI: {stoi_str}")
        print(f"  SI-SDR: {si_sdr_str}")
        print(f"  DNSMOS Overall: {dnsmos_str}")
        print(f"{'='*100}\n")

    # Merge CSVs for this SNR level
    print(f"\nMerging results for SNR {snr_dB} dB...")
    merged_path = merge_csvs(
        input_dir=results_dir_snr,
        output_dir=results_dir,
        output_filename=f'GTCRN_SS_TEST2_[{snr_dB}]dB.csv',
        keep_source=True
    )
    print(f"Merged results saved to: {merged_path}")

    # Clean up individual CSV files
    delete_csvs(input_directory=results_dir_snr)

print(f"\n{'='*100}")
print("EXPERIMENT COMPLETE")
print(f"All results saved to: {results_dir}")
print(f"{'='*100}")