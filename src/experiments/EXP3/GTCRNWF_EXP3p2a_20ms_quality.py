"""
Experiment GTCRNWF_EXP3p2a_20ms_quality: GTCRN with Wiener Filter Post-processing (20ms frames, quality-optimized)

This experiment implements a hybrid speech enhancement pipeline that combines deep learning
with traditional signal processing. The GTCRN model is used first to remove noise, followed
by Wiener filter post-processing optimized for quality (mu=0.980).

Processing pipeline:
1. GTCRN model - removes initial noise and distortions
2. Wiener filter (20ms frames, mu=0.980) - further refines the enhanced speech from GTCRN

Key features:
- Two-stage processing: GTCRN -> Wiener filter.
- GTCRN removes the majority noise.
- Wiener filter handles residual noise and fine details.
- 20ms frame duration for Wiener filter post-processing.
- mu=0.980 optimized for quality.

Models used:
- GTCRN: pretrained on DNS3 dataset
- Wiener filter: causal implementation

Purpose: Evaluate the effectiveness of using Wiener filter as a post-processor to improve
the performance of the GTCRN model by providing additional refinement and reducing
artifacts that could affect the neural network's output. This version optimizes for quality.

Datasets used:
- EARS dataset for clean speech
- NOIZEUS dataset for noise

SNR levels tested: -5, 0, 5, 10, 15 dB

Metrics computed: PESQ, STOI, SI-SDR, DNSMOS
"""

import os
import torch
import soundfile as sf
import pandas as pd
import torchaudio
from pathlib import Path
import sys
import numpy as np
import random
import torch

#set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.wiener_as import wiener_filter
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# Load GTCRN model
device = torch.device("cpu")
from deep_learning.gtcrn import GTCRN
gtcrn_model = GTCRN().eval()
ckpt_path = repo_root / "src" / "deep_learning" / "gtcrn" / "gtcrn_main" / "checkpoints" / "model_trained_on_dns3.tar"
ckpt = torch.load(ckpt_path, map_location=device)
gtcrn_model.load_state_dict(ckpt['model'])

output_dir = repo_root / 'sound_data' / 'processed' / 'gtcrn_processed_outputs' / 'GTCRNWF_EXP3p2a_20ms_quality_output' 
results_dir = repo_root / 'results' / 'EXP3' / 'GTCRN' / 'GTCRNWF_EXP3p2a_20ms_quality'

# Load test datasets
print("Loading EARS test dataset...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS files for test mode")

print("Loading NOIZEUS test dataset...")
noizeus_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noizeus_files)} NOIZEUS files for test mode")

# Create audio pairs
paired_files = create_audio_pairs(noizeus_files, ears_files)
print(f"Created {len(paired_files)} audio pairs for processing")

snr_dB_range = [-5, 0, 5, 10, 15]

for snr_dB in snr_dB_range:

    print(f"Processing SNR: {snr_dB} dB")

    output_dir_snr = output_dir / f"{snr_dB}dB"
    output_dir_snr.mkdir(parents=True, exist_ok=True)

    results_dir_snr = results_dir / f"{snr_dB}dB"
    results_dir_snr.mkdir(parents=True, exist_ok=True)

    for noise_path, clean_path in paired_files:

        participant = clean_path.parent.name
        print(f"Noise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant}")

        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio( #Returns Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] where clean sr = 16000Hz
            clean_speech=clean_path, 
            noisy_audio=noise_path, 
            snr_db=snr_dB
        )

        # clean_waveform shape: torch.Size([1, 306725])
        # noise_waveform shape: torch.Size([1, 488800])
        # noisy_speech shape: torch.Size([306725])

        clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
        noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
        output_filename = f"GTCRNWF_20ms_quality_{clean_filename}_{noise_filename}_SNR[{snr_dB}]dB.wav"

        # GTCRN inference
        input_stft = torch.stft(noisy_speech, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
        input_stft = torch.view_as_real(input_stft)  # Convert to (F, T, 2)
        with torch.no_grad():
            output_stft = gtcrn_model(input_stft[None])[0]  # Add batch dimension: (1, F, T, 2)
        output_stft = torch.complex(output_stft[..., 0], output_stft[..., 1])  # Convert back to complex
        gtcrn_enhanced_speech = torch.istft(output_stft, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()

        wf_enhanced_speech, enhanced_fs = wiener_filter(
                noisy_audio=torch.from_numpy(gtcrn_enhanced_speech),
                fs=clean_sr,
                frame_dur_ms=20,
                mu=0.980,
                a_dd=0.98,
                eta=0.15,
                #output_dir=output_dir_snr,
                #output_file=output_filename.replace('.wav', ''),
                input_name=clean_filename,
            )

        enhanced_speech = wf_enhanced_speech

        # Step 4: Compute and save metrics
        print("\n4. Computing speech enhancement metrics...")
        enhanced_tensor = enhanced_speech.unsqueeze(0)
        metrics = compute_and_save_speech_metrics(
            clean_tensor=clean_waveform,
            enhanced_tensor=enhanced_tensor,
            fs=enhanced_fs,
            clean_name=clean_filename,
            enhanced_name=output_filename,
            csv_dir=str(results_dir_snr),
            csv_filename='GTCRNWF_EXP3p2a_20ms_quality_data'
        )
        
        # Print summary
        print(f"\n{'='*100}")
        print(f"Completed Noise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant} | SNR: {snr_dB} dB")
        print(f"Enhanced audio saved to: {output_dir}")
        print(f"Results saved to: {results_dir_snr}")
        print(f"{'='*100}")
        print(f"Metrics:")
        
        # Handle potential NaN values in output
        import math
        pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN (No utterances detected)"
        stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
        si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
        dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"
        
        print(f"  PESQ: {pesq_str}")
        print(f"  STOI: {stoi_str}")
        print(f"  SI-SDR: {si_sdr_str}")
        print(f"  DNSMOS Overall: {dnsmos_str}")
        print(f"{'='*100}\n")

    merged_path = merge_csvs(
        input_dir=results_dir_snr,
        output_dir=results_dir,
        output_filename=f'GTCRNWF_EXP3p2a_20ms_quality_merged_[{snr_dB}]dB.csv',
        keep_source=True
    )

    delete_csvs(input_directory=results_dir_snr)