"""
Experiment WF_EXP1p2: Parameter Tuning for Causal Wiener Filter

This experiment performs comprehensive parameter tuning for the causal Wiener filter implementation
across different window sizes and SNR levels. The goal is to identify optimal parameter settings
that provide speech enhancement improvements.

Parameters tuned:
- mu: smoothing factor (0.10 to 0.99)
- a_dd: decision-directed factor (0.10 to 0.99)
- eta: noise overestimation factor (0.05 to 1.00)
- frame_dur_ms: frame duration in milliseconds (8, 20, 25)

Purpose: Tune causal Wiener filter parameters per SNR level to analyze performance improvements
across different window sizes and parameter combinations.

Datasets used:
- EARS dataset for clean speech
- NOIZEUS dataset for noise

SNR levels tested: -5, 0, 5, 10, 15 dB

Metrics computed: PESQ, STOI, SI-SDR, DNSMOS
"""

import pandas as pd
import torchaudio
import torch
from pathlib import Path
import sys
import math
import numpy as np
import random

#set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1p2' 
results_dir = repo_root / 'results' / 'EXP1' / 'wiener' / 'WF_EXP1p2'

from utils.audio_dataset_loader import load_noizeus_dataset, load_ears_dataset, create_audio_pairs, preprocess_audio 
from dsp_algorithms.wiener_as import wiener_filter
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# Load NOIZEUS noise dataset (test mode only)
noise_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noise_files)} NOIZEUS noise files")

# Load EARS clean speech dataset (test mode: participants 92-107)
clean_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(clean_files)} EARS clean speech files for test mode")

# Create pairs of noise and clean speech files
paired_files = create_audio_pairs(noise_files, clean_files)
print(f"Created {len(paired_files)} audio pairs using NOIZEUS dataset")

snr_dB_range = [-5, 0, 5, 10, 15]

mu_default = 0.98
a_dd_default = 0.98
eta_default = 0.15

# mu: from 0.80 → 0.99
mu_values = np.linspace(0.10, 0.99, 40).round(3).tolist()

# a_dd: from 0.80 → 0.99
a_dd_values = np.linspace(0.10, 0.99, 40).round(3).tolist()

# eta: from 0.05 → 1.00
eta_values = np.linspace(0.05, 1.00, 40).round(3).tolist()

# frame_dur_ms: window sizes to test
frame_dur_ms_values = [8, 20, 25]

# Debug: Print default values to verify they're correct
print(f"\n[DEBUG]: Default values - mu_default={mu_default}, a_dd_default={a_dd_default}, eta_default={eta_default}")
print(f"[DEBUG]: Parameter ranges - mu_values={mu_values}")
print(f"[DEBUG]: Parameter ranges - a_dd_values={a_dd_values}")
print(f"[DEBUG]: Parameter ranges - eta_values={eta_values}")

# Define parameter sets to loop through
parameter_sets = [
    ("mu", mu_values, mu_default, a_dd_default, eta_default),
    ("a_dd", a_dd_values, mu_default, a_dd_default, eta_default),
    ("eta", eta_values, mu_default, a_dd_default, eta_default)
]

for frame_dur_ms in frame_dur_ms_values:
    print(f"\n{'='*140}")
    print(f"FRAME DURATION: {frame_dur_ms} ms")
    print(f"{'='*140}")
    
    for param_name, param_values, default_mu, default_a_dd, default_eta in parameter_sets:
        print(f"\n{'='*120}")
        print(f"VARYING PARAMETER: {param_name.upper()}")
        print(f"[DEBUG]: Unpacked defaults - default_mu={default_mu}, default_a_dd={default_a_dd}, default_eta={default_eta}")
        print(f"[DEBUG]: Values to iterate: {param_values}")
        print(f"{'='*120}")
        
        for param_value in param_values:
            print(f"\nDEBUG: Processing param_value = {param_value} (type: {type(param_value)})")
            
            # Set current parameter values - ensure they're float type
            if param_name == "mu":
                current_mu = float(param_value)
                current_a_dd = float(default_a_dd)
                current_eta = float(default_eta)
            elif param_name == "a_dd":
                current_mu = float(default_mu)
                current_a_dd = float(param_value)
                current_eta = float(default_eta)
            else:  # eta
                current_mu = float(default_mu)
                current_a_dd = float(default_a_dd)
                current_eta = float(param_value)
                
            for snr_dB in snr_dB_range:

                print(f"Current parameters: mu={current_mu}, a_dd={current_a_dd}, eta={current_eta}")
                print(f"[DEBUG]: Types - mu: {type(current_mu)}, a_dd: {type(current_a_dd)}, eta: {type(current_eta)}")
                print(f"[DEBUG]: CSV filename will be: WF_EXP1p2_data_SNR[{snr_dB}]dB_mu{current_mu}_a{current_a_dd}_eta{current_eta}_frame{frame_dur_ms}ms")

                print(f"Processing SNR: {snr_dB} dB")

                output_dir_snr = output_dir / f"{snr_dB}dB"
                output_dir_snr.mkdir(parents=True, exist_ok=True)

                results_dir_snr = results_dir / f"{snr_dB}dB" / f"{frame_dur_ms}ms"
                results_dir_snr.mkdir(parents=True, exist_ok=True)

                for noise_path, clean_path in paired_files:

                    participant = clean_path.parent.name
                    print(f"NOIZEUS Noise: {noise_path.name} | EARS Clean: {clean_path.name} | Participant: {participant}")

                    # Import the preprocess_audio function
                    clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                        clean_speech=clean_path, 
                        noisy_audio=noise_path, 
                        snr_db=snr_dB
                    )

                    clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
                    noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
                    output_filename = f"WF_{clean_filename}_{noise_filename}_SNR[{snr_dB}]dB.wav"

                    # Step 2: Apply Wiener filtering (using causal processing)
                    print("\n2. Applying causal Wiener filtering...")
                    enhanced_speech, enhanced_fs = wiener_filter(
                        noisy_audio=noisy_speech,
                        fs=clean_sr,
                        mu=current_mu,
                        a_dd=current_a_dd,
                        eta=current_eta,
                        frame_dur_ms=frame_dur_ms
                    )
                
                    # Step 4: Compute and save metrics
                    print("\n4. Computing speech enhancement metrics...")
                    metrics = compute_and_save_speech_metrics(
                        clean_tensor=clean_waveform,
                        enhanced_tensor=enhanced_speech,
                        fs=enhanced_fs,
                        clean_name=clean_filename,
                        enhanced_name=output_filename,
                        csv_dir=str(results_dir_snr),
                        csv_filename=f'WF_EXP1p2_data_SNR[{snr_dB}]dB_MU[{current_mu}]_AADD[{current_a_dd}]_ETA[{current_eta}]_FRAME[{frame_dur_ms}]ms'
                    )
                    
                    # Print summary
                    print(f"\n{'='*100}")
                    print(f"Completed NOIZEUS: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant} | SNR: {snr_dB} dB")
                    print(f"Current parameters: mu={current_mu}, a_dd={current_a_dd}, eta={current_eta}, frame_dur_ms={frame_dur_ms}")
                    print(f"{'='*100}")
                    print(f"Enhanced audio saved to: {output_dir}")
                    print(f"Results saved to: {results_dir_snr}")
                    print(f"Metrics:")
                    
                    # Handle potential NaN values in output
                    pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN (No utterances detected)"
                    stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
                    si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
                    dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"
                    
                    print(f"  PESQ: {pesq_str}")
                    print(f"  STOI: {stoi_str}")
                    print(f"  SI-SDR: {si_sdr_str}")
                    print(f"  DNSMOS Overall: {dnsmos_str}")
                    print(f"{'='*100}\n")

# Merge CSV files after all parameter variations are complete
print(f"\n{'='*120}")
print("MERGING CSV FILES")
print(f"{'='*120}")

for frame_dur_ms in frame_dur_ms_values:
    for snr_dB in snr_dB_range:
        results_dir_snr = results_dir / f"{snr_dB}dB" / f"{frame_dur_ms}ms"
        
        merged_path = merge_csvs(
            input_dir=results_dir_snr,
            output_dir=results_dir,
            output_filename=f'WF_EXP1p2_merged_[{snr_dB}]dB_[{frame_dur_ms}]ms.csv',
            keep_source=True
        )
        print(f"Merged results for {snr_dB}dB {frame_dur_ms}ms: {merged_path}")

        # Step 5: Delete individual CSV files
        delete_csvs(input_directory=results_dir_snr)
        
        print(f"Deleted individual CSV files in {results_dir_snr}")

print(f"\n{'='*120}")
print("ALL PARAMETER VARIATIONS COMPLETED")
print(f"{'='*120}")