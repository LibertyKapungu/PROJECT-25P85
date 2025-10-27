import os
import pandas as pd
import torchaudio
import numpy as np
from pathlib import Path
import sys
import random
import torch
import time
import math

"""
Experiment EXP1p1b_AVRGING: Python mband Implementation Test
- Compares causal avrging where look at past three frames .
- Compares linear, log, and mel frequency spacings.
- Uses MATLAB default parameters (Nband=6, Noisefr=6, etc.).
- Computes PESQ, STOI, SI-SDR, DNSMOS.
- Saves results to CSV per configuration and SNR.
"""

# Set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

# --- Define Base Directories ---
# Base directory for all results of this Python mband test
results_dir_base = repo_root / 'results' / 'EXP1' / 'spectral' / 'SS_EXP1p1b' / 'AVRGING'/ 'xmagsm_change'
# If want a separate output dir for audio if plan to save it
# output_dir_base = repo_root / 'sound_data' / 'processed' / 'spectral_processed_outputs' / 'EXP1p1a_Python_mband'

# --- Utility Imports ---
from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)

from dsp_algorithms.mband_AVG_causal import mband
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# ============================================================================
# 1. DEFINE EXPERIMENT CONFIGURATIONS
# ============================================================================
EXPERIMENT_CONFIGS = [
    # --- MBAND (AVRGING=1) ---
    {
        'name': 'mband_py_N6_lin_AVR1',
        'params': {'Nband': 6, 'Freq_spacing': 'linear', 'AVRGING': 1, 'Noisefr': 6}
    },
    {
        'name': 'mband_py_N6_log_AVR1',
        'params': {'Nband': 6, 'Freq_spacing': 'log', 'AVRGING': 1, 'Noisefr': 6}
    },
    {
        'name': 'mband_py_N6_mel_AVR1',
        'params': {'Nband': 6, 'Freq_spacing': 'mel', 'AVRGING': 1, 'Noisefr': 6}
    },
    # --- MBAND (Causal, AVRGING=0) ---
    {
        'name': 'mband_py_N6_lin_AVR0_8',
        'params': {'Nband': 6, 'Freq_spacing': 'linear', 'AVRGING': 0, 'Noisefr': 6}
    },
    {
        'name': 'mband_py_N6_log_AVR0_8',
        'params': {'Nband': 6, 'Freq_spacing': 'log', 'AVRGING': 0, 'Noisefr': 6}
    },
    {
        'name': 'mband_py_N6_mel_AVR0_8',
        'params': {'Nband': 6, 'Freq_spacing': 'mel', 'AVRGING': 0, 'Noisefr': 6}
    },
]

# Shared parameters (MATLAB defaults)
SHARED_PARAMS = {
    'FRMSZ': 20,
    'OVLP': 50,
    'FLOOR': 0.002,
    'VAD': 1,
}

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

# ============================================================================
# 2. MAIN EXPERIMENT LOOP
# ============================================================================
start_time_total = time.time()

for config in EXPERIMENT_CONFIGS:
    alg_name = config['name']
    params = {**SHARED_PARAMS, **config['params']} # Combine shared and specific params
    print(f"\n{'='*100}")
    print(f"STARTING CONFIGURATION: {alg_name}")
    print(f"Parameters: {params}")
    print(f"{'='*100}")
    config_start_time = time.time()

    # Define results directory for this specific configuration
    results_dir_config = results_dir_base / alg_name
    results_dir_config.mkdir(parents=True, exist_ok=True)

    for snr_dB in snr_dB_range:
        print(f"\n--- Processing SNR: {snr_dB} dB for {alg_name} ---")
        snr_start_time = time.time()

        # Define results directory for this specific SNR within the config
        results_dir_snr = results_dir_config / f"{snr_dB}dB"
        results_dir_snr.mkdir(parents=True, exist_ok=True)

        file_counter = 0
        for noise_path, clean_path in paired_files:
            file_counter += 1
            participant = clean_path.parent.name
            print(f"\nProcessing file {file_counter}/{len(paired_files)}: Noise={noise_path.name} | Clean={clean_path.name}")

            # Step 1: Create noisy mixture (Resamples to 16kHz by default in preprocess_audio)
            try:
                clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                    clean_speech=clean_path,
                    noisy_audio=noise_path,
                    snr_db=snr_dB,
                    target_sr=16000 # Ensure consistent 16kHz processing
                )
                processing_sr = 16000 # Use 16kHz for processing
            except Exception as e:
                print(f"!!! ERROR during preprocessing: {e}. Skipping this file pair.")
                continue

            clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
            noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
            # Unique name for enhanced file (though not saving it here)
            enhanced_file_description = f"{alg_name}_{clean_filename}_{noise_filename}_SNR[{snr_dB}]dB"

            # Step 2: Apply Python mband filtering
            print(f"2. Applying Python mband ({alg_name})...")
            try:
                # Pass all parameters explicitly
                enhanced_speech, enhanced_fs = mband(
                    noisy_audio=noisy_speech,
                    fs=processing_sr, # Use the consistent processing_sr
                    Nband=params['Nband'],
                    Freq_spacing=params['Freq_spacing'],
                    FRMSZ=params['FRMSZ'],
                    OVLP=params['OVLP'],
                    AVRGING=params['AVRGING'],
                    Noisefr=params['Noisefr'], # Pass correct Noisefr
                    FLOOR=params['FLOOR'],
                    VAD=params['VAD'],
                    # output_dir=output_dir_snr, # Uncomment if want to save audio
                    # output_file=f"{enhanced_file_description}.wav"
                )
            except Exception as e:
                print(f"!!! ERROR during Python mband processing: {e}. Skipping this file.")
                continue

            # Step 3: Compute and save metrics
            print("3. Computing speech enhancement metrics...")
             # Ensure tensors are 1D for metrics if needed, and correct length
            if clean_waveform.dim() > 1:
                clean_for_metrics = clean_waveform[0, :]
            else:
                clean_for_metrics = clean_waveform

            if enhanced_speech.dim() > 1:
                enhanced_for_metrics = enhanced_speech[0, :]
            else:
                enhanced_for_metrics = enhanced_speech
                
            # Trim to minimum length BEFORE metrics
            min_len = min(clean_for_metrics.shape[0], enhanced_for_metrics.shape[0])
            clean_for_metrics = clean_for_metrics[:min_len]
            enhanced_for_metrics = enhanced_for_metrics[:min_len]

            try:
                metrics = compute_and_save_speech_metrics(
                    clean_tensor=clean_for_metrics,
                    enhanced_tensor=enhanced_for_metrics,
                    fs=enhanced_fs, # Should match processing_sr
                    clean_name=clean_filename,
                    enhanced_name=enhanced_file_description, # Use descriptive name
                    csv_dir=str(results_dir_snr),
                    csv_filename=f'{alg_name}_metrics.csv' # One CSV per config/SNR
                )

                # Print summary for this file
                pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN"
                stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
                si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
                dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"
                print(f"  Metrics -> PESQ: {pesq_str} | STOI: {stoi_str} | SI-SDR: {si_sdr_str} | DNSMOS: {dnsmos_str}")

            except Exception as e:
                 print(f"!!! ERROR during metric computation: {e}. Skipping metrics for this file.")
                 continue


        # Merge CSVs for this config and SNR level
        snr_end_time = time.time()
        print(f"\nMerging results for {alg_name} at {snr_dB} dB...")
        print(f"Time for SNR {snr_dB}dB: {snr_end_time - snr_start_time:.2f} seconds")
        try:
            merged_path = merge_csvs(
                input_dir=results_dir_snr,
                output_dir=results_dir_snr.parent, # Save in the alg_name folder
                output_filename=f'{alg_name}_[{snr_dB}]dB_MERGED.csv',
                keep_source= True
            )
            if merged_path:
                print(f"Merged results saved to: {merged_path}")
            else:
                 print(f"No CSVs found to merge for {alg_name} at {snr_dB} dB.")
            delete_csvs(input_directory=results_dir_snr) 
        except Exception as e:
            print(f"!!! ERROR merging CSVs: {e}")

    config_end_time = time.time()
    print(f"\n{'='*100}")
    print(f"COMPLETED CONFIGURATION: {alg_name}")
    print(f"Total time for this config: {config_end_time - config_start_time:.2f} seconds")
    print(f"{'='*100}")


# ============================================================================
# 4. FINAL CLEANUP & SUMMARY
# ============================================================================
end_time_total = time.time()
print(f"\n{'='*100}")
print("ALL EXPERIMENTS COMPLETE")
print(f"Total execution time: {end_time_total - start_time_total:.2f} seconds")
print(f"All results saved under base directory: {results_dir_base}")
print(f"{'='*100}")

