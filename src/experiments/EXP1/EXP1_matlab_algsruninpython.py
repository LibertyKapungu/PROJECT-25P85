"""
Experiment MATLAB_ALGS_COMPARISON:
This experiment performs a comparative evaluation of three classical spectral
subtraction algorithms from Loizou: specsub, ss_rdc, and mband.

It runs each algorithm on a test set of noisy audio mixtures (EARS + NOIZEUS)
and computes a full set of quality metrics for each.

Processing:
Clean speech + noise -> noisy mixture -> MATLAB algorithm -> compute metrics

Algorithms Tested:
- specsub (causal, no smoothing)
- ss_rdc (causal, single-band smoothing)
- mband (AVRGING=1) (non-causal, multi-band)
- mband_avr0 (AVRGING=0) (causal, multi-band)

Metrics computed: PESQ, STOI, SI-SDR, DNSMOS
"""

import pandas as pd
from pathlib import Path
import sys
import numpy as np
import random
import torch
import torchaudio
import matlab.engine  # <-- 1. Import the MATLAB Engine
import math

# Set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

results_dir_base = repo_root / 'results' / 'EXP_MATLAB_COMPARE'

# --- Utility Imports (Ensure these paths are correct in your project) ---
from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# ============================================================================
# 1. DEFINE YOUR MATLAB PATH
# ============================================================================
# This is the folder containing specsub.m, ss_rdc.m, mband.m, and mband_avr0.m
MATLAB_SCRIPTS_PATH = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS'

# ============================================================================
# 2. DEFINE ALL EXPERIMENTS TO RUN
# ============================================================================
EXPERIMENT_CONFIGS = [
    # {
    #     'name': 'specsub',
    #     'matlab_func': 'specsub',  # <-- Will call eng.specsub()
    #     'params': {}
    # },
    {
        'name': 'ss_rdc',
        'matlab_func': 'ss_rdc',    # <-- Will call eng.ss_rdc()
        'params': {}
    },
    # --- MBAND (Non-Causal, AVRGING=1) ---
    # {
    #     'name': 'mband_N6_lin_AVR1',
    #     'matlab_func': 'mband',    # <-- Will call eng.mband()
    #     'params': {'Nband': 6, 'Freq_spacing': 'linear'}
    # },
    # {
    #     'name': 'mband_N6_log_AVR1',
    #     'matlab_func': 'mband',    # <-- Will call eng.mband()
    #     'params': {'Nband': 6, 'Freq_spacing': 'log'}
    # },
    # {
    #     'name': 'mband_N6_mel_AVR1',
    #     'matlab_func': 'mband',    # <-- Will call eng.mband()
    #     'params': {'Nband': 6, 'Freq_spacing': 'mel'}
    # },
    # --- MBAND (Causal, AVRGING=0) ---
    # {
    #     'name': 'mband_N6_lin_AVR0',
    #     'matlab_func': 'mband_avr0', # <-- Will call eng.mband_avr0()
    #     'params': {'Nband': 6, 'Freq_spacing': 'linear'}
    # },
    # {
    #     'name': 'mband_N6_log_AVR0',
    #     'matlab_func': 'mband_avr0', # <-- Will call eng.mband_avr0()
    #     'params': {'Nband': 6, 'Freq_spacing': 'log'}
    # },
    # {
    #     'name': 'mband_N6_mel_AVR0',
    #     'matlab_func': 'mband_avr0', # <-- Will call eng.mband_avr0()
    #     'params': {'Nband': 6, 'Freq_spacing': 'mel'}
    # },
]

# ============================================================================
# 3. START MATLAB ENGINE
# ============================================================================
print("Starting MATLAB engine...")
# Start asynchronously to avoid blocking
# eng = matlab.engine.start_matlab("-desktop", async=True).result()
# NEW (FIXED) LINE
eng = matlab.engine.start_matlab("-desktop")
eng.addpath(MATLAB_SCRIPTS_PATH, nargout=0)
print(f"MATLAB engine started and path added: {MATLAB_SCRIPTS_PATH}")


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

# Create a temporary directory for MATLAB file I/O
temp_dir = results_dir_base / "temp_audio"
temp_dir.mkdir(parents=True, exist_ok=True)
temp_noisy_path = str(temp_dir / "temp_noisy_input.wav")
temp_enhanced_path = str(temp_dir / "temp_enhanced_output.wav")

snr_dB_range = [-5, 0, 5, 10, 15]

for snr_dB in snr_dB_range:
    print(f"\n{'='*100}")
    print(f"Processing SNR: {snr_dB} dB")
    print(f"{'='*100}")

    for config in EXPERIMENT_CONFIGS:
        alg_name = config['name']
        print(f"\n--- Processing Algorithm: {alg_name} ---")

        results_dir_snr = results_dir_base / alg_name / f"{snr_dB}dB"
        results_dir_snr.mkdir(parents=True, exist_ok=True)

        for noise_path, clean_path in paired_files:

            participant = clean_path.parent.name
            print(f"\nNoise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant}")

            # Step 1: Create noisy mixture
            clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                clean_speech=clean_path, 
                noisy_audio=noise_path, 
                snr_db=snr_dB
            )

            # Step 2: Resample to 16kHz if needed (or 8kHz if that's what your .m files expect!)
            # We'll use 16kHz as a standard based on ss_rdc logic
            if clean_sr != 16000:
                print(f"Resampling from {clean_sr}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)
                clean_waveform_16k = resampler(clean_waveform)
                noisy_speech_16k = resampler(noisy_speech)
                processing_sr = 16000
            else:
                clean_waveform_16k = clean_waveform
                noisy_speech_16k = noisy_speech
                processing_sr = clean_sr
            
            # Ensure waveform is mono for processing
            if noisy_speech_16k.dim() > 1:
                noisy_speech_16k = noisy_speech_16k[0, :]
            if clean_waveform_16k.dim() > 1:
                clean_waveform_16k = clean_waveform_16k[0, :]

            # Step 3: Enhance with MATLAB
            print(f"3. Enhancing speech with {alg_name}...")
            
            # Save noisy file to disk for MATLAB
            torchaudio.save(
                temp_noisy_path, 
                noisy_speech_16k.unsqueeze(0), # Add channel dim back for save
                processing_sr,
                bits_per_sample=16 # Use 16-bit for compatibility
            )

            # Call the correct MATLAB function
            try:
                func_name = config['matlab_func']
                p = config['params'] # Get params, will be empty for specsub/ss_rdc

                if func_name == 'specsub':
                    eng.specsub(temp_noisy_path, temp_enhanced_path, nargout=0)
                
                elif func_name == 'ss_rdc':
                    eng.ss_rdc(temp_noisy_path, temp_enhanced_path, nargout=0)
                
                elif func_name == 'mband':
                    # This calls mband.m (which has AVRGING=1)
                    eng.mband(
                        temp_noisy_path, 
                        temp_enhanced_path, 
                        p['Nband'], 
                        p['Freq_spacing'],
                        nargout=0
                    )
                
                elif func_name == 'mband_avr0':
                    # This calls mband_avr0.m (which has AVRGING=0)
                    eng.mband_avr0(
                        temp_noisy_path, 
                        temp_enhanced_path, 
                        p['Nband'], 
                        p['Freq_spacing'],
                        nargout=0
                    )

                # Load the enhanced result back into a tensor
                final_enhanced_speech, final_fs = torchaudio.load(temp_enhanced_path)

            except Exception as e:
                print(f"!!! MATLAB Error processing {temp_noisy_path}: {e}")
                print("!!! Skipping this file.")
                continue # Skip this file
            
            
            # Step 4: Prepare Tensors for Metrics (Apply Delay Fix)
            clean_for_metrics = clean_waveform_16k
            enhanced_for_metrics = final_enhanced_speech

            if config['name'] == 'ss_rdc':
                print("Applying delay compensation for ss_rdc SI-SDR.")
                
                # Your ss_rdc.m has fs-dependent M. 
                # M=32 for 8kHz, M=64 for 16kHz.
                # Delay = (M-1)/2
                if processing_sr == 16000:
                    DELAY_IN_SAMPLES = 32 # (64-1)/2 = 31.5 -> 32
                else: # 8000 Hz
                    DELAY_IN_SAMPLES = 16 # (32-1)/2 = 15.5 -> 16
                
                print(f"Using {DELAY_IN_SAMPLES} sample delay for {processing_sr}Hz.")
                zero_padding = torch.zeros((DELAY_IN_SAMPLES,), dtype=clean_for_metrics.dtype)
                padded_clean = torch.cat((zero_padding, clean_for_metrics), dim=0)

                min_len = min(padded_clean.shape[0], enhanced_for_metrics.shape[1])
                
                clean_for_metrics = padded_clean[:min_len]
                enhanced_for_metrics = enhanced_for_metrics[0, :min_len] # Make mono
            
            # Ensure both are mono and same length for other algs
            if enhanced_for_metrics.dim() > 1:
                enhanced_for_metrics = enhanced_for_metrics[0, :]
            
            min_len = min(clean_for_metrics.shape[0], enhanced_for_metrics.shape[0])
            clean_for_metrics = clean_for_metrics[:min_len]
            enhanced_for_metrics = enhanced_for_metrics[:min_len]


            # Step 5: Compute and save metrics
            clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
            noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
            
            print("5. Computing speech quality metrics...")
            metrics = compute_and_save_speech_metrics(
                clean_tensor=clean_for_metrics,
                enhanced_tensor=enhanced_for_metrics,
                fs=final_fs,
                clean_name=clean_filename,
                enhanced_name=f"{alg_name}_{clean_filename}_{noise_filename}_SNR[{snr_dB}]dB",
                csv_dir=str(results_dir_snr),
                csv_filename=f'{alg_name}_metrics.csv'
            )
            
            # Print metrics summary
            pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN"
            stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
            si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
            dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"
            print(f"  PESQ: {pesq_str} | STOI: {stoi_str} | SI-SDR: {si_sdr_str} | DNSMOS: {dnsmos_str}")


        # Merge CSVs for this config and SNR level
        print(f"\nMerging results for {alg_name} at {snr_dB} dB...")
        merged_path = merge_csvs(
            input_dir=results_dir_snr,
            output_dir=results_dir_snr.parent, # Save in the alg_name folder
            output_filename=f'{alg_name}_[{snr_dB}]dB_MERGED.csv',
            keep_source=True
        )
        print(f"Merged results saved to: {merged_path}")
        delete_csvs(input_directory=results_dir_snr)

# ============================================================================
# 6. STOP MATLAB ENGINE
# ============================================================================
print(f"\n{'='*100}")
print("EXPERIMENT COMPLETE")
print("Stopping MATLAB engine...")
eng.quit()
print(f"All results saved to: {results_dir_base}")
print(f"{'='*100}")