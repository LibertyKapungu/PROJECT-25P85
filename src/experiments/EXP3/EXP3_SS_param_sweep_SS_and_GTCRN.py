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
import itertools
from typing import Dict, List, Tuple, Optional

"""
Unified Parameter Sweep: Standalone SS vs Hybrid GTCRN+SS
==========================================
Mode 1 (standalone): Clean + Noise → Noisy → mband → Enhanced
Mode 2 (hybrid):     Clean + Noise → Noisy → GTCRN → mband → Enhanced

Key Insights Applied:
- Hanning window (proven 0.5-2% better)
- Skip Mel spacing (consistently underperforms)
- Skip AVRGING=0 (always worse)
- Hybrid needs FLOOR 10-100x higher than standalone
"""

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- Processing Mode ---
PROCESSING_MODE = "standalone"  # Options: "standalone" or "hybrid"

# --- Parameter Grid (Mode-Adaptive) ---
if PROCESSING_MODE == "standalone":
    PARAM_GRID = {
        'Freq_spacing': ['log'],      # Skip mel  ['linear', 'log', 'mel'], 
        'Nband': [4, 6, 8, 16],                     # Test band count
        'FRMSZ': [8, 20],                       # Latency vs quality
        'OVLP': [50, 75],                       # Standard vs smooth
        'Noisefr': [1,3],      # Low latency
        'FLOOR': [0.001, 0.02],          # Aggressive noise removal
    }
    
elif PROCESSING_MODE == "hybrid":
    PARAM_GRID = {
        'Freq_spacing': ['linear', 'log'],      # Both useful
        'Nband': [4, 6],                        # Minimal processing
        'FRMSZ': [8, 20],                       # Match GTCRN frame
        'OVLP': [50, 75],                       # Smooth blending
        'Noisefr': [1,3],      # Low latency
        'FLOOR': [0.01, 0.1, 0.3, 0.5, 0.8],  # CRITICAL: High floor!
    }
else:
    raise ValueError(f"Invalid PROCESSING_MODE: {PROCESSING_MODE}")

# --- Fixed Parameters (Proven Optimal) ---
FIXED_PARAMS = {
    'AVRGING': 1,      # Always better
    # 'Noisefr': 1,      # Low latency
    'VAD': 1,          # Essential
}

# --- SNR Range ---
SNR_RANGE = [-5, 0, 5, 10, 15]

# ============================================================================
# 2. PATH SETUP
# ============================================================================

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Add GTCRN to path for hybrid mode
if PROCESSING_MODE == "hybrid":
    gtcrn_path = repo_root / "src" / "deep_learning" / "gtcrn_model"
    sys.path.insert(0, str(gtcrn_path))

# Results directory
results_dir_base = repo_root / 'results' / 'EXP3' / 'spectral' /'PARAM_SWEEP' / PROCESSING_MODE
results_dir_base.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. IMPORTS
# ============================================================================

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)

from dsp_algorithms.spectral.mband_full_stream_hanning import mband
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# Conditional GTCRN import
if PROCESSING_MODE == "hybrid":
    from deep_learning.gtcrn_model.gtcrn import GTCRN

# ============================================================================
# 4. GTCRN ENHANCEMENT FUNCTION (Hybrid Mode)
# ============================================================================

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
    
    # Ensure correct shape and device
    if noisy_waveform.dim() == 1:
        mix = noisy_waveform.unsqueeze(0)
    else:
        mix = noisy_waveform
    
    if mix.shape[0] > 1:
        mix = mix[0:1, :]
    
    mix = mix.to(device)
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

# ============================================================================
# 5. LOAD DATASETS
# ============================================================================

print("="*80)
print(f"PARAMETER SWEEP MODE: {PROCESSING_MODE.upper()}")
print("="*80)

print("\nLoading test datasets...")
ears_files = load_ears_dataset(repo_root, mode="test")
noizeus_files = load_noizeus_dataset(repo_root)
paired_files = create_audio_pairs(noizeus_files, ears_files)

print(f"✓ Loaded {len(ears_files)} EARS files")
print(f"✓ Loaded {len(noizeus_files)} NOIZEUS files")
print(f"✓ Created {len(paired_files)} audio pairs")

# ============================================================================
# 6. GENERATE PARAMETER COMBINATIONS
# ============================================================================

param_combinations = []
for values in itertools.product(*PARAM_GRID.values()):
    config = dict(zip(PARAM_GRID.keys(), values))
    config.update(FIXED_PARAMS)
    param_combinations.append(config)

print(f"\n{'='*80}")
print(f"PARAMETER GRID SUMMARY")
print(f"{'='*80}")
print(f"Mode: {PROCESSING_MODE}")
print(f"Total combinations: {len(param_combinations)}")
print(f"Files per SNR: {len(paired_files)}")
print(f"SNR levels: {len(SNR_RANGE)}")
print(f"Total evaluations: {len(param_combinations) * len(paired_files) * len(SNR_RANGE)}")
print(f"\nParameter ranges:")
for key, values in PARAM_GRID.items():
    print(f"  {key}: {values}")
print(f"\nFixed parameters:")
for key, value in FIXED_PARAMS.items():
    print(f"  {key}: {value}")
print(f"{'='*80}\n")

# ============================================================================
# 7. LOAD GTCRN MODEL (Hybrid Mode)
# ============================================================================

gtcrn_model = None
device = None

if PROCESSING_MODE == "hybrid":
    print("Loading GTCRN model for hybrid processing...")
    try:
        checkpoint_path = repo_root / "src" / "deep_learning" / "gtcrn_model" / "checkpoints" / "model_trained_on_dns3.tar"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gtcrn_model = GTCRN().eval().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        gtcrn_model.load_state_dict(ckpt['model'])
        print(f"✓ GTCRN model loaded from: {checkpoint_path}")
        print(f"✓ Using device: {device}")
    except Exception as e:
        print(f"!!! ERROR loading GTCRN model: {e}")
        print("!!! Cannot run hybrid mode without GTCRN")
        sys.exit(1)

# ============================================================================
# 8. MAIN EXPERIMENT LOOP
# ============================================================================

start_time_total = time.time()
results_summary = []

for config_idx, config in enumerate(param_combinations, 1):
    # Generate configuration name
    config_name = (
        f"mband_{config['Freq_spacing']}_"
        f"N{config['Nband']}_"
        f"F{config['FRMSZ']}_"
        f"O{config['OVLP']}_"
        f"FL{str(config['FLOOR']).replace('.', 'p')}"
    )
    
    print(f"\n{'='*100}")
    print(f"CONFIG {config_idx}/{len(param_combinations)}: {config_name}")
    print(f"{'='*100}")
    print(f"Parameters: {config}")
    
    config_start_time = time.time()
    
    # Create results directory
    results_dir_config = results_dir_base / config_name
    results_dir_config.mkdir(parents=True, exist_ok=True)
    
    # Process each SNR level
    for snr_dB in SNR_RANGE:
        print(f"\n--- SNR: {snr_dB} dB ---")
        snr_start_time = time.time()
        
        results_dir_snr = results_dir_config / f"{snr_dB}dB"
        results_dir_snr.mkdir(parents=True, exist_ok=True)
        
        # Process each file pair
        for file_idx, (noise_path, clean_path) in enumerate(paired_files, 1):
            participant = clean_path.parent.name
            print(f"\nFile {file_idx}/{len(paired_files)}: {clean_path.name} + {noise_path.name}")
            
            # Step 1: Create noisy mixture
            try:
                clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                    clean_speech=clean_path,
                    noisy_audio=noise_path,
                    snr_db=snr_dB,
                    target_sr=16000  # GTCRN requires 16kHz
                )
                processing_sr = 16000
            except Exception as e:
                print(f"!!! ERROR preprocessing: {e}")
                continue
            
            # Step 2: Apply GTCRN if hybrid mode
            if PROCESSING_MODE == "hybrid":
                print("  → Running GTCRN enhancement...")
                try:
                    gtcrn_enhanced = enhance_with_gtcrn(
                        noisy_waveform=noisy_speech,
                        model=gtcrn_model,
                        device=device,
                        target_sr=processing_sr
                    )
                    input_for_ss = gtcrn_enhanced
                except Exception as e:
                    print(f"!!! ERROR running GTCRN: {e}")
                    continue
            else:
                input_for_ss = noisy_speech
            
            # Step 3: Apply spectral subtraction
            print(f"  → Running mband ({config_name})...")
            try:
                enhanced_speech, enhanced_fs = mband(
                    noisy_audio=input_for_ss,
                    fs=processing_sr,
                    Nband=config['Nband'],
                    Freq_spacing=config['Freq_spacing'],
                    FRMSZ=config['FRMSZ'],
                    OVLP=config['OVLP'],
                    AVRGING=config['AVRGING'],
                    Noisefr=config['Noisefr'],
                    FLOOR=config['FLOOR'],
                    VAD=config['VAD']
                )
            except Exception as e:
                print(f"!!! ERROR running mband: {e}")
                continue
            
            # Step 4: Compute metrics
            print("  → Computing metrics...")
            
            # Prepare tensors
            clean_for_metrics = clean_waveform[0] if clean_waveform.dim() > 1 else clean_waveform
            enhanced_for_metrics = enhanced_speech[0] if enhanced_speech.dim() > 1 else enhanced_speech
            
            # Trim to minimum length
            min_len = min(clean_for_metrics.shape[0], enhanced_for_metrics.shape[0])
            clean_for_metrics = clean_for_metrics[:min_len]
            enhanced_for_metrics = enhanced_for_metrics[:min_len]
            
            # Generate filenames
            clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
            noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
            
            mode_prefix = "GTCRN+SS" if PROCESSING_MODE == "hybrid" else "SS"
            enhanced_file_desc = f"{mode_prefix}_{config_name}_{clean_filename}_{noise_filename}_SNR{snr_dB}dB"
            
            try:
                metrics = compute_and_save_speech_metrics(
                    clean_tensor=clean_for_metrics,
                    enhanced_tensor=enhanced_for_metrics,
                    fs=enhanced_fs,
                    clean_name=clean_filename,
                    enhanced_name=enhanced_file_desc,
                    csv_dir=str(results_dir_snr),
                    csv_filename=f'{config_name}_metrics.csv'
                )
                
                # Print summary
                pesq = metrics.get('PESQ', float('nan'))
                si_sdr = metrics.get('SI_SDR', float('nan'))
                stoi = metrics.get('STOI', float('nan'))
                dnsmos = metrics.get('DNSMOS_mos_ovr', float('nan'))
                
                print(f"    PESQ: {pesq:.3f} | "
                      f"SI-SDR: {si_sdr:.2f} dB | "
                      f"STOI: {stoi:.3f} | "
                      f"DNSMOS: {dnsmos:.3f}")
                
            except Exception as e:
                print(f"!!! ERROR computing metrics: {e}")
                continue
        
        # Merge CSVs for this SNR
        snr_end_time = time.time()
        print(f"\n  Merging results for SNR {snr_dB} dB...")
        print(f"  Time: {snr_end_time - snr_start_time:.2f}s")
        
        try:
            merged_path = merge_csvs(
                input_dir=results_dir_snr,
                output_dir=results_dir_snr.parent,
                output_filename=f'{config_name}_SNR{snr_dB}dB_MERGED.csv',
                keep_source=True
            )
            if merged_path:
                print(f"  ✓ Merged: {merged_path.name}")
            delete_csvs(input_directory=results_dir_snr)
        except Exception as e:
            print(f"!!! ERROR merging: {e}")
    
    # Config summary
    config_end_time = time.time()
    config_time = config_end_time - config_start_time
    
    results_summary.append({
        'config': config_name,
        'time_seconds': config_time,
        'mode': PROCESSING_MODE,
        **config
    })
    
    print(f"\n{'='*100}")
    print(f"✓ COMPLETED: {config_name}")
    print(f"  Time: {config_time:.2f}s ({config_time/60:.1f} min)")
    print(f"  Avg per file: {config_time/(len(paired_files)*len(SNR_RANGE)):.1f}s")
    print(f"{'='*100}")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

end_time_total = time.time()
total_time = end_time_total - start_time_total

print(f"\n{'='*100}")
print("PARAMETER SWEEP COMPLETE")
print(f"{'='*100}")
print(f"Mode: {PROCESSING_MODE}")
print(f"Total configurations: {len(param_combinations)}")
print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hrs)")
print(f"Average per config: {total_time/len(param_combinations):.2f}s")
print(f"\nResults saved to: {results_dir_base}")

# Save summary CSV
summary_df = pd.DataFrame(results_summary)
summary_path = results_dir_base / f"sweep_summary_{PROCESSING_MODE}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")
print(f"{'='*100}\n")