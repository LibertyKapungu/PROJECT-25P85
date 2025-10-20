"""
FINAL GTCRN+SS Parameter Sweep - Full SNR Range - Optimized
Focus on FLOOR and FRMSZ (parameters that matter)
Uses actual dataset: All NOIZEUS noise × p092 EARS utterances
"""

import pandas as pd
from pathlib import Path
import sys
import numpy as np
import random
import torch
import torchaudio
from itertools import product
import time

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

gtcrn_path = repo_root / "src" / "deep_learning" / "gtcrn_model" 
sys.path.insert(0, str(gtcrn_path))

results_dir = repo_root / 'results' / 'FINAL_GTCRN_SS_SWEEP_OPTIMIZED'
results_dir.mkdir(parents=True, exist_ok=True)

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.mband import mband
from deep_learning.gtcrn_model.gtcrn import GTCRN

def enhance_with_gtcrn(noisy_waveform, model, device, target_sr=16000):
    """Enhance using GTCRN."""
    original_shape = noisy_waveform.shape
    
    if noisy_waveform.dim() == 1:
        mix = noisy_waveform.unsqueeze(0)
    else:
        mix = noisy_waveform
    
    if mix.shape[0] > 1:
        mix = mix[0:1, :]
    
    mix = mix.to(device)
    mix_np = mix.squeeze(0).cpu().numpy()
    
    input_stft = torch.stft(
        torch.from_numpy(mix_np),
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=False
    ).to(device)
    
    with torch.no_grad():
        output = model(input_stft[None])[0]
    
    real = output[..., 0]
    imag = output[..., 1]
    complex_output = torch.complex(real, imag)
    
    enhanced = torch.istft(
        complex_output,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=False
    )
    
    if len(original_shape) == 1:
        enhanced = enhanced.squeeze()
    else:
        enhanced = enhanced.unsqueeze(0)
    
    return enhanced

def compute_metrics_fast(clean, enhanced, fs):
    """Fast metric computation."""
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
    clean_cpu = clean.cpu().float().squeeze()
    enhanced_cpu = enhanced.cpu().float().squeeze()
    
    min_len = min(clean_cpu.shape[-1], enhanced_cpu.shape[-1])
    clean_cpu = clean_cpu[..., :min_len]
    enhanced_cpu = enhanced_cpu[..., :min_len]
    
    clean_batch = clean_cpu.unsqueeze(0)
    enhanced_batch = enhanced_cpu.unsqueeze(0)
    
    try:
        pesq_metric = PerceptualEvaluationSpeechQuality(fs, 'wb')
        pesq_score = pesq_metric(enhanced_batch, clean_batch).item()
    except:
        pesq_score = float('nan')
    
    try:
        stoi_metric = ShortTimeObjectiveIntelligibility(fs)
        stoi_score = stoi_metric(enhanced_batch, clean_batch).item()
    except:
        stoi_score = float('nan')
    
    try:
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        si_sdr_score = si_sdr_metric(enhanced_batch, clean_batch).item()
    except:
        si_sdr_score = float('nan')
    
    return {'PESQ': pesq_score, 'STOI': stoi_score, 'SI_SDR': si_sdr_score}

# ====================================
# PARAMETER RANGES - OPTIMIZED
# Focus on what matters: FLOOR and FRMSZ
# ====================================
param_grid = {
    'Nband': [4,6, 8],                                    
    'Freq_spacing': ['linear'],                      
    'FRMSZ': [2, 4, 8, 10, 20, 30],      
    'OVLP': [50, 75],                                    
    'FLOOR': [0.001, 0.01, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  
}

# Fixed parameters
FIXED_PARAMS = {
    'AVRGING': 1,
    'Noisefr': 1,
    'VAD': 0,
}

param_combinations = list(product(
    param_grid['Nband'],
    param_grid['Freq_spacing'],
    param_grid['FRMSZ'],
    param_grid['OVLP'],
    param_grid['FLOOR']
))

print(f"Total parameter combinations: {len(param_combinations)}")

# ====================================
# LOAD DATA
# ====================================
print(f"\nLoading GTCRN model...")
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cpu")
model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])

print("Loading EARS dataset (p092 only - test mode)...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS utterances from p092")

print("Loading NOIZEUS dataset (FULL)...")
noizeus_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noizeus_files)} NOIZEUS noise files")

# Create audio pairs
print("Creating audio pairs...")
paired_files = create_audio_pairs(noizeus_files, ears_files)
print(f"Total test pairs: {len(paired_files)}")

SNR_LEVELS = [-5, 0, 5, 10, 15]
total_tests = len(paired_files) * len(param_combinations) * len(SNR_LEVELS)
print(f"\nTotal tests to run: {total_tests:,}")
print(f"Estimated time: {total_tests * 0.01 / 3600:.1f} hours (rough estimate)")

# ====================================
# RUN SWEEP
# ====================================
all_results = []
baseline_results = []

overall_start = time.time()
test_count = 0
snr_count = 0

for snr_dB in SNR_LEVELS:
    snr_count += 1
    print(f"\n{'='*80}")
    print(f"SNR = {snr_dB} dB ({snr_count}/{len(SNR_LEVELS)})")
    print(f"{'='*80}")
    
    snr_start = time.time()
    
    for pair_idx, (noise_path, clean_path) in enumerate(paired_files):
        participant = clean_path.parent.name if hasattr(clean_path, 'parent') else 'unknown'
        noise_name = noise_path.stem if hasattr(noise_path, 'stem') else 'unknown'
        
        if pair_idx % max(1, len(paired_files)//5) == 0:
            print(f"  Pair {pair_idx+1}/{len(paired_files)}: {noise_name} + {participant}")
        
        try:
            clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                clean_speech=clean_path,
                noisy_audio=noise_path,
                snr_db=snr_dB
            )
            
            if clean_sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)
                clean_waveform_16k = resampler(clean_waveform)
                noisy_speech_16k = resampler(noisy_speech)
                processing_sr = 16000
            else:
                clean_waveform_16k = clean_waveform
                noisy_speech_16k = noisy_speech
                processing_sr = clean_sr
            
            # GTCRN enhancement
            gtcrn_enhanced = enhance_with_gtcrn(
                noisy_waveform=noisy_speech_16k,
                model=model,
                device=device,
                target_sr=processing_sr
            )
            
            gtcrn_metrics = compute_metrics_fast(clean_waveform_16k, gtcrn_enhanced, processing_sr)
            gtcrn_pesq_norm = (gtcrn_metrics['PESQ'] + 0.5) / 5.0
            gtcrn_weighted = 0.5 * gtcrn_pesq_norm + 0.5 * gtcrn_metrics['STOI']
            
            baseline_results.append({
                'SNR_dB': snr_dB,
                'Noise_File': noise_name,
                'Participant': participant,
                'Method': 'GTCRN_only',
                'PESQ': gtcrn_metrics['PESQ'],
                'STOI': gtcrn_metrics['STOI'],
                'SI_SDR': gtcrn_metrics['SI_SDR'],
                'Weighted_Score': gtcrn_weighted
            })
            
            # Test all SS configurations
            for params in param_combinations:
                nband, freq_spacing, frmsz, ovlp, floor = params
                
                try:
                    final_enhanced, final_fs = mband(
                        noisy_audio=gtcrn_enhanced,
                        fs=processing_sr,
                        Nband=nband,
                        Freq_spacing=freq_spacing,
                        FRMSZ=frmsz,
                        OVLP=ovlp,
                        AVRGING=FIXED_PARAMS['AVRGING'],
                        Noisefr=FIXED_PARAMS['Noisefr'],
                        FLOOR=floor,
                        VAD=FIXED_PARAMS['VAD'],
                    )
                    
                    metrics = compute_metrics_fast(clean_waveform_16k, final_enhanced, final_fs)
                    pesq_norm = (metrics['PESQ'] + 0.5) / 5.0
                    weighted_score = 0.5 * pesq_norm + 0.5 * metrics['STOI']
                    
                    pesq_improvement = metrics['PESQ'] - gtcrn_metrics['PESQ']
                    stoi_improvement = metrics['STOI'] - gtcrn_metrics['STOI']
                    
                    result = {
                        'SNR_dB': snr_dB,
                        'Noise_File': noise_name,
                        'Participant': participant,
                        'Method': 'GTCRN+SS',
                        'FRMSZ': frmsz,
                        'FLOOR': floor,
                        'PESQ': metrics['PESQ'],
                        'STOI': metrics['STOI'],
                        'SI_SDR': metrics['SI_SDR'],
                        'Weighted_Score': weighted_score,
                        'PESQ_improvement': pesq_improvement,
                        'STOI_improvement': stoi_improvement,
                        'Status': 'Success'
                    }
                    
                except Exception as e:
                    result = {
                        'SNR_dB': snr_dB,
                        'Noise_File': noise_name,
                        'Participant': participant,
                        'Method': 'GTCRN+SS',
                        'FRMSZ': frmsz,
                        'FLOOR': floor,
                        'PESQ': None,
                        'STOI': None,
                        'SI_SDR': None,
                        'Weighted_Score': None,
                        'PESQ_improvement': None,
                        'STOI_improvement': None,
                        'Status': 'Error'
                    }
                
                all_results.append(result)
                test_count += 1
                
                if test_count % 200 == 0:
                    elapsed = time.time() - overall_start
                    rate = test_count / elapsed
                    remaining = (total_tests - test_count) / rate / 3600
                    progress = test_count / total_tests * 100
                    print(f"    Progress: {test_count:,}/{total_tests:,} ({progress:.1f}%) - {remaining:.1f}h remaining")
        
        except Exception as e:
            continue
    
    snr_elapsed = time.time() - snr_start
    print(f"  SNR {snr_dB} dB completed in {snr_elapsed/60:.1f} minutes")

# ====================================
# SAVE RESULTS
# ====================================
results_df = pd.DataFrame(all_results)
baseline_df = pd.DataFrame(baseline_results)

full_df = pd.concat([baseline_df, results_df], ignore_index=True)
full_df.to_csv(results_dir / 'gtcrn_ss_sweep.csv', index=False)

success_df = results_df[results_df['Status'] == 'Success'].copy()
success_df.to_csv(results_dir / 'gtcrn_ss_configurations.csv', index=False)

elapsed = time.time() - overall_start
print(f"\n{'='*80}")
print(f"SWEEP COMPLETE in {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")
print(f"{'='*80}")

# ====================================
# ANALYSIS
# ====================================
print("\n" + "="*80)
print("ANALYSIS - FOCUSED ON FRMSZ AND FLOOR")
print("="*80)

if len(success_df) > 0:
    print(f"\nDataset: {len(paired_files)} audio pairs × {len(SNR_LEVELS)} SNR levels")
    print(f"Total configurations: {len(success_df):,}")
    
    # Overall improvement
    avg_pesq_imp = success_df['PESQ_improvement'].mean()
    avg_stoi_imp = success_df['STOI_improvement'].mean()
    print(f"\nAverage improvement over GTCRN baseline:")
    print(f"  PESQ: {avg_pesq_imp:+.4f}")
    print(f"  STOI: {avg_stoi_imp:+.4f}")
    
    # Impact of FLOOR
    print("\n" + "="*80)
    print("FLOOR PARAMETER IMPACT")
    print("="*80)
    floor_impact = success_df.groupby('FLOOR').agg({
        'Weighted_Score': 'mean',
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean'
    }).round(4)
    print(floor_impact)
    
    # Impact of FRMSZ
    print("\n" + "="*80)
    print("FRMSZ PARAMETER IMPACT")
    print("="*80)
    frmsz_impact = success_df.groupby('FRMSZ').agg({
        'Weighted_Score': 'mean',
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean'
    }).round(4)
    print(frmsz_impact)
    
    # Best by SNR
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY SNR")
    print("="*80)
    
    for snr in SNR_LEVELS:
        df_snr = success_df[success_df['SNR_dB'] == snr]
        if len(df_snr) > 0:
            best = df_snr.loc[df_snr['Weighted_Score'].idxmax()]
            print(f"\nSNR = {snr} dB:")
            print(f"  Best: FRMSZ={int(best['FRMSZ'])}ms, FLOOR={best['FLOOR']:.3f}")
            print(f"  Metrics: PESQ={best['PESQ']:.4f}, STOI={best['STOI']:.4f}, SI-SDR={best['SI_SDR']:.2f}")
            print(f"  Improvement: PESQ {best['PESQ_improvement']:+.4f}, STOI {best['STOI_improvement']:+.4f}")
    
    # Universal best config
    print("\n" + "="*80)
    print("BEST UNIVERSAL CONFIGURATION (across all SNRs)")
    print("="*80)
    
    best_overall = success_df.loc[success_df['Weighted_Score'].idxmax()]
    print(f"\nFRMSZ={int(best_overall['FRMSZ'])}ms, FLOOR={best_overall['FLOOR']:.3f}")
    print(f"Metrics: PESQ={best_overall['PESQ']:.4f}, STOI={best_overall['STOI']:.4f}, SI-SDR={best_overall['SI_SDR']:.2f}")
    print(f"Improvement: PESQ {best_overall['PESQ_improvement']:+.4f}, STOI {best_overall['STOI_improvement']:+.4f}")

print(f"\n{'='*80}")
print(f"Results saved to: {results_dir}")
print(f"{'='*80}")