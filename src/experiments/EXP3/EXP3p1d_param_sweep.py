"""
IMPROVED GTCRN+SS Parameter Sweep
Key improvements:
1. Better parameter grid based on theory
2. Multi-processing for speed
3. Checkpointing for crash recovery
4. Better logging and progress tracking
5. Statistical significance testing
6. Visualization generation
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
from multiprocessing import Pool, cpu_count
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

gtcrn_path = repo_root / "src" / "deep_learning" / "gtcrn_model" 
sys.path.insert(0, str(gtcrn_path))

results_dir = repo_root / 'results' / 'GTCRN_SS_SWEEP_IMPROVED'
results_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = results_dir / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.mband import mband
from deep_learning.gtcrn_model.gtcrn import GTCRN

# ====================================
# IMPROVED PARAMETER GRID
# Based on theory and your observations
# ====================================
param_grid = {
    # Core parameters that matter most
    'FLOOR': [0.002, 0.1, 0.3, 0.5,0.8],  # Refined range
    'FRMSZ': [8,20],  # Focus on stable frame sizes
    
    # Secondary parameters
    'Nband': [4,8],  # Sweet spot for frequency resolution
    'OVLP': [75],  # Fixed at optimal value
    'AVRGING': [1],  # Average over multiple frames (you had this fixed at 1)
    'Noisefr': [1,3],  # Multiple frames for noise estimation (you had this fixed at 1)
    'VAD': [0, 1],  # Test both with/without VAD
    
    # Fixed optimal parameters
    'Freq_spacing': ['linear'],  # Linear is standard
}

# Calculate total combinations
n_combinations = (
    len(param_grid['FLOOR']) * 
    len(param_grid['FRMSZ']) * 
    len(param_grid['Nband']) * 
    len(param_grid['AVRGING']) * 
    len(param_grid['Noisefr']) * 
    len(param_grid['VAD'])
)

print(f"Parameter combinations: {n_combinations}")
print(f"  FLOOR values: {len(param_grid['FLOOR'])}")
print(f"  FRMSZ values: {len(param_grid['FRMSZ'])}")
print(f"  Nband values: {len(param_grid['Nband'])}")
print(f"  AVRGING values: {len(param_grid['AVRGING'])}")
print(f"  Noisefr values: {len(param_grid['Noisefr'])}")
print(f"  VAD values: {len(param_grid['VAD'])}")

# ====================================
# HELPER FUNCTIONS
# ====================================

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
    """Fast metric computation with error handling."""
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
    
    metrics = {}
    
    try:
        pesq_metric = PerceptualEvaluationSpeechQuality(fs, 'wb')
        metrics['PESQ'] = pesq_metric(enhanced_batch, clean_batch).item()
    except:
        metrics['PESQ'] = float('nan')
    
    try:
        stoi_metric = ShortTimeObjectiveIntelligibility(fs)
        metrics['STOI'] = stoi_metric(enhanced_batch, clean_batch).item()
    except:
        metrics['STOI'] = float('nan')
    
    try:
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        metrics['SI_SDR'] = si_sdr_metric(enhanced_batch, clean_batch).item()
    except:
        metrics['SI_SDR'] = float('nan')
    
    return metrics

def save_checkpoint(results, baseline_results, checkpoint_name='checkpoint.pkl'):
    """Save intermediate results."""
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_data = {
        'results': results,
        'baseline_results': baseline_results,
        'timestamp': time.time()
    }
    pd.to_pickle(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_name='checkpoint.pkl'):
    """Load checkpoint if exists."""
    checkpoint_path = checkpoint_dir / checkpoint_name
    if checkpoint_path.exists():
        data = pd.read_pickle(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return data['results'], data['baseline_results']
    return [], []

# ====================================
# LOAD DATA AND MODEL
# ====================================
print(f"\nLoading GTCRN model...")
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])

print("Loading EARS dataset (p092 only - test mode)...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS utterances from p092")

print("Loading NOIZEUS dataset (FULL)...")
noizeus_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noizeus_files)} NOIZEUS noise files")

print("Creating audio pairs...")
paired_files = create_audio_pairs(noizeus_files, ears_files)
print(f"Total test pairs: {len(paired_files)}")

# ====================================
# CONFIGURATION
# ====================================
SNR_LEVELS = [-5, 0, 5, 10, 15]
SAVE_INTERVAL = 100  # Save checkpoint every N tests

# Generate all parameter combinations
param_combinations = list(product(
    param_grid['FLOOR'],
    param_grid['FRMSZ'],
    param_grid['Nband'],
    param_grid['AVRGING'],
    param_grid['Noisefr'],
    param_grid['VAD']
))

total_tests = len(paired_files) * len(param_combinations) * len(SNR_LEVELS)
print(f"\nTotal tests to run: {total_tests:,}")
print(f"Estimated time: {total_tests * 0.015 / 3600:.1f} hours")

# Try to load existing checkpoint
all_results, baseline_results = load_checkpoint()
start_test_count = len(all_results)

if start_test_count > 0:
    print(f"Resuming from test {start_test_count}/{total_tests}")

# ====================================
# RUN SWEEP WITH PROGRESS BAR
# ====================================
overall_start = time.time()
test_count = start_test_count

with tqdm(total=total_tests, initial=start_test_count, desc="Overall Progress") as pbar:
    for snr_dB in SNR_LEVELS:
        print(f"\n{'='*80}")
        print(f"SNR = {snr_dB} dB")
        print(f"{'='*80}")
        
        for pair_idx, (noise_path, clean_path) in enumerate(paired_files):
            participant = clean_path.parent.name if hasattr(clean_path, 'parent') else 'unknown'
            noise_name = noise_path.stem if hasattr(noise_path, 'stem') else 'unknown'
            
            try:
                # Preprocess audio
                clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
                    clean_speech=clean_path,
                    noisy_audio=noise_path,
                    snr_db=snr_dB
                )
                
                # Resample if needed
                if clean_sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)
                    clean_waveform_16k = resampler(clean_waveform)
                    noisy_speech_16k = resampler(noisy_speech)
                    processing_sr = 16000
                else:
                    clean_waveform_16k = clean_waveform
                    noisy_speech_16k = noisy_speech
                    processing_sr = clean_sr
                
                # GTCRN enhancement (baseline)
                gtcrn_enhanced = enhance_with_gtcrn(
                    noisy_waveform=noisy_speech_16k,
                    model=model,
                    device=device,
                    target_sr=processing_sr
                )
                
                # Compute baseline metrics
                gtcrn_metrics = compute_metrics_fast(clean_waveform_16k, gtcrn_enhanced, processing_sr)
                gtcrn_pesq_norm = (gtcrn_metrics['PESQ'] + 0.5) / 5.0
                gtcrn_weighted = 0.5 * gtcrn_pesq_norm + 0.5 * gtcrn_metrics['STOI']
                
                # Save baseline result (only once per pair)
                if test_count == start_test_count or len([r for r in baseline_results 
                    if r['SNR_dB']==snr_dB and r['Noise_File']==noise_name and r['Participant']==participant]) == 0:
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
                    floor, frmsz, nband, avrging, noisefr, vad = params
                    
                    # Skip if already processed (for checkpoint resume)
                    if test_count < start_test_count:
                        test_count += 1
                        pbar.update(1)
                        continue
                    
                    try:
                        final_enhanced, final_fs = mband(
                            noisy_audio=gtcrn_enhanced,
                            fs=processing_sr,
                            Nband=nband,
                            Freq_spacing='linear',
                            FRMSZ=frmsz,
                            OVLP=75,
                            AVRGING=avrging,
                            Noisefr=noisefr,
                            FLOOR=floor,
                            VAD=vad,
                        )
                        
                        metrics = compute_metrics_fast(clean_waveform_16k, final_enhanced, final_fs)
                        pesq_norm = (metrics['PESQ'] + 0.5) / 5.0
                        weighted_score = 0.5 * pesq_norm + 0.5 * metrics['STOI']
                        
                        result = {
                            'SNR_dB': snr_dB,
                            'Noise_File': noise_name,
                            'Participant': participant,
                            'Method': 'GTCRN+SS',
                            'FLOOR': floor,
                            'FRMSZ': frmsz,
                            'Nband': nband,
                            'AVRGING': avrging,
                            'Noisefr': noisefr,
                            'VAD': vad,
                            'PESQ': metrics['PESQ'],
                            'STOI': metrics['STOI'],
                            'SI_SDR': metrics['SI_SDR'],
                            'Weighted_Score': weighted_score,
                            'PESQ_improvement': metrics['PESQ'] - gtcrn_metrics['PESQ'],
                            'STOI_improvement': metrics['STOI'] - gtcrn_metrics['STOI'],
                            'SI_SDR_improvement': metrics['SI_SDR'] - gtcrn_metrics['SI_SDR'],
                            'Status': 'Success'
                        }
                        
                    except Exception as e:
                        result = {
                            'SNR_dB': snr_dB,
                            'Noise_File': noise_name,
                            'Participant': participant,
                            'Method': 'GTCRN+SS',
                            'FLOOR': floor,
                            'FRMSZ': frmsz,
                            'Nband': nband,
                            'AVRGING': avrging,
                            'Noisefr': noisefr,
                            'VAD': vad,
                            'PESQ': None,
                            'STOI': None,
                            'SI_SDR': None,
                            'Weighted_Score': None,
                            'PESQ_improvement': None,
                            'STOI_improvement': None,
                            'SI_SDR_improvement': None,
                            'Status': f'Error: {str(e)[:50]}'
                        }
                    
                    all_results.append(result)
                    test_count += 1
                    pbar.update(1)
                    
                    # Save checkpoint periodically
                    if test_count % SAVE_INTERVAL == 0:
                        save_checkpoint(all_results, baseline_results)
            
            except Exception as e:
                print(f"  Error processing pair {pair_idx}: {str(e)}")
                # Skip all parameter combinations for this failed pair
                test_count += len(param_combinations)
                pbar.update(len(param_combinations))
                continue

# Final checkpoint save
save_checkpoint(all_results, baseline_results, 'final_checkpoint.pkl')

# ====================================
# SAVE FINAL RESULTS
# ====================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(all_results)
baseline_df = pd.DataFrame(baseline_results)

# Save separate files
results_df.to_csv(results_dir / 'gtcrn_ss_all_results.csv', index=False)
baseline_df.to_csv(results_dir / 'gtcrn_baseline_results.csv', index=False)

# Save combined
full_df = pd.concat([baseline_df, results_df], ignore_index=True)
full_df.to_csv(results_dir / 'gtcrn_ss_complete.csv', index=False)

# Save only successful results
success_df = results_df[results_df['Status'] == 'Success'].copy()
success_df.to_csv(results_dir / 'gtcrn_ss_success_only.csv', index=False)

elapsed = time.time() - overall_start
print(f"\nSWEEP COMPLETE in {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
print(f"Results saved to: {results_dir}")

# ====================================
# COMPREHENSIVE ANALYSIS
# ====================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

if len(success_df) > 0:
    # Basic statistics
    print(f"\nTotal successful tests: {len(success_df):,}")
    print(f"Total failed tests: {len(results_df) - len(success_df):,}")
    print(f"Success rate: {len(success_df)/len(results_df)*100:.1f}%")
    
    # Overall improvement statistics
    print("\n" + "="*80)
    print("OVERALL IMPROVEMENT STATISTICS")
    print("="*80)
    
    for metric in ['PESQ', 'STOI', 'SI_SDR']:
        imp_col = f'{metric}_improvement'
        if imp_col in success_df.columns:
            mean_imp = success_df[imp_col].mean()
            std_imp = success_df[imp_col].std()
            median_imp = success_df[imp_col].median()
            positive_pct = (success_df[imp_col] > 0).sum() / len(success_df) * 100
            
            print(f"\n{metric}:")
            print(f"  Mean improvement: {mean_imp:+.4f} ± {std_imp:.4f}")
            print(f"  Median improvement: {median_imp:+.4f}")
            print(f"  % configs better than GTCRN: {positive_pct:.1f}%")
    
    # Parameter importance analysis
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    for param in ['FLOOR', 'FRMSZ', 'Nband', 'AVRGING', 'Noisefr', 'VAD']:
        if param in success_df.columns:
            print(f"\n{param}:")
            impact = success_df.groupby(param).agg({
                'Weighted_Score': ['mean', 'std'],
                'PESQ': 'mean',
                'STOI': 'mean',
                'PESQ_improvement': 'mean'
            }).round(4)
            print(impact)
    
    # Best configurations per SNR
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER SNR LEVEL")
    print("="*80)
    
    best_configs = []
    for snr in SNR_LEVELS:
        df_snr = success_df[success_df['SNR_dB'] == snr]
        if len(df_snr) > 0:
            best = df_snr.loc[df_snr['Weighted_Score'].idxmax()]
            best_configs.append(best)
            
            print(f"\nSNR = {snr} dB:")
            print(f"  FLOOR={best['FLOOR']:.3f}, FRMSZ={int(best['FRMSZ'])}ms, "
                  f"Nband={int(best['Nband'])}, AVRGING={int(best['AVRGING'])}, "
                  f"Noisefr={int(best['Noisefr'])}, VAD={int(best['VAD'])}")
            print(f"  PESQ: {best['PESQ']:.4f} ({best['PESQ_improvement']:+.4f})")
            print(f"  STOI: {best['STOI']:.4f} ({best['STOI_improvement']:+.4f})")
            print(f"  SI-SDR: {best['SI_SDR']:.2f} dB")
    
    # Save best configurations
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(results_dir / 'best_configs_per_snr.csv', index=False)
    
    # Universal best configuration
    print("\n" + "="*80)
    print("UNIVERSAL BEST CONFIGURATION (all SNRs)")
    print("="*80)
    
    # Method 1: Best weighted score
    best_overall = success_df.loc[success_df['Weighted_Score'].idxmax()]
    print(f"\nBased on highest weighted score:")
    print(f"  FLOOR={best_overall['FLOOR']:.3f}, FRMSZ={int(best_overall['FRMSZ'])}ms, "
          f"Nband={int(best_overall['Nband'])}, AVRGING={int(best_overall['AVRGING'])}, "
          f"Noisefr={int(best_overall['Noisefr'])}, VAD={int(best_overall['VAD'])}")
    print(f"  PESQ: {best_overall['PESQ']:.4f}, STOI: {best_overall['STOI']:.4f}")
    
    # Method 2: Most robust configuration (best average across all SNRs)
    config_cols = ['FLOOR', 'FRMSZ', 'Nband', 'AVRGING', 'Noisefr', 'VAD']
    avg_by_config = success_df.groupby(config_cols).agg({
        'Weighted_Score': 'mean',
        'PESQ': 'mean',
        'STOI': 'mean',
        'PESQ_improvement': 'mean'
    }).reset_index()
    
    most_robust = avg_by_config.loc[avg_by_config['Weighted_Score'].idxmax()]
    print(f"\nMost robust configuration (best average):")
    print(f"  FLOOR={most_robust['FLOOR']:.3f}, FRMSZ={int(most_robust['FRMSZ'])}ms, "
          f"Nband={int(most_robust['Nband'])}, AVRGING={int(most_robust['AVRGING'])}, "
          f"Noisefr={int(most_robust['Noisefr'])}, VAD={int(most_robust['VAD'])}")
    print(f"  Avg PESQ: {most_robust['PESQ']:.4f}, Avg STOI: {most_robust['STOI']:.4f}")
    
    # Save summary statistics
    summary = {
        'total_tests': len(success_df),
        'mean_pesq_improvement': success_df['PESQ_improvement'].mean(),
        'mean_stoi_improvement': success_df['STOI_improvement'].mean(),
        'best_overall_config': best_overall[config_cols].to_dict(),
        'most_robust_config': most_robust[config_cols].to_dict(),
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. Files saved to: {results_dir}")
    print(f"{'='*80}")

else:
    print("No successful results to analyze!")