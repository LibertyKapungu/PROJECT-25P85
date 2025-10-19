"""
FINAL GTCRN+SS Parameter Sweep - Optimized for Time
Based on findings: linear freq, 75% overlap are optimal
Only test parameters that matter: Nband and Floor
Use representative noise files from each category
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

results_dir = repo_root / 'results' / 'FINAL_GTCRN_SS_SWEEP'
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

# def compute_metrics_fast(clean, enhanced, fs):
#     """Fast metric computation."""
#     from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
#     from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
#     from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
#     clean_cpu = clean.cpu().float()
#     enhanced_cpu = enhanced.cpu().float()
    
#     min_len = min(clean_cpu.shape[-1], enhanced_cpu.shape[-1])
#     clean_cpu = clean_cpu[..., :min_len]
#     enhanced_cpu = enhanced_cpu[..., :min_len]
    
#     try:
#         pesq_metric = PerceptualEvaluationSpeechQuality(fs, 'wb')
#         pesq_score = pesq_metric(enhanced_cpu.unsqueeze(0), clean_cpu.unsqueeze(0)).item()
#     except:
#         pesq_score = float('nan')
    
#     try:
#         stoi_metric = ShortTimeObjectiveIntelligibility(fs)
#         stoi_score = stoi_metric(enhanced_cpu.unsqueeze(0), clean_cpu.unsqueeze(0)).item()
#     except:
#         stoi_score = float('nan')
    
#     try:
#         si_sdr_metric = ScaleInvariantSignalDistortionRatio()
#         si_sdr_score = si_sdr_metric(enhanced_cpu.unsqueeze(0), clean_cpu.unsqueeze(0)).item()
#     except:
#         si_sdr_score = float('nan')
    
#     return {'PESQ': pesq_score, 'STOI': stoi_score, 'SI_SDR': si_sdr_score}

def compute_metrics_fast(clean, enhanced, fs):
    """Fast metric computation with proper shape handling."""
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    
    # Ensure both are 1D for consistent processing
    clean_cpu = clean.cpu().float().squeeze()
    enhanced_cpu = enhanced.cpu().float().squeeze()
    
    # Match lengths
    min_len = min(clean_cpu.shape[-1], enhanced_cpu.shape[-1])
    clean_cpu = clean_cpu[..., :min_len]
    enhanced_cpu = enhanced_cpu[..., :min_len]
    
    # Add batch dimension consistently
    clean_batch = clean_cpu.unsqueeze(0)
    enhanced_batch = enhanced_cpu.unsqueeze(0)
    
    print(f"DEBUG: clean_batch shape={clean_batch.shape}, enhanced_batch shape={enhanced_batch.shape}")
    
    try:
        pesq_metric = PerceptualEvaluationSpeechQuality(fs, 'wb')
        pesq_score = pesq_metric(enhanced_batch, clean_batch).item()
    except Exception as e:
        print(f"PESQ error: {e}")
        pesq_score = float('nan')
    
    try:
        stoi_metric = ShortTimeObjectiveIntelligibility(fs)
        stoi_score = stoi_metric(enhanced_batch, clean_batch).item()
    except Exception as e:
        print(f"STOI error: {e}")
        stoi_score = float('nan')
    
    try:
        si_sdr_metric = ScaleInvariantSignalDistortionRatio()
        si_sdr_score = si_sdr_metric(enhanced_batch, clean_batch).item()
    except Exception as e:
        print(f"SI_SDR error: {e}")
        si_sdr_score = float('nan')
    
    return {'PESQ': pesq_score, 'STOI': stoi_score, 'SI_SDR': si_sdr_score}
# ====================================
# MINIMAL PARAMETER GRID
# Based on standalone SS findings
# ====================================
param_grid = {
    'Freq_spacing': ['linear'],      # Consistently best
    'Nband': [4, 6, 8],              # Test around optimal
    'FRMSZ': [8,20],                    # 8ms was best at relevant SNRs
    'OVLP': [75],                    # Clearly optimal
    'AVRGING': [1],                  # Keep on
    'Noisefr': [1,3],                  # Standard
    'VAD': [1],                      # Keep on
    'FLOOR': [0.002, 0.01]   # Gentler for GTCRN output
}

# Total: 1 × 3 × 1 × 1 × 1 × 1 × 1 × 3 = 9 configurations

param_combinations = list(product(
    param_grid['Freq_spacing'],
    param_grid['Nband'],
    param_grid['FRMSZ'],
    param_grid['OVLP'],
    param_grid['AVRGING'],
    param_grid['Noisefr'],
    param_grid['VAD'],
    param_grid['FLOOR']
))

print(f"Total configurations: {len(param_combinations)}")

# ====================================
# REPRESENTATIVE NOISE FILES
# Based on noise categories
# ====================================
NOISE_DIR = Path(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\sound_data\raw\NOIZEUS_NOISE_DATASET\Noise Recordings")

REPRESENTATIVE_NOISES = {
    'babble': 'cafeteria_babble.wav',
    'train': 'Inside Train_1.wav',
    'street': 'Street Noise.wav',
    'stationary': 'PC Fan Noise.wav'
}

# Verify noise files exist
noise_files = []
for category, filename in REPRESENTATIVE_NOISES.items():
    noise_path = NOISE_DIR / filename
    if noise_path.exists():
        noise_files.append((category, noise_path))
        print(f" Found {category}: {filename}")
    else:
        print(f" Missing {category}: {filename}")

if len(noise_files) != 4:
    print("\n WARNING: Not all noise files found!")
    print("Available files in directory:")
    for f in sorted(NOISE_DIR.glob("*.wav")):
        print(f"  - {f.name}")
    sys.exit(1)

# Load GTCRN model
print("\nLoading GTCRN model...")
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cpu")
model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])

# Load EARS dataset
print("Loading EARS test dataset...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS files")

# Use 2 speech files per noise type for efficiency
NUM_SPEECH_FILES = 2
# paired_files = []
# for category, noise_path in noise_files:
#     for ears_path in ears_files[:NUM_SPEECH_FILES]:
#         paired_files.append((category, noise_path, ears_path))

paired_files = []
for category, noise_path in noise_files:
    for ears_file in ears_files[:NUM_SPEECH_FILES]:
        # Extract path from dictionary if needed
        if isinstance(ears_file, dict):
            clean_path = ears_file['file']  # Already a Path object
        else:
            clean_path = ears_file
        
        paired_files.append((category, noise_path, clean_path))

print(f"\nTotal test pairs: {len(paired_files)}")

# Test at key SNR levels
#SNR_LEVELS = [-5, 0, 5, 10, 15]  
SNR_LEVELS = [5]  

print(f"SNR levels: {SNR_LEVELS}")
print(f"Total tests: {len(paired_files)} pairs {len(param_combinations)} configs {len(SNR_LEVELS)} SNRs")
print(f"            = {len(paired_files) * len(param_combinations) * len(SNR_LEVELS)} total")

# ====================================
# RUN SWEEP
# ====================================
all_results = []
baseline_results = []  # Store GTCRN-only results for comparison

overall_start = time.time()
test_count = 0
total_tests = len(paired_files) * len(param_combinations) * len(SNR_LEVELS)

for snr_dB in SNR_LEVELS:
    print(f"\n{'='*80}")
    print(f"SNR = {snr_dB} dB")
    print(f"{'='*80}")
    
    for category, noise_path, clean_path in paired_files:
        participant = clean_path.parent.name
        
        print(f"\n{category.upper()} + {participant}")
        
        # Create noisy mixture
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
        
        # GTCRN enhancement (once per pair)
        gtcrn_enhanced = enhance_with_gtcrn(
            noisy_waveform=noisy_speech_16k,
            model=model,
            device=device,
            target_sr=processing_sr
        )
        
        # Compute GTCRN-only baseline
        gtcrn_metrics = compute_metrics_fast(clean_waveform_16k, gtcrn_enhanced, processing_sr)
        gtcrn_pesq_norm = (gtcrn_metrics['PESQ'] + 0.5) / 5.0
        gtcrn_weighted = 0.5 * gtcrn_pesq_norm + 0.5 * gtcrn_metrics['STOI']
        
        baseline_results.append({
            'SNR_dB': snr_dB,
            'Noise_Type': category,
            'Participant': participant,
            'Method': 'GTCRN_only',
            'Nband': None,
            'Floor': None,
            'PESQ': gtcrn_metrics['PESQ'],
            'STOI': gtcrn_metrics['STOI'],
            'SI_SDR': gtcrn_metrics['SI_SDR'],
            'Weighted_Score': gtcrn_weighted
        })
        
        # Test all SS post-processing configurations
        for params in param_combinations:
            freq_spacing, nband, frmsz, ovlp, avrging, noisefr, vad, floor = params
            
            try:
                # Apply SS post-processing
                final_enhanced, final_fs = mband(
                    noisy_audio=gtcrn_enhanced,
                    fs=processing_sr,
                    Nband=nband,
                    Freq_spacing=freq_spacing,
                    FRMSZ=frmsz,
                    OVLP=ovlp,
                    AVRGING=avrging,
                    Noisefr=noisefr,
                    FLOOR=floor,
                    VAD=vad,
                )
                
                # Compute metrics
                metrics = compute_metrics_fast(clean_waveform_16k, final_enhanced, final_fs)
                pesq_norm = (metrics['PESQ'] + 0.5) / 5.0
                weighted_score = 0.5 * pesq_norm + 0.5 * metrics['STOI']
                
                # Calculate improvement over GTCRN-only
                pesq_improvement = metrics['PESQ'] - gtcrn_metrics['PESQ']
                stoi_improvement = metrics['STOI'] - gtcrn_metrics['STOI']
                
                result = {
                    'SNR_dB': snr_dB,
                    'Noise_Type': category,
                    'Participant': participant,
                    'Method': 'GTCRN+SS',
                    'Nband': nband,
                    'Floor': floor,
                    'PESQ': metrics['PESQ'],
                    'STOI': metrics['STOI'],
                    'SI_SDR': metrics['SI_SDR'],
                    'Weighted_Score': weighted_score,
                    'GTCRN_PESQ': gtcrn_metrics['PESQ'],
                    'GTCRN_STOI': gtcrn_metrics['STOI'],
                    'PESQ_improvement': pesq_improvement,
                    'STOI_improvement': stoi_improvement,
                    'Status': 'Success'
                }
                

            except Exception as e:
                print(f"    ERROR in config (Nband={nband}, Floor={floor}): {str(e)}")
                import traceback
                traceback.print_exc()
                result = {
                    'SNR_dB': snr_dB,
                    'Noise_Type': category,
                    'Participant': participant,
                    'Method': 'GTCRN+SS',
                    'Nband': nband,
                    'Floor': floor,
                    'PESQ': None,
                    'STOI': None,
                    'SI_SDR': None,
                    'Weighted_Score': None,
                    'GTCRN_PESQ': gtcrn_metrics['PESQ'],
                    'GTCRN_STOI': gtcrn_metrics['STOI'],
                    'PESQ_improvement': None,
                    'STOI_improvement': None,
                    'Status': f'Error: {str(e)}'
            }
            # except Exception as e:
            #     result = {
            #         'SNR_dB': snr_dB,
            #         'Noise_Type': category,
            #         'Participant': participant,
            #         'Method': 'GTCRN+SS',
            #         'Nband': nband,
            #         'Floor': floor,
            #         'PESQ': None,
            #         'STOI': None,
            #         'SI_SDR': None,
            #         'Weighted_Score': None,
            #         'GTCRN_PESQ': gtcrn_metrics['PESQ'],
            #         'GTCRN_STOI': gtcrn_metrics['STOI'],
            #         'PESQ_improvement': None,
            #         'STOI_improvement': None,
            #         'Status': f'Error: {str(e)}'
            #     }
            
            all_results.append(result)
            test_count += 1
            
            if test_count % 20 == 0:
                elapsed = time.time() - overall_start
                rate = test_count / elapsed
                remaining = (total_tests - test_count) / rate / 60
                progress = test_count / total_tests * 100
                print(f"  Progress: {test_count}/{total_tests} ({progress:.1f}%) - {remaining:.1f} min remaining")

# ====================================
# SAVE RESULTS
# ====================================
results_df = pd.DataFrame(all_results)
baseline_df = pd.DataFrame(baseline_results)

# Combine for full comparison
full_df = pd.concat([baseline_df, results_df], ignore_index=True)
full_df.to_csv(results_dir / 'gtcrn_ss_comparison_full.csv', index=False)

# Save just successful GTCRN+SS results
success_df = results_df[results_df['Status'] == 'Success'].copy()
success_df.to_csv(results_dir / 'gtcrn_ss_configurations.csv', index=False)

elapsed = time.time() - overall_start
print(f"\n{'='*80}")
print(f"SWEEP COMPLETE in {elapsed/60:.1f} minutes")
print(f"{'='*80}")

# ====================================
# ANALYSIS
# ====================================
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

# Overall improvement statistics
print("\n--- Overall Improvement Over GTCRN Baseline ---")
avg_pesq_imp = success_df['PESQ_improvement'].mean()
avg_stoi_imp = success_df['STOI_improvement'].mean()
print(f"Average PESQ improvement: {avg_pesq_imp:+.4f}")
print(f"Average STOI improvement: {avg_stoi_imp:+.4f}")

improved_pesq = (success_df['PESQ_improvement'] > 0).sum()
improved_stoi = (success_df['STOI_improvement'] > 0).sum()
total = len(success_df)
print(f"Configurations improving PESQ: {improved_pesq}/{total} ({improved_pesq/total*100:.1f}%)")
print(f"Configurations improving STOI: {improved_stoi}/{total} ({improved_stoi/total*100:.1f}%)")

# Best configurations by SNR
print("\n" + "="*80)
print("BEST CONFIGURATIONS BY SNR")
print("="*80)

for snr in SNR_LEVELS:
    df_snr = success_df[success_df['SNR_dB'] == snr]
    
    if len(df_snr) == 0:
        continue
    
    # Best by weighted score
    best = df_snr.loc[df_snr['Weighted_Score'].idxmax()]
    
    print(f"\n--- SNR = {snr} dB ---")
    print(f"Best configuration:")
    print(f"  Nband: {int(best['Nband'])}, Floor: {best['Floor']:.4f}")
    print(f"  GTCRN+SS: PESQ={best['PESQ']:.4f}, STOI={best['STOI']:.4f}")
    print(f"  GTCRN only: PESQ={best['GTCRN_PESQ']:.4f}, STOI={best['GTCRN_STOI']:.4f}")
    print(f"  Improvement: PESQ={best['PESQ_improvement']:+.4f}, STOI={best['STOI_improvement']:+.4f}")

# Best by noise type
print("\n" + "="*80)
print("BEST CONFIGURATIONS BY NOISE TYPE")
print("="*80)

for noise_type in success_df['Noise_Type'].unique():
    df_noise = success_df[success_df['Noise_Type'] == noise_type]
    
    # Best config for this noise
    best = df_noise.loc[df_noise['Weighted_Score'].idxmax()]
    
    print(f"\n--- {noise_type.upper()} ---")
    print(f"  Best: Nband={int(best['Nband'])}, Floor={best['Floor']:.4f}")
    print(f"  PESQ improvement: {best['PESQ_improvement']:+.4f}")
    print(f"  STOI improvement: {best['STOI_improvement']:+.4f}")

# Summary recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find most consistent good config
config_performance = success_df.groupby(['Nband', 'Floor']).agg({
    'PESQ_improvement': 'mean',
    'STOI_improvement': 'mean',
    'Weighted_Score': 'mean'
}).round(4)

best_config = config_performance.loc[config_performance['Weighted_Score'].idxmax()]
best_nband, best_floor = config_performance['Weighted_Score'].idxmax()

print(f"\nMost consistent configuration across all conditions:")
print(f"  Nband: {int(best_nband)}")
print(f"  Floor: {best_floor:.4f}")
print(f"  Average PESQ improvement: {best_config['PESQ_improvement']:+.4f}")
print(f"  Average STOI improvement: {best_config['STOI_improvement']:+.4f}")

if best_config['PESQ_improvement'] > 0 and best_config['STOI_improvement'] > 0:
    print("\n✓ SS post-processing IMPROVES GTCRN output!")
elif best_config['PESQ_improvement'] > 0:
    print("\n SS improves PESQ but may reduce STOI")
else:
    print("\n SS post-processing may not be beneficial")

print(f"\nAll results saved to: {results_dir}")
print("="*80)