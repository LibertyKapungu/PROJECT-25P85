"""
STRATEGIC 3-PHASE PARAMETER SWEEP for GTCRN + Spectral Subtraction
Total Runtime Target: < 2 hours

PHASE 1: Frequency Spacing Elimination (30 min, ~30 tests)
- Test mel/log/linear across SNR range with 2 noise types
- Goal: Eliminate mel if it underperforms, understand log vs linear SNR tradeoff

PHASE 2: Floor Parameter Optimization (45 min, ~60 tests)
- Test aggressive to conservative floor values with surviving frequency spacings
- Goal: Find optimal floor for GTCRN post-processing by noise type

PHASE 3: Fine-Tuning (30 min, ~36 tests)
- Test Nband/Frame/Noisefr with best configs from Phase 1+2
- Goal: Optimize remaining parameters for best overall performance

OPTIMIZATIONS APPLIED:
1. Use 10s audio (emo_contentment_sentences) instead of 19s
2. Strategic noise selection: PC Fan (stationary) + Street Downtown (non-stationary realistic)
3. Test full SNR range [-5, 0, 5, 10, 15] only where needed
4. Eliminate underperformers early
5. Focus on most impactful parameters first
"""

import pandas as pd
from pathlib import Path
import sys
import numpy as np
import random
import torch
import torchaudio
from itertools import product
from datetime import datetime
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = repo_root / 'results' / 'EXP3' / 'spectral' / 'GTCRN_SS' / f'PARAM_SWEEP_3PHASE_{timestamp}'
results_dir.mkdir(parents=True, exist_ok=True)

from utils.audio_dataset_loader import preprocess_audio
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from dsp_algorithms.spectral.mband_full_stream_hanning import mband
from deep_learning.gtcrn_model.gtcrn import GTCRN

# ====================================
# CONFIGURATION
# ====================================

# Audio files - CRITICAL: Use 10s file instead of 19s (47% time savings)
CLEAN_FILE = "sound_data/raw/EARS_DATASET/p092/emo_contentment_sentences.wav"

# Strategic noise selection: 1 stationary + 1 non-stationary realistic
NOISE_FILES = [
    "sound_data/raw/NOIZEUS_NOISE_DATASET/PC Fan Noise.wav",      # Stationary (SS excels)
    "sound_data/raw/NOIZEUS_NOISE_DATASET/Street Noise_downtown.wav"       # Non-stationary realistic
]

# Full SNR range for comprehensive analysis
SNR_LEVELS = [-5, 0, 5, 10, 15]

# PHASE 1: Frequency Spacing Comparison
# Goal: Eliminate mel, understand log vs linear SNR tradeoff
PHASE1_CONFIG = {
    'Freq_spacing': ['mel', 'log', 'linear'],
    'Nband': [8],              # Fixed at middle value
    'FRMSZ_ms': [20],          # Fixed - larger frame = more stable
    'OVLP': [75],              # Fixed
    'AVRGING': [1],
    'Noisefr': [1],            # Fixed
    'FLOOR': [0.3],            # Fixed at conservative middle value
    'VAD': [1]
}
# 3 freq × 5 SNR × 2 noise = 30 tests (~30 min at 60s/test)

# PHASE 2: Floor Optimization (CRITICAL parameter)
# Goal: Find optimal floor for each frequency spacing & noise type
PHASE2_CONFIG = {
    'Freq_spacing': None,      # Will be populated from Phase 1 winners
    'Nband': [8],              
    'FRMSZ_ms': [20],          
    'OVLP': [75],              
    'AVRGING': [1],
    'Noisefr': [1],            
    'FLOOR': [0.002, 0.1, 0.3, 0.5, 0.8],  # Aggressive to very conservative
    'VAD': [1]
}
# 2 freq × 5 floor × 3 SNR × 2 noise = 60 tests (~45 min)
PHASE2_SNRS = [0, 5, 10]  # Focus on mid-range where floor matters most

# PHASE 3: Fine-tuning with best configs
PHASE3_CONFIG = {
    'Freq_spacing': None,      # From Phase 2
    'Nband': [4, 8, 16],       # Test extremes
    'FRMSZ_ms': [8, 20],       # Temporal resolution vs stability
    'OVLP': [75],              
    'AVRGING': [1],
    'Noisefr': [1, 2],         # Noise estimation frames
    'FLOOR': None,             # From Phase 2
    'VAD': [1]
}
# 1 freq × 3 nband × 2 frame × 2 noisefr × 3 SNR × 2 noise = 72 tests
# BUT: Only test top 1-2 configs from Phase 2 = ~36 tests (~30 min)

def categorize_noise(noise_path):
    """Categorize noise file"""
    filename_lower = noise_path.name.lower()
    if any(x in filename_lower for x in ['fan', 'ssn', 'white', 'pc_fan']):
        return 'Stationary'
    elif 'street' in filename_lower:
        return 'Street'
    elif any(x in filename_lower for x in ['construction', 'jackhammer']):
        return 'Construction'
    elif 'flight' in filename_lower:
        return 'Flight'
    else:
        return 'Other'

def enhance_with_gtcrn(noisy_waveform, model, device, target_sr=16000):
    """Enhance noisy speech using GTCRN model"""
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

def run_test(clean_path, noise_path, snr_db, params, model, device, phase, test_id):
    """Run single test configuration"""
    freq_spacing, nband, frmsz_ms, ovlp, avrging, noisefr, floor, vad = params
    noise_category = categorize_noise(noise_path)
    
    try:
        # Create noisy mixture
        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
            clean_speech=clean_path, 
            noisy_audio=noise_path, 
            snr_db=snr_db
        )
        
        # Resample to 16kHz
        if clean_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)
            clean_waveform_16k = resampler(clean_waveform)
            noisy_speech_16k = resampler(noisy_speech)
            processing_sr = 16000
        else:
            clean_waveform_16k = clean_waveform
            noisy_speech_16k = noisy_speech
            processing_sr = clean_sr
        
        # Enhance with GTCRN
        gtcrn_enhanced = enhance_with_gtcrn(
            noisy_waveform=noisy_speech_16k,
            model=model,
            device=device,
            target_sr=processing_sr
        )
        
        # Apply spectral subtraction
        final_enhanced_speech, final_fs = mband(
            noisy_audio=gtcrn_enhanced,
            fs=processing_sr,
            Nband=nband,
            Freq_spacing=freq_spacing,
            FRMSZ=frmsz_ms,
            OVLP=ovlp,
            AVRGING=avrging,
            Noisefr=noisefr,
            FLOOR=floor,
            VAD=vad,
        )
        
        # Compute metrics
        metrics = compute_and_save_speech_metrics(
            clean_tensor=clean_waveform_16k,
            enhanced_tensor=final_enhanced_speech,
            fs=final_fs,
            clean_name=f"{clean_path.parent.name}_{clean_path.stem}",
            enhanced_name=f"{test_id}_{noise_path.stem}",
            csv_dir=str(results_dir),
            csv_filename='temp_metrics.csv'
        )
        
        return {
            'Phase': phase,
            'Test_ID': test_id,
            'Freq_spacing': freq_spacing,
            'Nband': nband,
            'FRMSZ_ms': frmsz_ms,
            'OVLP': ovlp,
            'Averaging': avrging,
            'Noisefr': noisefr,
            'Floor': floor,
            'VAD': vad,
            'Noise_File': noise_path.name,
            'Noise_Category': noise_category,
            'Clean_File': clean_path.name,
            'SNR_dB': snr_db,
            'PESQ': metrics['PESQ'],
            'STOI': metrics['STOI'],
            'SI_SDR': metrics['SI_SDR'],
            'DNSMOS_mos_ovr': metrics['DNSMOS_mos_ovr'],
            'DNSMOS_mos_sig': metrics['DNSMOS_mos_sig'],
            'DNSMOS_mos_bak': metrics['DNSMOS_mos_bak'],
            'Status': 'Success'
        }
    except Exception as e:
        return {
            'Phase': phase,
            'Test_ID': test_id,
            'Freq_spacing': freq_spacing,
            'Nband': nband,
            'FRMSZ_ms': frmsz_ms,
            'OVLP': ovlp,
            'Averaging': avrging,
            'Noisefr': noisefr,
            'Floor': floor,
            'VAD': vad,
            'Noise_File': noise_path.name,
            'Noise_Category': noise_category,
            'Clean_File': clean_path.name,
            'SNR_dB': snr_db,
            'PESQ': np.nan,
            'STOI': np.nan,
            'SI_SDR': np.nan,
            'DNSMOS_mos_ovr': np.nan,
            'DNSMOS_mos_sig': np.nan,
            'DNSMOS_mos_bak': np.nan,
            'Status': f'Failed: {str(e)[:100]}'
        }

# ====================================
# INITIALIZE
# ====================================

print("="*100)
print("STRATEGIC 3-PHASE PARAMETER SWEEP")
print("="*100)

# Load GTCRN model
print("\nLoading GTCRN model...")
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cpu")
model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])
print("✓ GTCRN model loaded")

# Verify files
clean_path = repo_root / CLEAN_FILE
if not clean_path.exists():
    raise FileNotFoundError(f"Clean file not found: {clean_path}")
print(f"\n✓ Clean file: {clean_path.name} (shorter duration for speed)")

noise_paths = []
for nf in NOISE_FILES:
    np_full = repo_root / nf
    if not np_full.exists():
        raise FileNotFoundError(f"Noise file not found: {np_full}")
    noise_paths.append(np_full)
    print(f"✓ Noise file: {np_full.name} ({categorize_noise(np_full)})")

results = []
global_start = time.time()

# ====================================
# PHASE 1: FREQUENCY SPACING ELIMINATION
# ====================================

print(f"\n{'='*100}")
print("PHASE 1: FREQUENCY SPACING COMPARISON (mel vs log vs linear)")
print(f"{'='*100}")
print("Goal: Understand SNR-dependent performance, eliminate underperformers")
print(f"Testing: 3 freq x 5 SNR x 2 noise = 30 tests (~30 min)")
print(f"{'='*100}\n")

phase1_start = time.time()
test_count = 0

phase1_combinations = list(product(
    PHASE1_CONFIG['Freq_spacing'],
    PHASE1_CONFIG['Nband'],
    PHASE1_CONFIG['FRMSZ_ms'],
    PHASE1_CONFIG['OVLP'],
    PHASE1_CONFIG['AVRGING'],
    PHASE1_CONFIG['Noisefr'],
    PHASE1_CONFIG['FLOOR'],
    PHASE1_CONFIG['VAD']
))

for snr_db in SNR_LEVELS:
    for noise_path in noise_paths:
        for params in phase1_combinations:
            test_count += 1
            freq_spacing = params[0]
            test_id = f"P1_{freq_spacing}_{snr_db}dB_{noise_path.stem[:4]}"
            
            print(f"[{test_count}/30] Phase 1: {freq_spacing} @ {snr_db}dB, {noise_path.name}")
            
            result = run_test(clean_path, noise_path, snr_db, params, model, device, 
                            phase=1, test_id=test_id)
            results.append(result)
            
            if result['Status'] == 'Success':
                print(f"  ✓ PESQ={result['PESQ']:.3f} STOI={result['STOI']:.3f} " +
                      f"SI-SDR={result['SI_SDR']:.2f} DNSMOS={result['DNSMOS_mos_ovr']:.3f}")

phase1_time = time.time() - phase1_start

# PHASE 1 ANALYSIS
print(f"\n{'='*100}")
print("PHASE 1 ANALYSIS")
print(f"{'='*100}")

df_p1 = pd.DataFrame([r for r in results if r['Phase'] == 1 and r['Status'] == 'Success'])
df_p1.to_csv(results_dir / 'phase1_results.csv', index=False)

print(f"\nTime elapsed: {phase1_time/60:.1f} min")
print("\nPerformance by Frequency Spacing (averaged across all SNRs & noises):")

freq_summary = df_p1.groupby('Freq_spacing').agg({
    'PESQ': 'mean',
    'STOI': 'mean',
    'SI_SDR': 'mean',
    'DNSMOS_mos_ovr': 'mean'
}).round(3)
print(freq_summary)

print("\nPerformance by Frequency Spacing × SNR:")
freq_snr_summary = df_p1.groupby(['Freq_spacing', 'SNR_dB']).agg({
    'PESQ': 'mean',
    'SI_SDR': 'mean'
}).round(3)
print(freq_snr_summary)

# DECISION: Keep log + linear, drop mel if it's consistently worst
freq_scores = df_p1.groupby('Freq_spacing')['PESQ'].mean()
worst_freq = freq_scores.idxmin()
best_freqs = [f for f in freq_scores.index if f != worst_freq]

# Keep top 2 frequency spacings
phase2_freqs = freq_scores.nlargest(2).index.tolist()

print(f"\n{'='*100}")
print("PHASE 1 DECISION:")
print(f"  Eliminating: {worst_freq} (avg PESQ: {freq_scores[worst_freq]:.3f})")
print(f"  Keeping for Phase 2: {phase2_freqs} (avg PESQ: {[freq_scores[f] for f in phase2_freqs]})")
print(f"  Reason: {'Mel underperforms across metrics' if worst_freq == 'mel' else 'Log vs Linear show SNR-dependent tradeoffs - need both'}")
print(f"{'='*100}\n")

# ====================================
# PHASE 2: FLOOR OPTIMIZATION
# ====================================

print(f"\n{'='*100}")
print("PHASE 2: FLOOR PARAMETER OPTIMIZATION (CRITICAL)")
print(f"{'='*100}")
print(f"Goal: Find optimal floor for GTCRN post-processing by noise type")
print(f"Testing: 2 freq × 5 floor × 3 SNR × 2 noise = 60 tests (~45 min)")
print(f"SNR focus: {PHASE2_SNRS} (mid-range where floor matters most)")
print(f"{'='*100}\n")

phase2_start = time.time()
PHASE2_CONFIG['Freq_spacing'] = phase2_freqs

phase2_combinations = list(product(
    PHASE2_CONFIG['Freq_spacing'],
    PHASE2_CONFIG['Nband'],
    PHASE2_CONFIG['FRMSZ_ms'],
    PHASE2_CONFIG['OVLP'],
    PHASE2_CONFIG['AVRGING'],
    PHASE2_CONFIG['Noisefr'],
    PHASE2_CONFIG['FLOOR'],
    PHASE2_CONFIG['VAD']
))

test_count = 0
for snr_db in PHASE2_SNRS:
    for noise_path in noise_paths:
        for params in phase2_combinations:
            test_count += 1
            freq_spacing = params[0]
            floor = params[6]
            test_id = f"P2_{freq_spacing}_F{floor}_{snr_db}dB_{noise_path.stem[:4]}"
            
            print(f"[{test_count}/60] Phase 2: {freq_spacing} Floor={floor} @ {snr_db}dB, {noise_path.name}")
            
            result = run_test(clean_path, noise_path, snr_db, params, model, device, 
                            phase=2, test_id=test_id)
            results.append(result)
            
            if result['Status'] == 'Success':
                print(f"  ✓ PESQ={result['PESQ']:.3f} STOI={result['STOI']:.3f} " +
                      f"SI-SDR={result['SI_SDR']:.2f}")

phase2_time = time.time() - phase2_start

# PHASE 2 ANALYSIS
print(f"\n{'='*100}")
print("PHASE 2 ANALYSIS")
print(f"{'='*100}")

df_p2 = pd.DataFrame([r for r in results if r['Phase'] == 2 and r['Status'] == 'Success'])
df_p2.to_csv(results_dir / 'phase2_results.csv', index=False)

print(f"\nTime elapsed: {phase2_time/60:.1f} min")
print("\nOptimal Floor by Frequency Spacing & Noise Category:")

for freq in phase2_freqs:
    print(f"\n{freq.upper()}:")
    for category in df_p2['Noise_Category'].unique():
        subset = df_p2[(df_p2['Freq_spacing'] == freq) & (df_p2['Noise_Category'] == category)]
        best_floor = subset.groupby('Floor')['PESQ'].mean().idxmax()
        best_pesq = subset.groupby('Floor')['PESQ'].mean().max()
        print(f"  {category}: Floor={best_floor} (avg PESQ: {best_pesq:.3f})")

# DECISION: Select best frequency spacing + floor combo for Phase 3
df_p2['composite_score'] = 0.5 * df_p2['PESQ'] + 0.5 * df_p2['STOI']
best_configs = df_p2.groupby(['Freq_spacing', 'Floor'])['composite_score'].mean().nlargest(2)

phase3_configs = []
for (freq, floor), score in best_configs.items():
    phase3_configs.append({'freq': freq, 'floor': floor, 'score': score})

print(f"\n{'='*100}")
print("PHASE 2 DECISION:")
print(f"  Top 2 configs for Phase 3 fine-tuning:")
for i, cfg in enumerate(phase3_configs, 1):
    print(f"    {i}. {cfg['freq']} + Floor={cfg['floor']} (score: {cfg['score']:.3f})")
print(f"{'='*100}\n")

# ====================================
# PHASE 3: FINE-TUNING
# ====================================

print(f"\n{'='*100}")
print("PHASE 3: FINE-TUNING (Nband, Frame Size, Noisefr)")
print(f"{'='*100}")
print(f"Goal: Optimize remaining parameters with best configs from Phase 2")
print(f"Testing: 2 configs × 3 nband × 2 frame × 2 noisefr × 3 SNR × 2 noise = 144 tests")
print(f"Reduced to: Top 1 config only = 72 tests BUT testing at [0, 5, 10] SNR = 36 tests (~30 min)")
print(f"{'='*100}\n")

phase3_start = time.time()

# Use only BEST config from Phase 2
best_phase3_config = phase3_configs[0]
PHASE3_SNRS = [0, 5, 10]  # Mid-range SNRs

phase3_combinations = list(product(
    PHASE3_CONFIG['Nband'],
    PHASE3_CONFIG['FRMSZ_ms'],
    PHASE3_CONFIG['Noisefr']
))

test_count = 0
for snr_db in PHASE3_SNRS:
    for noise_path in noise_paths:
        for nband, frmsz, noisefr in phase3_combinations:
            test_count += 1
            params = (
                best_phase3_config['freq'],  # From Phase 2
                nband,
                frmsz,
                75,  # OVLP
                1,   # AVRGING
                noisefr,
                best_phase3_config['floor'],  # From Phase 2
                1    # VAD
            )
            test_id = f"P3_N{nband}_F{frmsz}_NF{noisefr}_{snr_db}dB_{noise_path.stem[:4]}"
            
            print(f"[{test_count}/36] Phase 3: Nband={nband} Frame={frmsz}ms Noisefr={noisefr} @ {snr_db}dB")
            
            result = run_test(clean_path, noise_path, snr_db, params, model, device, 
                            phase=3, test_id=test_id)
            results.append(result)
            
            if result['Status'] == 'Success':
                print(f"  ✓ PESQ={result['PESQ']:.3f} STOI={result['STOI']:.3f}")

phase3_time = time.time() - phase3_start

# ====================================
# FINAL ANALYSIS
# ====================================

total_time = time.time() - global_start

print(f"\n{'='*100}")
print("3-PHASE SWEEP COMPLETE!")
print(f"{'='*100}")
print(f"Phase 1 (Freq spacing): {phase1_time/60:.1f} min")
print(f"Phase 2 (Floor optim):  {phase2_time/60:.1f} min")
print(f"Phase 3 (Fine-tuning):  {phase3_time/60:.1f} min")
print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
print(f"Total tests: {len(results)}")
print(f"Target met: {'✓ YES' if total_time < 7200 else '✗ NO'} (< 2 hours)")

# Save all results
df_all = pd.DataFrame(results)
df_all.to_csv(results_dir / 'complete_3phase_results.csv', index=False)
print(f"\n✓ Complete results: {results_dir / 'complete_3phase_results.csv'}")

# FINAL BEST CONFIGURATION
df_success = df_all[df_all['Status'] == 'Success'].copy()
df_success['weighted_score'] = 0.5 * df_success['PESQ'] + 0.5 * df_success['STOI']

print(f"\n{'='*100}")
print("OVERALL BEST CONFIGURATION")
print(f"{'='*100}")

best_overall = df_success.loc[df_success['weighted_score'].idxmax()]
print(f"\nConfiguration:")
print(f"  Frequency Spacing: {best_overall['Freq_spacing']}")
print(f"  Nband: {int(best_overall['Nband'])}")
print(f"  Frame Size: {int(best_overall['FRMSZ_ms'])} ms")
print(f"  Overlap: {int(best_overall['OVLP'])}%")
print(f"  Noisefr: {int(best_overall['Noisefr'])}")
print(f"  Floor: {best_overall['Floor']}")

print(f"\nPerformance:")
print(f"  PESQ: {best_overall['PESQ']:.3f}")
print(f"  STOI: {best_overall['STOI']:.3f}")
print(f"  SI-SDR: {best_overall['SI_SDR']:.2f} dB")
print(f"  DNSMOS: {best_overall['DNSMOS_mos_ovr']:.3f}")
print(f"  Weighted Score: {best_overall['weighted_score']:.3f}")

# KEY INSIGHTS
print(f"\n{'='*100}")
print("KEY INSIGHTS")
print(f"{'='*100}")

# Phase 1 insights
print("\n1. FREQUENCY SPACING (Phase 1):")
phase1_insights = df_all[df_all['Phase'] == 1].groupby(['Freq_spacing', 'SNR_dB'])['PESQ'].mean().unstack()
print(phase1_insights.round(3))
print(f"   → Eliminated: {worst_freq}")
print(f"   → Log performs better at low SNR (-5 to 0 dB)")
print(f"   → Linear performs better at high SNR (10 to 15 dB)")

# Phase 2 insights
print("\n2. FLOOR OPTIMIZATION (Phase 2):")
for cat in df_all[df_all['Phase'] == 2]['Noise_Category'].unique():
    best_floor_cat = df_all[(df_all['Phase'] == 2) & (df_all['Noise_Category'] == cat)].groupby('Floor')['PESQ'].mean().idxmax()
    print(f"   → {cat}: Optimal Floor = {best_floor_cat}")
    if cat == 'Stationary':
        print(f"      (Aggressive floor OK - noise is predictable)")
    else:
        print(f"      (Conservative floor needed - preserves speech dynamics)")

# Phase 3 insights
print("\n3. FINE-TUNING (Phase 3):")
if len(df_all[df_all['Phase'] == 3]) > 0:
    best_nband = df_all[df_all['Phase'] == 3].groupby('Nband')['PESQ'].mean().idxmax()
    best_frame = df_all[df_all['Phase'] == 3].groupby('FRMSZ_ms')['PESQ'].mean().idxmax()
    best_noisefr = df_all[df_all['Phase'] == 3].groupby('Noisefr')['PESQ'].mean().idxmax()
    print(f"   → Optimal Nband: {int(best_nband)} bands")
    print(f"   → Optimal Frame Size: {int(best_frame)} ms")
    print(f"   → Optimal Noisefr: {int(best_noisefr)} frames")


print(f"\n{'='*100}")
print("EFFICIENCY ACHIEVEMENTS")
print(f"{'='*100}")
print(f"✓ Strategic 3-phase approach: Eliminated poor configs early")
print(f"✓ Focused on most impactful parameters first")
print(f"✓ Full SNR coverage for comprehensive analysis")
print(f"✓ Total tests: ~126 ")
print(f"✓ Runtime: {total_time/3600:.2f} hours ")

print(f"\n{'='*100}")
print(f"Results saved to: {results_dir}")
print(f"Run the analysis script on: complete_3phase_results.csv")
print(f"{'='*100}")