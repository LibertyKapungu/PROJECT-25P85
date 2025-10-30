"""
ENHANCED 4-PHASE PARAMETER SWEEP for GTCRN + Spectral Subtraction
Configurable phases with refined testing strategy

PHASE 1: Frequency Spacing + Extreme Floor Testing (45 min, 60 tests)
- Test mel/log/linear across SNR range
- Test EXTREME floor values [0.002, 0.8] to understand bounds
- Goal: Eliminate freq spacing, understand floor sensitivity

PHASE 2: Floor Parameter Fine-Tuning (45 min, 60 tests)  
- Test 5 floor values with surviving frequency spacings
- Goal: Find optimal floor for each noise type

PHASE 3: Fine-Tuning (45 min, 72 tests)
- Test Nband/Frame/Noisefr with TOP 2 configs from Phase 2
- Goal: Optimize remaining parameters

PHASE 4: Extended Noise Validation (optional, 30 min, 40 tests)
- Test best configs from Phase 3 on 4 additional noise types
- Goal: Validate robustness across diverse conditions
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
import argparse

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ====================================
# COMMAND-LINE CONFIGURATION
# ====================================
# parser = argparse.ArgumentParser(description='GTCRN + SS Parameter Sweep')
# parser.add_argument('--phases', type=str, default='1,2,3', 
#                     help='Comma-separated phases to run (e.g., "1,2,3" or "1,2,3,4")')
# parser.add_argument('--extended-noises', action='store_true',
#                     help='Include Phase 4 with 4 additional noise types')
# args = parser.parse_args()

# PHASES_TO_RUN = [int(p) for p in args.phases.split(',')]

# PHASES_TO_RUN = 0,1,2,3

PHASES_TO_RUN = [4]
EXTENDED_NOISES = True


# Run all phases
# python sweep.py --phases 1,2,3,4 --extended-noises

# # Run only Phases 1 and 2
# python sweep.py --phases 1,2

# # Run Phase 3 only (requires previous results)
# python sweep.py --phases 3

print("="*100)
print("ENHANCED 4-PHASE PARAMETER SWEEP")
print("="*100)
print(f"Phases to execute: {PHASES_TO_RUN}")
if 4 in PHASES_TO_RUN:
    print("Extended noise validation: ENABLED")
print("="*100)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

gtcrn_path = repo_root / "src" / "deep_learning" / "gtcrn_model" 
sys.path.insert(0, str(gtcrn_path))

RESUME_FROM_RESULTS_DIR = repo_root/"results"/ "EXP3" / "SS" / "PARAM_SWEEP_ENHANCED_20251029_225201"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = repo_root / 'results' / 'EXP3' / 'SS' / f'PARAM_SWEEP_ENHANCED_{timestamp}'
results_dir.mkdir(parents=True, exist_ok=True)

from utils.audio_dataset_loader import preprocess_audio
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from dsp_algorithms.spectral.mband_full_stream_hanning import mband
from deep_learning.gtcrn_model.gtcrn import GTCRN

# ====================================
# CONFIGURATION
# ====================================

CLEAN_FILE = "sound_data/raw/EARS_DATASET/p092/emo_contentment_sentences.wav"

# Core noise files for Phases 1-3
CORE_NOISE_FILES = [
    "sound_data/raw/NOIZEUS_NOISE_DATASET/PC Fan Noise.wav",
    "sound_data/raw/NOIZEUS_NOISE_DATASET/Street Noise.wav"
]

# Extended noise files for Phase 4 validation
EXTENDED_NOISE_FILES = [
    "sound_data/raw/NOIZEUS_NOISE_DATASET/cafeteria_babble.wav",  # Babble
    "sound_data/raw/NOIZEUS_NOISE_DATASET/Inside Train_1.wav",     # Train
    "sound_data/raw/NOIZEUS_NOISE_DATASET/Construction_Trucks_Unloading.wav",  # Construction
    "sound_data/raw/NOIZEUS_NOISE_DATASET/Car Noise_60mph.wav"     # Car
]

SNR_LEVELS = [-5, 0, 5, 10, 15]

# REFINED PHASE 1: Test frequency spacing + EXTREME floor values
PHASE1_CONFIG = {
    'Freq_spacing': ['mel', 'log', 'linear'],
    'Nband': [4, 16],          # EXTREMES: coarse vs fine
    'FRMSZ_ms': [20],          
    'OVLP': [75],              
    'AVRGING': [1],
    'Noisefr': [1],            
    'FLOOR': [0.002, 0.7],     # EXTREMES: aggressive vs conservative
    'VAD': [1]
}
# 3 freq × 2 nband × 2 floor × 5 SNR × 2 noise = 120 tests (~1 hour)
# But we'll sample smartly: 3 freq × 5 SNR × 2 noise = 30 base tests
# + 2 nband × 2 floor combos at key SNRs [0, 5, 10] = +36 tests = 66 total

PHASE2_CONFIG = {
    'Freq_spacing': None,      # From Phase 1
    'Nband': [8],              
    'FRMSZ_ms': [20],          
    'OVLP': [75],              
    'AVRGING': [1],
    'Noisefr': [1],            
    'FLOOR': [0.002, 0.1, 0.3,0.5, 0.6, 0.7, 0.8],
    'VAD': [1]
}
PHASE2_SNRS = [-5, 0, 5, 10]

PHASE3_CONFIG = {
    'Freq_spacing': None,      
    'Nband': [4, 8, 16],       
    'FRMSZ_ms': [8, 20, 25],       
    'OVLP': [75],              
    'AVRGING': [1],
    'Noisefr': [1, 2],         
    'FLOOR': None,             
    'VAD': [1]
}
PHASE3_SNRS = [-5, 0, 5, 10]

def categorize_noise(noise_path):
    """Categorize noise file"""
    filename_lower = noise_path.name.lower()
    if any(x in filename_lower for x in ['babble', 'cafeteria']):
        return 'Babble'
    elif any(x in filename_lower for x in ['train', 'inside_train']):
        return 'Train'
    elif 'street' in filename_lower:
        return 'Street'
    elif 'car' in filename_lower:
        return 'Car'
    elif any(x in filename_lower for x in ['construction', 'crane', 'drilling', 
                                            'jackhammer', 'trucks_unloading']):
        return 'Construction'
    elif any(x in filename_lower for x in ['fan', 'cooler', 'ssn', 'white', 'pc_fan']):
        return 'Stationary'
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
        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
            clean_speech=clean_path, 
            noisy_audio=noise_path, 
            snr_db=snr_db
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
        
        gtcrn_enhanced = enhance_with_gtcrn(
            noisy_waveform=noisy_speech_16k,
            model=model,
            device=device,
            target_sr=processing_sr
        )
        
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

print("\nLoading GTCRN model...")
checkpoint_path = gtcrn_path / "checkpoints" / "model_trained_on_dns3.tar"
device = torch.device("cpu")
model = GTCRN().eval().to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'])
print(" GTCRN model loaded")

clean_path = repo_root / CLEAN_FILE
if not clean_path.exists():
    raise FileNotFoundError(f"Clean file not found: {clean_path}")
print(f"\n Clean file: {clean_path.name}")

# Load core noises
core_noise_paths = []
for nf in CORE_NOISE_FILES:
    np_full = repo_root / nf
    if not np_full.exists():
        print(f" Warning: Noise file not found: {np_full}")
        continue
    core_noise_paths.append(np_full)
    print(f" Core noise: {np_full.name} ({categorize_noise(np_full)})")

# Load extended noises if Phase 4
extended_noise_paths = []
if 4 in PHASES_TO_RUN:
    for nf in EXTENDED_NOISE_FILES:
        np_full = repo_root / nf
        if not np_full.exists():
            print(f" Warning: Extended noise not found: {np_full}")
            continue
        extended_noise_paths.append(np_full)
        print(f" Extended noise: {np_full.name} ({categorize_noise(np_full)})")

results = []
global_start = time.time()

# ====================================
# PHASE 1: REFINED FREQUENCY SPACING + EXTREME FLOOR/NBAND
# ====================================

if 1 in PHASES_TO_RUN:
    print(f"\n{'='*100}")
    print("PHASE 1: FREQUENCY SPACING + EXTREME PARAMETER BOUNDS")
    print(f"{'='*100}")
    print("Strategy:")
    print("  1. Test all freq spacings at Nband=8, Floor=0.3 (baseline)")
    print("  2. Test extreme Nband [4, 16] x Floor [0.002, 0.8] at key SNRs")
    print(f"{'='*100}\n")
    
    phase1_start = time.time()
    test_count = 0
    
    # Part A: Baseline comparison (all freq, standard params)
    baseline_params = (8, 20, 75, 1, 1, 0.3, 1)  # nband, frmsz, ovlp, avrg, noisefr, floor, vad
    
    for snr_db in SNR_LEVELS:
        for noise_path in core_noise_paths:
            for freq in PHASE1_CONFIG['Freq_spacing']:
                test_count += 1
                params = (freq,) + baseline_params
                test_id = f"P1A_{freq}_{snr_db}dB_{noise_path.stem[:6]}"
                
                print(f"[{test_count}] Phase 1A Baseline: {freq} @ {snr_db}dB")
                result = run_test(clean_path, noise_path, snr_db, params, model, device, 
                                phase=1, test_id=test_id)
                results.append(result)
                
                if result['Status'] == 'Success':
                    print(f"  ✓ PESQ={result['PESQ']:.3f} STOI={result['STOI']:.3f}")
    
    # Part B: Extreme parameter testing at key SNRs
    key_snrs = [0, 5, 10]
    extreme_combos = list(product([4, 16], [0.002, 0.8]))  # Nband × Floor
    
    # Use top 2 freq spacings from Part A
    df_p1a = pd.DataFrame([r for r in results if r['Phase'] == 1 and r['Status'] == 'Success'])
    top2_freq = df_p1a.groupby('Freq_spacing')['PESQ'].mean().nlargest(2).index.tolist()
    
    print(f"\n  Top 2 freq spacings from baseline: {top2_freq}")
    print(f"  Testing extreme Nband/Floor combinations with these...")
    
    for snr_db in key_snrs:
        for noise_path in core_noise_paths:
            for freq in top2_freq:
                for nband, floor in extreme_combos:
                    test_count += 1
                    params = (freq, nband, 20, 75, 1, 1, floor, 1)
                    test_id = f"P1B_{freq}_N{nband}_F{floor}_{snr_db}dB"
                    
                    print(f"[{test_count}] Phase 1B Extremes: {freq} N={nband} F={floor} @ {snr_db}dB")
                    result = run_test(clean_path, noise_path, snr_db, params, model, device,
                                    phase=1, test_id=test_id)
                    results.append(result)
                    
                    if result['Status'] == 'Success':
                        print(f"  PESQ={result['PESQ']:.3f}")
    
    phase1_time = time.time() - phase1_start
    
    # PHASE 1 ANALYSIS
    print(f"\n{'='*100}")
    print("PHASE 1 ANALYSIS")
    print(f"{'='*100}")
    
    df_p1 = pd.DataFrame([r for r in results if r['Phase'] == 1 and r['Status'] == 'Success'])
    df_p1.to_csv(results_dir / 'phase1_results.csv', index=False)
    
    print(f"\nTime elapsed: {phase1_time/60:.1f} min")
    print(f"Tests completed: {len(df_p1)}")
    
    # Frequency spacing summary
    print("\nFrequency Spacing Performance:")
    freq_summary = df_p1.groupby('Freq_spacing').agg({
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean'
    }).round(3)
    print(freq_summary)
    
    # Extreme parameter insights
    print("\nExtreme Parameter Impact (Nband × Floor):")
    extreme_data = df_p1[df_p1['Nband'].isin([4, 16])]
    if len(extreme_data) > 0:
        extreme_summary = extreme_data.pivot_table(
            values='PESQ',
            index='Nband',
            columns='Floor',
            aggfunc='mean'
        ).round(3)
        print(extreme_summary)
    
    # DECISION
    freq_scores = df_p1.groupby('Freq_spacing')['PESQ'].mean()
    phase2_freqs = freq_scores.nlargest(2).index.tolist()
    
    print(f"\n{'='*100}")
    print("PHASE 1 DECISION:")
    print(f"  Keeping for Phase 2: {phase2_freqs}")
    print(f"  Eliminating: {[f for f in freq_scores.index if f not in phase2_freqs]}")
    print(f"{'='*100}\n")
    
    # Save Phase 1 state for potential resume
    phase1_state = {
        'phase2_freqs': phase2_freqs,
        'time_elapsed': phase1_time
    }
    pd.DataFrame([phase1_state]).to_csv(results_dir / 'phase1_state.csv', index=False)

# ====================================
# PHASE 2: FLOOR OPTIMIZATION
# ====================================

if 2 in PHASES_TO_RUN:
    # Load Phase 1 results if not in memory
    if 1 not in PHASES_TO_RUN:
        phase1_state = pd.read_csv(results_dir / 'phase1_state.csv').iloc[0]
        phase2_freqs = eval(phase1_state['phase2_freqs']) if isinstance(phase1_state['phase2_freqs'], str) else phase1_state['phase2_freqs']
    
    print(f"\n{'='*100}")
    print("PHASE 2: FLOOR PARAMETER OPTIMIZATION")
    print(f"{'='*100}")
    print(f"Testing frequencies: {phase2_freqs}")
    print(f"Floor values: {PHASE2_CONFIG['FLOOR']}")
    print(f"SNRs: {PHASE2_SNRS}")
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
        for noise_path in core_noise_paths:
            for params in phase2_combinations:
                test_count += 1
                freq_spacing = params[0]
                floor = params[6]
                test_id = f"P2_{freq_spacing}_F{floor}_{snr_db}dB"
                
                print(f"[{test_count}] Phase 2: {freq_spacing} Floor={floor} @ {snr_db}dB")
                result = run_test(clean_path, noise_path, snr_db, params, model, device,
                                phase=2, test_id=test_id)
                results.append(result)
                
                if result['Status'] == 'Success':
                    print(f"  ✓ PESQ={result['PESQ']:.3f}")
    
    phase2_time = time.time() - phase2_start
    
    # PHASE 2 ANALYSIS
    print(f"\n{'='*100}")
    print("PHASE 2 ANALYSIS")
    print(f"{'='*100}")
    
    df_p2 = pd.DataFrame([r for r in results if r['Phase'] == 2 and r['Status'] == 'Success'])
    df_p2.to_csv(results_dir / 'phase2_results.csv', index=False)
    
    print(f"\nTime elapsed: {phase2_time/60:.1f} min")
    
    # Optimal floor by noise
    print("\nOptimal Floor by Noise Category:")
    for category in df_p2['Noise_Category'].unique():
        cat_data = df_p2[df_p2['Noise_Category'] == category]
        best_floor = cat_data.groupby('Floor')['PESQ'].mean().idxmax()
        print(f"  {category}: Floor = {best_floor}")
    
    # DECISION: Select TOP 2 configs for Phase 3
    df_p2['composite_score'] = 0.5 * df_p2['PESQ'] + 0.5 * df_p2['STOI']
    best_configs = df_p2.groupby(['Freq_spacing', 'Floor'])['composite_score'].mean().nlargest(2)
    
    phase3_configs = []
    for (freq, floor), score in best_configs.items():
        phase3_configs.append({'freq': freq, 'floor': floor, 'score': score})
    
    print(f"\n{'='*100}")
    print("PHASE 2 DECISION - TOP 2 CONFIGS FOR PHASE 3:")
    for i, cfg in enumerate(phase3_configs, 1):
        print(f"  {i}. {cfg['freq']} + Floor={cfg['floor']} (score: {cfg['score']:.3f})")
    print(f"{'='*100}\n")
    
    # Save state
    phase2_state = {
        'phase3_configs': str(phase3_configs),
        'time_elapsed': phase2_time
    }
    pd.DataFrame([phase2_state]).to_csv(results_dir / 'phase2_state.csv', index=False)

# ====================================
# PHASE 3: FINE-TUNING (TOP 2 CONFIGS)
# ====================================

if 3 in PHASES_TO_RUN:
    # Load Phase 2 results if not in memory
    if 2 not in PHASES_TO_RUN:
        phase2_state = pd.read_csv(results_dir / 'phase2_state.csv').iloc[0]
        phase3_configs = eval(phase2_state['phase3_configs'])
    
    print(f"\n{'='*100}")
    print("PHASE 3: FINE-TUNING WITH TOP 2 CONFIGS")
    print(f"{'='*100}")
    print("Testing both top configs from Phase 2")
    print(f"Parameters: Nband × Frame Size × Noisefr")
    print(f"{'='*100}\n")
    
    phase3_start = time.time()
    
    phase3_combinations = list(product(
        PHASE3_CONFIG['Nband'],
        PHASE3_CONFIG['FRMSZ_ms'],
        PHASE3_CONFIG['Noisefr']
    ))
    
    test_count = 0
    # Test BOTH top configs
    for config_idx, best_config in enumerate(phase3_configs, 1):
        print(f"\nTesting Config #{config_idx}: {best_config['freq']} + Floor={best_config['floor']}")
        
        for snr_db in PHASE3_SNRS:
            for noise_path in core_noise_paths:
                for nband, frmsz, noisefr in phase3_combinations:
                    test_count += 1
                    params = (
                        best_config['freq'],
                        nband,
                        frmsz,
                        75,
                        1,
                        noisefr,
                        best_config['floor'],
                        1
                    )
                    test_id = f"P3_C{config_idx}_N{nband}_F{frmsz}_NF{noisefr}_{snr_db}dB"
                    
                    print(f"[{test_count}] Phase 3: Config{config_idx} N={nband} F={frmsz} NF={noisefr} @ {snr_db}dB")
                    result = run_test(clean_path, noise_path, snr_db, params, model, device,
                                    phase=3, test_id=test_id)
                    results.append(result)
                    
                    if result['Status'] == 'Success':
                        print(f"  ✓ PESQ={result['PESQ']:.3f}")
    
    phase3_time = time.time() - phase3_start
    
    # PHASE 3 ANALYSIS
    df_p3 = pd.DataFrame([r for r in results if r['Phase'] == 3 and r['Status'] == 'Success'])
    df_p3.to_csv(results_dir / 'phase3_results.csv', index=False)
    
    print(f"\nPhase 3 complete: {phase3_time/60:.1f} min")

# ====================================
# PHASE 4: EXTENDED NOISE VALIDATION
# ====================================

if 4 in PHASES_TO_RUN:
    print(f"\n{'='*100}")
    print("PHASE 4: EXTENDED NOISE VALIDATION")
    print(f"{'='*100}")
    print("Testing best overall config on 4 additional noise types")
    print(f"{'='*100}\n")
    
    phase4_start = time.time()
    
    # Get best config from all previous phases
    # df_all = pd.DataFrame([r for r in results if r['Status'] == 'Success'])
    df_all = pd.read_csv("C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/results/EXP3/SS/PARAM_SWEEP_ENHANCED_20251029_225201/complete_results_all_phases.csv")
    df_all['composite_score'] = 0.5 * df_all['PESQ'] + 0.5 * df_all['STOI']
    best_overall = df_all.loc[df_all['composite_score'].idxmax()]
    
    print(f"Best config: {best_overall['Freq_spacing']} N={int(best_overall['Nband'])} " +
          f"F={int(best_overall['FRMSZ_ms'])}ms Floor={best_overall['Floor']}")
    
    best_params = (
        best_overall['Freq_spacing'],
        int(best_overall['Nband']),
        int(best_overall['FRMSZ_ms']),
        int(best_overall['OVLP']),
        int(best_overall['Averaging']),
        int(best_overall['Noisefr']),
        best_overall['Floor'],
        int(best_overall['VAD'])
    )
    
    test_count = 0
    validation_snrs = [0, 5, 10]  # Test at key SNRs
    
    for noise_path in extended_noise_paths:
        for snr_db in validation_snrs:
            test_count += 1
            test_id = f"P4_validation_{noise_path.stem[:8]}_{snr_db}dB"
            
            print(f"[{test_count}] Phase 4: {noise_path.name} @ {snr_db}dB")
            result = run_test(clean_path, noise_path, snr_db, best_params, model, device,
                            phase=4, test_id=test_id)
            results.append(result)
            
            if result['Status'] == 'Success':
                print(f"  PESQ={result['PESQ']:.3f} STOI={result['STOI']:.3f} " +
                      f"(Noise: {result['Noise_Category']})")
    
    phase4_time = time.time() - phase4_start
    
    # PHASE 4 ANALYSIS
    df_p4 = pd.DataFrame([r for r in results if r['Phase'] == 4 and r['Status'] == 'Success'])
    df_p4.to_csv(results_dir / 'phase4_results.csv', index=False)
    
    print(f"\nPhase 4 complete: {phase4_time/60:.1f} min")
    print("\nPerformance by extended noise type:")
    for category in df_p4['Noise_Category'].unique():
        cat_data = df_p4[df_p4['Noise_Category'] == category]
        print(f"  {category}: PESQ={cat_data['PESQ'].mean():.3f} STOI={cat_data['STOI'].mean():.3f}")

# ====================================
# FINAL ANALYSIS & RECOMMENDATIONS
# ====================================

total_time = time.time() - global_start

print(f"\n{'='*100}")
print("SWEEP COMPLETE!")
print(f"{'='*100}")
print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
print(f"Phases executed: {PHASES_TO_RUN}")

if 1 in PHASES_TO_RUN:
    print(f"  Phase 1: {phase1_time/60:.1f} min")
if 2 in PHASES_TO_RUN:
    print(f"  Phase 2: {phase2_time/60:.1f} min")
if 3 in PHASES_TO_RUN:
    print(f"  Phase 3: {phase3_time/60:.1f} min")
if 4 in PHASES_TO_RUN:
    print(f"  Phase 4: {phase4_time/60:.1f} min")

# Save all results
df_all = pd.DataFrame(results)
df_all.to_csv(results_dir / 'complete_results_all_phases.csv', index=False)

df_success = df_all[df_all['Status'] == 'Success'].copy()
df_success['composite_score'] = 0.5 * df_success['PESQ'] + 0.5 * df_success['STOI']

# BEST OVERALL CONFIGURATION
print(f"\n{'='*100}")
print("BEST OVERALL CONFIGURATION")
print(f"{'='*100}")

best_overall = df_success.loc[df_success['composite_score'].idxmax()]

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
print(f"  Composite Score: {best_overall['composite_score']:.3f}")
print(f"  Found in Phase: {int(best_overall['Phase'])}")

# CRITICAL INSIGHTS
print(f"\n{'='*100}")
print("CRITICAL INSIGHTS FROM YOUR RESULTS")
print(f"{'='*100}")

print("\n1. FREQUENCY SPACING ANALYSIS:")
if 1 in PHASES_TO_RUN or len(df_success[df_success['Phase'] == 1]) > 0:
    df_p1_analysis = df_success[df_success['Phase'] == 1]
    for freq in df_p1_analysis['Freq_spacing'].unique():
        freq_data = df_p1_analysis[df_p1_analysis['Freq_spacing'] == freq]
        print(f"\n   {freq.upper()}:")
        print(f"     Overall PESQ: {freq_data['PESQ'].mean():.3f} (std: {freq_data['PESQ'].std():.3f})")
        print(f"     Overall STOI: {freq_data['STOI'].mean():.3f}")
        print(f"     Robustness (SI-SDR std): {freq_data['SI_SDR'].std():.2f} dB")
        
        # SNR-specific insights
        low_snr = freq_data[freq_data['SNR_dB'] <= 0]
        high_snr = freq_data[freq_data['SNR_dB'] >= 10]
        if len(low_snr) > 0 and len(high_snr) > 0:
            print(f"     Low SNR (≤0dB): PESQ={low_snr['PESQ'].mean():.3f}")
            print(f"     High SNR (≥10dB): PESQ={high_snr['PESQ'].mean():.3f}")

print("\n2. FLOOR PARAMETER ANALYSIS:")
if 2 in PHASES_TO_RUN or len(df_success[df_success['Phase'] == 2]) > 0:
    df_p2_analysis = df_success[df_success['Phase'] == 2]
    print("\n   Optimal Floor by Noise Type:")
    for category in df_p2_analysis['Noise_Category'].unique():
        cat_data = df_p2_analysis[df_p2_analysis['Noise_Category'] == category]
        best_floor = cat_data.groupby('Floor')['PESQ'].mean().idxmax()
        worst_floor = cat_data.groupby('Floor')['PESQ'].mean().idxmin()
        
        best_pesq = cat_data[cat_data['Floor'] == best_floor]['PESQ'].mean()
        worst_pesq = cat_data[cat_data['Floor'] == worst_floor]['PESQ'].mean()
        
        print(f"\n   {category}:")
        print(f"     Best Floor: {best_floor} (PESQ: {best_pesq:.3f})")
        print(f"     Worst Floor: {worst_floor} (PESQ: {worst_pesq:.3f})")
        print(f"     Impact: {best_pesq - worst_pesq:.3f} PESQ points")
        
        if category == 'Stationary' and best_floor >= 0.5:
            print(f"       INSIGHT: High floor optimal for stationary!")
            print(f"         → GTCRN already removes stationary noise well")
            print(f"         → SS with low floor causes over-suppression")

print("\n3. PARAMETER IMPACT RANKING:")
if len(df_success) > 10:
    # Calculate variance explained by each parameter
    param_impacts = {}
    for param in ['Freq_spacing', 'Floor', 'Nband', 'FRMSZ_ms', 'Noisefr']:
        if param in df_success.columns and df_success[param].nunique() > 1:
            # Calculate between-group variance as proxy for impact
            grouped_means = df_success.groupby(param)['PESQ'].mean()
            impact = grouped_means.std()
            param_impacts[param] = impact
    
    sorted_impacts = sorted(param_impacts.items(), key=lambda x: x[1], reverse=True)
    print("\n   Parameters ranked by impact on PESQ:")
    for i, (param, impact) in enumerate(sorted_impacts, 1):
        print(f"     {i}. {param}: {impact:.4f} (std of group means)")

# RECOMMENDATIONS
print(f"\n{'='*100}")
print("RECOMMENDATIONS & TRADEOFFS")
print(f"{'='*100}")

print("\nCONFIGURATION OPTIONS:")

print("\n1. BEST OVERALL (Balanced Performance):")
print(f"   Config: {best_overall['Freq_spacing']} | N={int(best_overall['Nband'])} | " +
      f"Frame={int(best_overall['FRMSZ_ms'])}ms | Floor={best_overall['Floor']}")
print(f"   PESQ: {best_overall['PESQ']:.3f} | STOI: {best_overall['STOI']:.3f}")
print(f"   Use when: General purpose, unknown noise type")

# Find best for low SNR
low_snr_data = df_success[df_success['SNR_dB'] <= 0]
if len(low_snr_data) > 0:
    best_low_snr = low_snr_data.loc[low_snr_data['PESQ'].idxmax()]
    print("\n2. BEST FOR LOW SNR (≤0 dB):")
    print(f"   Config: {best_low_snr['Freq_spacing']} | N={int(best_low_snr['Nband'])} | " +
          f"Frame={int(best_low_snr['FRMSZ_ms'])}ms | Floor={best_low_snr['Floor']}")
    print(f"   PESQ: {best_low_snr['PESQ']:.3f} @ {int(best_low_snr['SNR_dB'])}dB")
    print(f"   Use when: Very noisy conditions, need maximum noise reduction")

# Find best for high SNR
high_snr_data = df_success[df_success['SNR_dB'] >= 10]
if len(high_snr_data) > 0:
    best_high_snr = high_snr_data.loc[high_snr_data['PESQ'].idxmax()]
    print("\n3. BEST FOR HIGH SNR (≥10 dB):")
    print(f"   Config: {best_high_snr['Freq_spacing']} | N={int(best_high_snr['Nband'])} | " +
          f"Frame={int(best_high_snr['FRMSZ_ms'])}ms | Floor={best_high_snr['Floor']}")
    print(f"   PESQ: {best_high_snr['PESQ']:.3f} @ {int(best_high_snr['SNR_dB'])}dB")
    print(f"   Use when: Clean audio, need maximum quality preservation")

# Find best for stationary noise
if 'Stationary' in df_success['Noise_Category'].values:
    stationary_data = df_success[df_success['Noise_Category'] == 'Stationary']
    best_stationary = stationary_data.loc[stationary_data['PESQ'].idxmax()]
    print("\n4. BEST FOR STATIONARY NOISE:")
    print(f"   Config: {best_stationary['Freq_spacing']} | N={int(best_stationary['Nband'])} | " +
          f"Frame={int(best_stationary['FRMSZ_ms'])}ms | Floor={best_stationary['Floor']}")
    print(f"   PESQ: {best_stationary['PESQ']:.3f}")
    print(f"   Use when: Fan noise, HVAC, steady background hum")

# Find best for non-stationary
non_stat_categories = ['Street', 'Construction', 'Babble', 'Train', 'Car']
non_stat_data = df_success[df_success['Noise_Category'].isin(non_stat_categories)]
if len(non_stat_data) > 0:
    best_non_stat = non_stat_data.loc[non_stat_data['PESQ'].idxmax()]
    print("\n5. BEST FOR NON-STATIONARY NOISE:")
    print(f"   Config: {best_non_stat['Freq_spacing']} | N={int(best_non_stat['Nband'])} | " +
          f"Frame={int(best_non_stat['FRMSZ_ms'])}ms | Floor={best_non_stat['Floor']}")
    print(f"   PESQ: {best_non_stat['PESQ']:.3f}")
    print(f"   Use when: Street, traffic, construction, babble")


print(f"\n{'='*100}")
print(f"All results saved to: {results_dir}")
print(f"Run enhanced analysis script on: complete_results_all_phases.csv")
print(f"{'='*100}")