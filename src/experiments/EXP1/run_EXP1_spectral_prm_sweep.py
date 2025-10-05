"""
Parameter sweep script for spectral subtraction
Systematically tests different configurations to find optimal settings
"""
import pandas as pd
import torchaudio
from pathlib import Path
import sys
from itertools import product
import time

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'spectral_processed_outputs' / 'PARAM_SWEEP'
results_dir = repo_root / 'results' / 'PARAM_SWEEP' / 'spectral'

import utils.audio_dataset_loader as loader
from dsp_algorithms.mband import mband
# from utils.generate_and_save_spectrogram import generate_and_save_spectrogram  # COMMENTED OUT
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics

# Create directories
output_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# ========== PARAMETER GRID DEFINITION ==========
param_grid = {
    'Freq_spacing': ['linear', 'log', 'mel'],
    'Nband': [4,8,32],           # Test different band counts
    'FRMSZ': [8,10,20],       # Frame duration in ms
    'OVLP': [50,75],         # Overlap percentage
    'AVRGING': [1],                   # Smoothing on/off
    'Noisefr': [1],                   # Noise frame
    'VAD': [1],                       # VAD on/off
    'FLOOR': [0.002]       # Spectral floor values
}

# Generate all combinations
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

print(f"Total configurations to test: {len(param_combinations)}")

# ========== LOAD DATASET ==========
# dataset = loader.load_dataset(repo_root, mode="test")
# paired_files = loader.pair_sequentially(dataset["urban"], dataset["ears"])
ears_files = loader.load_ears_dataset(repo_root, mode="test")
noizeus_files = loader.load_noizeus_dataset(repo_root)
paired_files = loader.create_audio_pairs(noizeus_files, ears_files)

# Process only FIRST audio pair for parameter sweep
# urban_path, ears_path = next(iter(paired_files))
#urban_path, ears_path = paired_files[5]
noizeus_path, ears_path = paired_files[2]  
participant = ears_path.parent.name
print(f"\nUsing test file:")
print(f"Noizeus: {noizeus_path.name} | EARS: {ears_path.name} | Participant: {participant}")

#snr_db = 5
snr_levels = [0, 5, 10, 15]  # Test multiple SNR levels

clean_filename = f"{ears_path.parent.name}_{ears_path.stem}"
noise_filename = f"{noizeus_path.parent.name}_{noizeus_path.stem}"

# ========== RUN PARAMETER SWEEP ==========
all_results = []
start_time = time.time()
total_tests = len(param_combinations) * len(snr_levels)
test_idx = 0

for snr_db in snr_levels:  
    print(f"\n{'='*70}")
    print(f"TESTING SNR = {snr_db} dB")
    print(f"{'='*70}")
    
    # Generate noisy speech at this SNR 
    clean_waveform, noise_waveform, noisy_speech, clean_sr = loader.preprocess_audio(   
        clean_speech=ears_path, 
        noisy_audio=noizeus_path, 
        snr_db=snr_db
    )

    for idx, params in enumerate(param_combinations, 1):
        freq_spacing, nband, FRMSZ, OVLP, AVRGING, Noisefr, VAD, FLOOR = params
        print(f"\n[{idx}/{len(param_combinations)}] Testing configuration:")
        print(f"  Freq: {freq_spacing}, Bands: {nband}, Frame: {FRMSZ}ms, "
            f"Overlap: {OVLP}%, Avg: {AVRGING}, VAD: {VAD}, Floor: {FLOOR}, Noisefr: {Noisefr}")

        try:
            # Create unique output filename
            output_filename = (f"spectral_F{freq_spacing}_B{nband}_"
                            f"FR{FRMSZ}_OV{OVLP}_"
                            f"AVG{AVRGING}_VAD{VAD}_FL{FLOOR}_NF{Noisefr}.wav")
            
            # Apply spectral subtraction with current parameters
            enhanced_speech, enhanced_fs = mband(
                noisy_audio=noisy_speech,
                fs=clean_sr,
                output_dir=None,  # Don't save audio files (saves space)
                output_file=None,
                input_name=clean_filename,
                Nband=nband,
                Freq_spacing=freq_spacing,
                FRMSZ=FRMSZ,
                OVLP=OVLP,
                AVRGING=AVRGING,
                Noisefr=Noisefr,
                FLOOR=FLOOR,
                VAD=VAD
            )
            
            # NO SPECTROGRAM GENERATION (commented out to save time)
            # generate_and_save_spectrogram(...)
            
            # Compute metrics
            metrics = compute_and_save_speech_metrics(
                clean_tensor=clean_waveform,
                enhanced_tensor=enhanced_speech,
                fs=enhanced_fs,
                clean_name=clean_filename,
                enhanced_name=output_filename,
                csv_dir= str(results_dir),  
                csv_filename= f'temp_metrics_config_{idx}'  # Temporary per-config CSV,
            )
            
            # Store results with parameters
            result_row = {
                'Test_ID': test_idx,
                'SNR_dB': snr_db,
                'Config_ID': idx,
                'Freq_spacing': freq_spacing,
                'Nband': nband,
                'FRMSZ_ms': FRMSZ,
                'OVLP': OVLP,
                'Averaging': AVRGING,
                'Noisefr': Noisefr,
                'VAD': VAD,
                'Floor': FLOOR,
                'PESQ': metrics['PESQ'],
                'STOI': metrics['STOI'],
                'SI_SDR': metrics['SI_SDR'],
                'DNSMOS_mos_ovr': metrics['DNSMOS_mos_ovr'],
                'DNSMOS_mos_sig': metrics['DNSMOS_mos_sig'],
                'DNSMOS_mos_bak': metrics['DNSMOS_mos_bak'],
                'Status': 'Success'
            }
            
            print(f"  ✓ PESQ: {metrics['PESQ']:.3f} | STOI: {metrics['STOI']:.3f} | "
                f"SI-SDR: {metrics['SI_SDR']:.2f} dB | DNSMOS: {metrics['DNSMOS_mos_ovr']:.3f}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            result_row = {
                'Config_ID': idx,
                'Freq_spacing': freq_spacing,
                'Nband': nband,
                'FRMSZ_ms': FRMSZ,
                'OVLP': OVLP,
                'Averaging': AVRGING,
                'Noisefr': Noisefr,
                'VAD': VAD,
                'Floor': FLOOR,
                'PESQ': None,
                'STOI': None,
                'SI_SDR': None,
                'DNSMOS_mos_ovr': None,
                'DNSMOS_mos_sig': None,
                'DNSMOS_mos_bak': None,
                'Status': f'Error: {str(e)}'
            }
        
        all_results.append(result_row)

# ========== SAVE COMPREHENSIVE RESULTS ==========
results_df = pd.DataFrame(all_results)
results_csv_path = results_dir / 'spectral_parameter_sweep_results.csv'
results_df.to_csv(results_csv_path, index=False)

elapsed_time = time.time() - start_time
print(f"\n{'='*70}")
print("PARAMETER SWEEP COMPLETED")
print(f"{'='*70}")
print(f"Total time: {elapsed_time/60:.1f} minutes")
print(f"Configurations tested: {len(param_combinations)}")
print(f"Results saved to: {results_csv_path}")

# ========== FIND TOP CONFIGURATIONS ==========
print(f"\n{'='*70}")
print("TOP 5 CONFIGURATIONS BY METRIC:")
print(f"{'='*70}")

for metric in ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']:
    print(f"\n--- Best {metric} ---")
    top_configs = results_df.nlargest(5, metric)[
        ['Config_ID', 'Freq_spacing', 'Nband', 'FRMSZ_ms', 
         'OVLP', 'Averaging', 'VAD', 'Floor', 'Noisefr', metric]
    ]
    print(top_configs.to_string(index=False))

print(f"\n{'='*70}")
print("Full results available in CSV for detailed analysis")
print(f"{'='*70}")

# At the end of the sweep script
import glob
temp_files = glob.glob(str(results_dir / 'temp_*.csv'))
for f in temp_files:
    Path(f).unlink()
print(f"Cleaned up {len(temp_files)} temporary files")