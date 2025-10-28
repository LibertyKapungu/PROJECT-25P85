import pandas as pd
import torchaudio
from pathlib import Path
import sys
import numpy as np
import random
import torch

#set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.spectral.mband_btrnaming import mband
from deep_learning.TinyDenoiser import TinyDenoiser
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs


output_dir = repo_root / 'sound_data' / 'processed' / 'matlab_og_algs'
results_dir = repo_root / 'results' / 'EXP1' / 'MATLAB_OG_ALGS' / 'mband_log_og' 

clean_path = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\validation_dataset\\clean_speech\\S_56_02.wav'
#enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\Processed_sounds\\out_mband_avr1_not_og_log_compare.wav'
enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\Matlab2025Files\\SS\\Processed_sounds\\out_mband_avr1_og_log_compare.wav'

clean_waveform,  clean_fs = torchaudio.load(clean_path)
enhanced_waveform, enhanced_fs = torchaudio.load(enhanced_speech)

clean_filename = f"Mband_tests"
output_filename = f"SS_{clean_filename}_{enhanced_speech}.wav"

results_dir_snr = results_dir
results_dir_snr.mkdir(parents=True, exist_ok=True)


# # =====================================================================
# # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ADD THIS FIX ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# # =====================================================================

# # FOR RDC 

# # We calculated the delay for 8kHz (M=32) to be (M-1)/2 = 15.5 samples
# # We will round this to 16 samples.
# DELAY_IN_SAMPLES = 16

# # 1. Create zero padding for the start of the clean signal
# zero_padding = torch.zeros((clean_waveform.shape[0], DELAY_IN_SAMPLES), 
#                            dtype=clean_waveform.dtype)

# # 2. Add the padding to the beginning of the clean waveform to "delay" it
# padded_clean_waveform = torch.cat((zero_padding, clean_waveform), dim=1)

# # 3. Now, trim both signals to be the same length (the shortest of the two)
# min_len = min(padded_clean_waveform.shape[1], enhanced_waveform.shape[1])

# clean_waveform = padded_clean_waveform[:, :min_len]
# enhanced_waveform = enhanced_waveform[:, :min_len]

# print(f"\nApplied {DELAY_IN_SAMPLES}-sample delay compensation for SI-SDR.")

# # =====================================================================
# # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ END OF FIX ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# # =====================================================================

# Step 4: Compute and save metrics
print("\n4. Computing speech enhancement metrics...")
metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_waveform,
        enhanced_tensor=enhanced_waveform,
        fs=enhanced_fs,
        clean_name=clean_filename,
        enhanced_name=output_filename,
        csv_dir=str(results_dir_snr),
        csv_filename='mband_5db_EXP1_data'
)
# Print summary
print(f"\n{'='*100}")
print(f"Enhanced audio saved to: {output_dir}")
print(f"Results saved to: {results_dir_snr}")
print(f"{'='*100}")
print(f"Metrics:")
        
# Handle potential NaN values in output
import math
pesq_str = f"{metrics['PESQ']:.3f}" if not math.isnan(metrics['PESQ']) else "NaN (No utterances detected)"
stoi_str = f"{metrics['STOI']:.3f}" if not math.isnan(metrics['STOI']) else "NaN"
si_sdr_str = f"{metrics['SI_SDR']:.2f} dB" if not math.isnan(metrics['SI_SDR']) else "NaN dB"
dnsmos_str = f"{metrics['DNSMOS_mos_ovr']:.3f}" if not math.isnan(metrics['DNSMOS_mos_ovr']) else "NaN"

print(f"  PESQ: {pesq_str}")
print(f"  STOI: {stoi_str}")
print(f"  SI-SDR: {si_sdr_str}")
print(f"  DNSMOS Overall: {dnsmos_str}")
print(f"{'='*100}\n")

merged_path = merge_csvs(
input_dir=results_dir_snr,
output_dir=results_dir,
output_filename=f'EXP1_OGmat_mband_causal_python_5db_merged_.csv',
keep_source=True
)

# delete_csvs(input_directory=results_dir_snr)