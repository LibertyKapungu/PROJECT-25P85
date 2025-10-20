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
from dsp_algorithms.mband_btrnaming import mband
from deep_learning.TinyDenoiser import TinyDenoiser
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs


onnx_model_dir = repo_root / "models" / "pretrained" / "ONNX"

output_dir = repo_root / 'sound_data' / 'processed' / 'tinydenoiser_processed_outputs' / 'EXP3p1b_output2' 
results_dir = repo_root / 'results' / 'EXP3' / 'GTCRN' / 'wf_test2'

clean_path = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn_model\\test_wavs\\clean_reference.wav'
# enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\enh_mband_normal.wav'
# enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP2\\spectral\\NOISE_ESTIMATION\\mband_wo_VAD_standard_mode_BANDS4_SPACINGLINEAR_FRAME8ms.wav'
enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\output\\enh1_test_FRAME8ms_METHODPOWER_ALPHA5PCT_FREQDEP.wav'
noisy_audio = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn_model\\test_wavs\\noisy_input.wav"

clean_waveform,  clean_fs = torchaudio.load(clean_path)
enhanced_waveform, enhanced_fs = torchaudio.load(enhanced_speech)

# clean_waveform, clean_sr, enhanced_speech, enhanced_fs = preprocess_audio( #Returns Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] where clean sr = 16000Hz
#             clean_speech=clean_path, 
#         )

clean_filename = f"GTCRN_TEST_wovad"
output_filename = f"SS_{clean_filename}_{enhanced_speech}.wav"

results_dir_snr = results_dir
results_dir_snr.mkdir(parents=True, exist_ok=True)

# Step 4: Compute and save metrics
print("\n4. Computing speech enhancement metrics...")
metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_waveform,
        enhanced_tensor=enhanced_waveform,
        fs=enhanced_fs,
        clean_name=clean_filename,
        enhanced_name=output_filename,
        csv_dir=str(results_dir_snr),
        csv_filename='GTCRN_mband_wovad_EXP3p1b_data'
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
output_filename=f'GTCRN_mband_woVAD_EXP3p1b_merged_.csv',
keep_source=True
)

# delete_csvs(input_directory=results_dir_snr)