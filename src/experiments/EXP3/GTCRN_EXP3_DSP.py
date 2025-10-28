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
from dsp_algorithms.wiener_GTCRN import wiener_filter
from deep_learning.TinyDenoiser import TinyDenoiser
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs


onnx_model_dir = repo_root / "models" / "pretrained" / "ONNX"

output_dir = repo_root / 'sound_data' / 'processed' / 'GTCRN' / 'EXP3p1b_output' 
results_dir = repo_root / 'results' / 'EXP3' / 'GTCRN' / 'wf_difnoise_est'

clean_path = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\clean_reference.wav'
# enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\enh_mband_normal.wav'
enhanced_speech = 'C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\enh_noisy_input.wav'
noisy_audio = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\src\\deep_learning\\gtcrn\\gtcrn-main\\test_wavs\\noisy_input.wav"

clean_waveform,  clean_fs = torchaudio.load(clean_path)
enhanced_waveform, enhanced_fs = torchaudio.load(enhanced_speech)

# clean_waveform, clean_sr, enhanced_speech, enhanced_fs = preprocess_audio( #Returns Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] where clean sr = 16000Hz
#             clean_speech=clean_path, 
#         )

clean_filename = f"GTCRN_TEST"
output_filename = f"WF_{clean_filename}_test.wav"

results_dir_snr = results_dir
results_dir_snr.mkdir(parents=True, exist_ok=True)

# Step 2: Apply Wiener filtering (using causal processing)
print("\n2. Applying causal Wiener filtering...")
enh_speech, enh_fs = wiener_filter(
    noisy_audio=enhanced_waveform.squeeze(),
    fs=enhanced_fs,
    mu=0.98,
    a_dd=0.98,
    eta=0.15,
    frame_dur_ms=8, 
    output_dir=output_dir,
    output_file=output_filename,
)

# Step 4: Compute and save metrics
print("\n4. Computing speech enhancement metrics...")
metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_waveform,
        enhanced_tensor=enh_speech,
        fs=enh_fs,
        clean_name=clean_filename,
        enhanced_name=output_filename,
        csv_dir=str(results_dir_snr),
        csv_filename='GTCRN_wf_EXP3p1b_data'
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
output_filename=f'wf_dif_ne_EXP3p1b_merged_.csv',
keep_source=True
)

# delete_csvs(input_directory=results_dir_snr)