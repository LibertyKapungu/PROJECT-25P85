import pandas as pd
import torchaudio
from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1p1a_output' 
results_dir = repo_root / 'results' / 'EXP1' / 'wiener' / 'WF_EXP1p1a'

from utils.audio_dataset_loader import (
    load_ears_dataset,
    load_noizeus_dataset,
    create_audio_pairs,
    preprocess_audio
)
from dsp_algorithms.wiener_as import wiener_filter
from dsp_algorithms.wiener_PCL import wiener_pcl_filter
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs
from utils.delete_csvs import delete_csvs_in_directory as delete_csvs

# Load test datasets
print("Loading EARS test dataset...")
ears_files = load_ears_dataset(repo_root, mode="test")
print(f"Loaded {len(ears_files)} EARS files for test mode")

print("Loading NOIZEUS test dataset...")
noizeus_files = load_noizeus_dataset(repo_root)
print(f"Loaded {len(noizeus_files)} NOIZEUS files for test mode")

# Create audio pairs
paired_files = create_audio_pairs(noizeus_files, ears_files)
print(f"Created {len(paired_files)} audio pairs for processing")

snr_dB_range = [-5, 0, 5, 10, 15]

for snr_dB in snr_dB_range:

    print(f"Processing SNR: {snr_dB} dB")

    output_dir_snr = output_dir / f"{snr_dB}dB"
    output_dir_snr.mkdir(parents=True, exist_ok=True)

    results_dir_snr = results_dir / f"{snr_dB}dB"
    results_dir_snr.mkdir(parents=True, exist_ok=True)

    for noise_path, clean_path in paired_files:

        participant = clean_path.parent.name
        print(f"Noise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant}")

        clean_waveform, noise_waveform, noisy_speech, clean_sr = preprocess_audio(
            clean_speech=clean_path, 
            noisy_audio=noise_path, 
            snr_db=snr_dB
        )

        clean_filename = f"{clean_path.parent.name}_{clean_path.stem}"
        noise_filename = f"{noise_path.parent.name}_{noise_path.stem}"
        output_filename = f"WF_PCL_{clean_filename}_{noise_filename}_SNR{snr_dB}dB.wav"

        # Step 2: Apply Wiener filtering (using causal processing)
        print("\n2. Applying causal Wiener filtering...")
        enhanced_speech, enhanced_fs = wiener_pcl_filter(
            noisy_audio=noisy_speech,
            fs=clean_sr,
            mu=0.98,
            a_dd=0.98,
            eta=0.15,
            frame_dur_ms=8
        )
        
        # Step 4: Compute and save metrics
        print("\n4. Computing speech enhancement metrics...")
        metrics = compute_and_save_speech_metrics(
            clean_tensor=clean_waveform,
            enhanced_tensor=enhanced_speech,
            fs=enhanced_fs,
            clean_name=clean_filename,
            enhanced_name=output_filename,
            csv_dir=str(results_dir_snr),
            csv_filename='WF_PCL_EXP1p1a_data'
        )
        
        # Print summary
        print(f"\n{'='*100}")
        print(f"Completed Noise: {noise_path.name} | EARS: {clean_path.name} | Participant: {participant}")
        print(f"{'='*100}")
        print(f"Enhanced audio saved to: {output_dir}")
        print(f"Results saved to: {results_dir_snr}")
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
        output_filename=f'WF_PCL_EXP1p1a_merged_{snr_dB}dB.csv',
        keep_source=True
    )

    delete_csvs(input_directory=results_dir_snr)
