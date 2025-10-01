import pandas as pd
import torchaudio
from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1p2_output' 
results_dir = repo_root / 'results' / 'EXP1' / 'wiener' / 'WF_EXP1p2'

import utils.audio_dataset_loader as loader
from dsp_algorithms.wiener_as import wiener_filter
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics
from utils.parse_and_merge_csvs import merge_csvs

dataset = loader.load_dataset(repo_root, mode="classid_unique_test")
paired_files = loader.pair_sequentially(dataset["urban"], dataset["ears"])

snr_dB_range = [-5, 0, 5, 10, 15]

mu_default = 0.98
a_dd_default = 0.98
eta_default = 0.15

mu_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
a_dd_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
eta_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Debug: Print default values to verify they're correct
print(f"\nDEBUG: Default values - mu_default={mu_default}, a_dd_default={a_dd_default}, eta_default={eta_default}")
print(f"DEBUG: Parameter ranges - mu_values={mu_values}")
print(f"DEBUG: Parameter ranges - a_dd_values={a_dd_values}")
print(f"DEBUG: Parameter ranges - eta_values={eta_values}")

# Define parameter sets to loop through
parameter_sets = [
    ("mu", mu_values, mu_default, a_dd_default, eta_default),
    ("a_dd", a_dd_values, mu_default, a_dd_default, eta_default),
    ("eta", eta_values, mu_default, a_dd_default, eta_default)
]

for param_name, param_values, default_mu, default_a_dd, default_eta in parameter_sets:
    print(f"\n{'='*120}")
    print(f"VARYING PARAMETER: {param_name.upper()}")
    print(f"DEBUG: Unpacked defaults - default_mu={default_mu}, default_a_dd={default_a_dd}, default_eta={default_eta}")
    print(f"DEBUG: Values to iterate: {param_values}")
    print(f"{'='*120}")
    
    for param_value in param_values:
        print(f"\nDEBUG: Processing param_value = {param_value} (type: {type(param_value)})")
        
        # Set current parameter values - ensure they're float type
        if param_name == "mu":
            current_mu = float(param_value)
            current_a_dd = float(default_a_dd)
            current_eta = float(default_eta)
        elif param_name == "a_dd":
            current_mu = float(default_mu)
            current_a_dd = float(param_value)
            current_eta = float(default_eta)
        else:  # eta
            current_mu = float(default_mu)
            current_a_dd = float(default_a_dd)
            current_eta = float(param_value)
            
        print(f"Current parameters: mu={current_mu}, a_dd={current_a_dd}, eta={current_eta}")
        print(f"DEBUG: Types - mu: {type(current_mu)}, a_dd: {type(current_a_dd)}, eta: {type(current_eta)}")
        print(f"DEBUG: CSV filename will be: WF_EXP1p2_data_SNR{{snr_dB}}dB_mu{current_mu}_a{current_a_dd}_eta{current_eta}")

        for snr_dB in snr_dB_range:

            print(f"Processing SNR: {snr_dB} dB")

            output_dir_snr = output_dir / f"{snr_dB}dB"
            output_dir_snr.mkdir(parents=True, exist_ok=True)

            results_dir_snr = results_dir / f"{snr_dB}dB"
            results_dir_snr.mkdir(parents=True, exist_ok=True)

            for urban_path, ears_path in paired_files:

                participant = ears_path.parent.name
                print(f"Urban: {urban_path.name} | EARS: {ears_path.name} | Participant: {participant}")

                clean_waveform, noise_waveform, noisy_speech, clean_sr = loader.prerocess_audio(noisy_audio=urban_path, clean_speech=ears_path, snr_db=snr_dB)

                clean_filename = f"{ears_path.parent.name}_{ears_path.stem}"
                noise_filename = f"{urban_path.parent.name}_{urban_path.stem}"
                output_filename = f"WF_{clean_filename}_{noise_filename}_SNR{snr_dB}dB.wav"

                # Step 2: Apply Wiener filtering (using causal processing)
                print("\n2. Applying causal Wiener filtering...")
                enhanced_speech, enhanced_fs = wiener_filter(
                    noisy_audio=noisy_speech,
                    fs=clean_sr,
                    mu=current_mu,
                    a_dd=current_a_dd,
                    eta=current_eta,
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
                    csv_filename=f'WF_EXP1p2_data_SNR{snr_dB}dB_mu{current_mu}_a{current_a_dd}_eta{current_eta}'
                )
                
                # Print summary
                print(f"\n{'='*100}")
                print(f"Completed Urban: {urban_path.name} | EARS: {ears_path.name} | Participant: {participant}")
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

# Merge CSV files after all parameter variations are complete
print(f"\n{'='*120}")
print("MERGING CSV FILES")
print(f"{'='*120}")


for snr_dB in snr_dB_range:
    results_dir_snr = results_dir / f"{snr_dB}dB"
    
    merged_path = merge_csvs(
        input_dir=results_dir_snr,
        output_dir=results_dir,
        output_filename=f'WF_EXP1p2_merged_{snr_dB}dB.csv',
        keep_source=True
    )
    print(f"Merged results for {snr_dB}dB: {merged_path}")

print(f"\n{'='*120}")
print("ALL PARAMETER VARIATIONS COMPLETED")
print(f"{'='*120}")
