import pandas as pd
import torchaudio
from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1p1b_output' 
results_dir = repo_root / 'results' / 'EXP1' / 'wiener' / 'WF_EXP1p1b'

import utils.audio_dataset_loader as loader
from dsp_algorithms.wiener_as import wiener_filter
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics

dataset = loader.load_dataset(repo_root, mode="test")
paired_files = loader.pair_sequentially(dataset["urban"], dataset["ears"])

snr_dB_range = [-5, 0, 5, 10, 15]

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
            mu=0.98,
            a_dd=0.98,
            eta=0.15,
            frame_dur_ms=8
        )

        # # Step 3: Generate spectrogram
        # print("\n3. Generating spectrograms...")
        # generate_and_save_spectrogram(
        #     waveform=enhanced_speech,
        #     sample_rate=enhanced_fs,
        #     output_image_path=str(results_dir_snr),
        #     output_file_name='wiener_mel_spectrogram',
        #     title=f'Wiener Enhanced Speech - EARS:{clean_filename} URBS:{noise_filename} SNR:{snr_dB}dB',
        #     include_metadata_in_filename=True,
        #     audio_name=output_filename
        # )

        # generate_and_save_spectrogram(
        #     waveform=noisy_speech,
        #     sample_rate=enhanced_fs,
        #     output_image_path=str(results_dir_snr),
        #     output_file_name='overlap_spectrogram',
        #     title=f'EARS:{clean_filename} URBS:{noise_filename} SNR:{snr_dB}dB',
        #     include_metadata_in_filename=True,
        #     audio_name=output_filename
        # )

        # generate_and_save_spectrogram(
        #     waveform=noise_waveform,
        #     sample_rate=enhanced_fs,
        #     output_image_path=str(results_dir_snr),
        #     output_file_name='noisy_wave_spectrogram',
        #     title=f'URBS:{noise_filename}',
        #     include_metadata_in_filename=True,
        #     audio_name=output_filename
        # )

        # generate_and_save_spectrogram(
        #     waveform=clean_waveform,
        #     sample_rate=enhanced_fs,
        #     output_image_path=str(results_dir_snr),
        #     output_file_name='clean_spectrogram',
        #     title=f'EARS:{clean_filename}',
        #     include_metadata_in_filename=True,
        #     audio_name=output_filename
        # )
        
        # Step 4: Compute and save metrics
        print("\n4. Computing speech enhancement metrics...")
        metrics = compute_and_save_speech_metrics(
            clean_tensor=clean_waveform,
            enhanced_tensor=enhanced_speech,
            fs=enhanced_fs,
            clean_name=clean_filename,
            enhanced_name=output_filename,
            csv_dir=str(results_dir_snr),
            csv_filename='WF_EXP1p1b_data'
        )
        
        # Print summary
        print(f"\n{'='*100}")
        print(f"Completed Urban: {urban_path.name} | EARS: {ears_path.name} | Participant: {participant}")
        print(f"{'='*100}")
        print(f"Enhanced audio saved to: {output_dir}")
        print(f"Results saved to: {results_dir_snr}")
        print(f"Metrics:")
        print(f"  PESQ: {metrics['PESQ']:.3f}")
        print(f"  STOI: {metrics['STOI']:.3f}")
        print(f"  SI-SDR: {metrics['SI_SDR']:.2f} dB")
        print(f"  DNSMOS Overall: {metrics['DNSMOS_mos_ovr']:.3f}")
        print(f"{'='*100}\n")
