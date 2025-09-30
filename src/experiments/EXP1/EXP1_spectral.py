import pandas as pd
import torchaudio
from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'spectral_processed_outputs' / 'EXP1_output'
results_dir = repo_root / 'results' / 'EXP1' / 'spectral'

import utils.audio_dataset_loader as loader
from dsp_algorithms.mband import mband
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics

dataset = loader.load_dataset(repo_root, mode="test")
paired_files = loader.pair_sequentially(dataset["urban"], dataset["ears"])

for urban_path, ears_path in paired_files:

    participant = ears_path.parent.name
    print(f"Urban: {urban_path.name} | EARS: {ears_path.name} | Participant: {participant}")

    snr_db = 5
    clean_waveform, noise_waveform, noisy_speech, clean_sr = loader.prerocess_audio(noisy_audio=urban_path, clean_speech=ears_path, snr_db=snr_db)

    clean_filename = f"{ears_path.parent.name}_{ears_path.stem}"
    noise_filename = f"{urban_path.parent.name}_{urban_path.stem}"
    output_filename = f"spectral_{clean_filename}_{noise_filename}_SNR{snr_db}dB.wav"

    # Step 2: Apply spectral filtering (using causal processing)
    print("\n2. Applying causal spectral filtering...")
    enhanced_speech, enhanced_fs = mband(
        noisy_audio=noisy_speech,
        fs=clean_sr,
        output_dir=output_dir,
        output_file=output_filename,
        input_name=clean_filename,
        Nband = 4,
        Freq_spacing = 'log',
        FRMSZ = 8,
        OVLP = 50,
        AVRGING = 1,
        Noisefr = 1,
        FLOOR = 0.002,
        VAD = 1
    )

    # Step 3: Generate spectrogram
    print("\n3. Generating spectrogram...")
    generate_and_save_spectrogram(
        waveform=enhanced_speech,
        sample_rate=enhanced_fs,
        output_image_path=str(results_dir),
        output_file_name='spectral_mel_spectrogram',
        title=f'Spectral Enhanced Speech - EARS:{clean_filename} URBS:{noise_filename} SNR:{snr_db}dB',
        include_metadata_in_filename=True,
        audio_name=output_filename
    )
    
    # Step 4: Compute and save metrics
    print("\n4. Computing speech enhancement metrics...")
    
    metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_waveform,
        enhanced_tensor=enhanced_speech,
        fs=enhanced_fs,
        clean_name=clean_filename,
        enhanced_name=output_filename,
        csv_dir=str(results_dir),
        csv_filename='spectral_EXP1_results'
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"Enhanced audio saved to: {output_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"Metrics:")
    print(f"  PESQ: {metrics['PESQ']:.3f}")
    print(f"  STOI: {metrics['STOI']:.3f}")
    print(f"  SI-SDR: {metrics['SI_SDR']:.2f} dB")
    print(f"  DNSMOS Overall: {metrics['DNSMOS_mos_ovr']:.3f}")
    print(f"{'='*50}")
    
    break
