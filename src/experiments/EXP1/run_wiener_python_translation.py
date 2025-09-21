"""
Fixed script to run Wiener filtering experiment with updated module structure.
"""

import sys
import os
from pathlib import Path
import torchaudio

# Path setup
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent 
sys.path.insert(0, str(src_dir))

# Import updated modules
from dsp_algorithms.wiener_as import wiener_filter
from utils.add_noise_over_speech import add_noise_over_speech
from utils.generate_and_save_spectrogram import generate_and_save_spectrogram
from utils.compute_and_save_speech_metrics import compute_and_save_speech_metrics

def main():
    """Run complete Wiener filtering experiment."""
    
    # Experiment parameters
    participant_id = 107
    sentence = 23
    
    # Urban noise parameters
    fold = 10
    fsID = 188813
    classID = 7
    occurrenceID = 10
    sliceID = 1
    
    snr = 100
    
    print(f"Starting Wiener filtering experiment:")
    print(f"  Participant: {participant_id}")
    print(f"  Sentence: {sentence}")
    print(f"  Noise fold: {fold}")
    print(f"  SNR: {snr} dB")
    
    # File paths
    clean_dir = f'sound_data/raw/EARS_DATASET/p{participant_id}'
    clean_filename = f'sentences_{sentence}_regular.wav'  # Added .wav extension
    noise_dir = f'sound_data/raw/URBANSOUND8K_DATASET/fold{fold}'
    noise_filename = f'{fsID}-{classID}-{occurrenceID}-{sliceID}.wav'  # Added .wav extension
    
    # Verify input directories and files exist
    clean_dir_path = Path(clean_dir)
    noise_dir_path = Path(noise_dir)
    clean_file_path = clean_dir_path / clean_filename
    noise_file_path = noise_dir_path / noise_filename
    
    if not clean_dir_path.exists():
        raise FileNotFoundError(f"Clean speech directory does not exist: {clean_dir_path}")
    if not noise_dir_path.exists():
        raise FileNotFoundError(f"Noise directory does not exist: {noise_dir_path}")
    if not clean_file_path.exists():
        raise FileNotFoundError(f"Clean speech file does not exist: {clean_file_path}")
    if not noise_file_path.exists():
        raise FileNotFoundError(f"Noise file does not exist: {noise_file_path}")
    
    print(f"Verified input files exist")
    
    # Output directories - must already exist
    output_dir = src_dir.parent / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1_output'
    results_dir = src_dir.parent / 'results' / 'EXP1'
    
    # Verify directories exist
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    
    print(f"Using output directory: {output_dir}")
    print(f"Using results directory: {results_dir}")
    
    # Step 1: Add noise to clean speech
    print("\n1. Adding noise to clean speech...")
    # Load clean and noise tensors
    clean_tensor, clean_sr = torchaudio.load(clean_file_path)
    noise_tensor, noise_sr = torchaudio.load(noise_file_path)

    noisy_speech, noisy_fs = add_noise_over_speech(
        clean_audio=clean_tensor,
        clean_sr=clean_sr,
        noise_audio=noise_tensor,
        noise_sr=noise_sr,
        snr_db=snr,
        output_dir=None,
        clean_name=clean_filename,
        noise_name=noise_filename,
        sr=16000
    )
    
    # Step 2: Apply Wiener filtering (using causal processing)
    print("\n2. Applying causal Wiener filtering...")
    enhanced_speech, enhanced_fs = wiener_filter(
        noisy_audio=noisy_speech,
        fs=noisy_fs,
        output_dir=str(output_dir),
        output_file=f'wiener_filter_priori_P{participant_id}_S{sentence}_F{fold}_SNR{snr}',
        causal=True,  # Use causal processing
        mu=0.98,
        a_dd=0.95,
        eta=0.15,
        frame_dur=10
    )
    
    print(f"enhanced_fs: {enhanced_fs}, type: {type(enhanced_fs)}")

    # Step 3: Generate spectrogram
    print("\n3. Generating spectrogram...")
    generate_and_save_spectrogram(
        waveform=enhanced_speech,
        sample_rate=enhanced_fs,
        output_image_path=str(results_dir),
        output_file_name='wiener_mel_spectrogram',
        title=f'Wiener Enhanced Speech - P{participant_id} S{sentence} F{fold} SNR{snr}dB',
        include_metadata_in_filename=True,
        audio_name=f'wiener_enhanced_P{participant_id}_S{sentence}_F{fold}_SNR{snr}'
    )
    
    # Step 4: Compute and save metrics
    print("\n4. Computing speech enhancement metrics...")
    
    # For tensor mode, we need to provide the enhanced tensor and sampling rate
    metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_tensor,
        enhanced_tensor=clean_tensor,
        #enhanced_speech,
        fs=enhanced_fs,
        clean_name=clean_filename,
        enhanced_name=f'enhanced_tensor_P{participant_id}_S{sentence}_F{fold}_SNR{snr}.wav',
        csv_dir=str(results_dir),
        csv_filename='wiener_filter_results'
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
    
    return metrics

if __name__ == "__main__":
    # Run single experiment
    print("Running single experiment...")
    main()
    
    # Uncomment to run batch experiment
    # print("\nRunning batch experiment...")
    # run_batch_experiment()