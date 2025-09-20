import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_and_save_spectrogram(audio_file: str, output_image_path: str, output_file_name: str, 
                             title: str="Mel Spectrogram", n_mels: int=256, hop_length: int=128,
                             n_fft: int=1024, include_metadata_in_filename: bool=True):
    """
    Generate and save mel spectrogram using torchaudio
    
    Parameters:
    - audio_file: path to input audio file
    - output_image_path: base output directory
    - output_file_name: base filename (without extension)
    - title: plot title
    - n_mels: number of mel filter banks
    - hop_length: number of samples between successive frames
    - n_fft: length of FFT window
    - include_metadata_in_filename: whether to include metadata in filename
    """
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract audio filename without extension for metadata
    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    
    fmax = sample_rate // 2
    duration = waveform.shape[1] / sample_rate
    
    # Metadata variables for filename
    sr = sample_rate
    num_mels = n_mels
    hop_len = hop_length
    fft_size = n_fft
    max_freq = int(fmax)
    duration_sec = round(duration, 2)
    
    # Create mel spectrogram transform
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_max=fmax
    )
    
    # Generate spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to dB scale
    amplitude_to_db = T.AmplitudeToDB()
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # Convert to numpy for plotting (squeeze to remove batch dimension)
    mel_spec_db_np = mel_spec_db.squeeze().numpy()
    
    # Create time and frequency axes
    time_frames = mel_spec_db_np.shape[1]
    time_axis = np.linspace(0, duration, time_frames)
    
    plt.figure(figsize=(12, 6), dpi=400)
    
    # Plot spectrogram
    plt.imshow(mel_spec_db_np, 
               aspect='auto', 
               origin='lower',
               extent=[0, duration, 0, fmax],
               cmap='magma')
    
    # Set font properties and create colorbar
    cbar = plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontfamily='Times New Roman', fontsize=18)
    plt.xlabel('Time (s)', fontfamily='Times New Roman', fontsize=18)
    plt.ylabel('Mel Frequency', fontfamily='Times New Roman', fontsize=18)
    
    # Set tick label fonts
    plt.xticks(fontfamily='Times New Roman', fontsize=18)
    plt.yticks(fontfamily='Times New Roman', fontsize=18)
    
    # Set colorbar font
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    
    # Create filename with metadata
    if include_metadata_in_filename:
        metadata_string = f"{audio_filename}_SAMPLINGRATE{sr}_MELS{num_mels}_HOPLENGTH{hop_len}_FFTSIZE{fft_size}_DURATION{duration_sec}s"
        filename = f"{output_file_name}_{metadata_string}"
    else:
        filename = output_file_name
    
    base_path = os.path.join(output_image_path, filename)
    
    plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight', dpi=500)
    plt.savefig(f"{base_path}.png", format='png', bbox_inches='tight', dpi=500)
    plt.close()
    
    print(f"Spectrogram saved with metadata:")
    print(f"  - Audio file: {audio_filename}")
    print(f"  - Sample rate: {sr} Hz")
    print(f"  - Duration: {duration_sec} seconds")
    print(f"  - Mel banks: {num_mels}")
    print(f"  - Hop length: {hop_len}")
    print(f"  - FFT size: {fft_size}")
    print(f"  - Max frequency: {max_freq} Hz")
    print(f"  - Output files: {base_path}.pdf, {base_path}.png")

# Example usage
# generate_and_save_spectrogram("audio_stuff/S_56_02.wav", "yoh/", "spectrogram_3", title="My Mel Spectrogram")