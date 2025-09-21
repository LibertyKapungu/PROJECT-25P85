"""
Spectrogram generation and visualization module.

Creates high-quality mel spectrograms for audio analysis.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union


def generate_and_save_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    output_image_path: Union[str, Path] = "spectrograms",
    output_file_name: str = "spectrogram",
    title: str = "Mel Spectrogram",
    n_mels: int = 128,
    hop_length: int = 128,
    n_fft: int = 1024,
    include_metadata_in_filename: bool = True,
    audio_name: Optional[str] = None,
    colormap: str = 'magma',
    figure_size: tuple = (12, 6),
    dpi: int = 300
) -> None:
    """
    Generate and save mel spectrogram with publication-quality formatting.
    
    Args:
        waveform: Required. Audio tensor (channels, samples) or (samples,) for mono.
        sample_rate: Required. Sampling rate in Hz.
        output_image_path: Output directory for spectrogram images.
        output_file_name: Base filename (without extension).
        title: Plot title.
        n_mels: Number of mel filter banks.
        hop_length: Hop length in samples.
        n_fft: FFT size.
        include_metadata_in_filename: Whether to include parameters in filename.
        audio_name: Optional name for the audio source used in metadata/filename.
        colormap: Matplotlib colormap name.
        figure_size: Figure size as (width, height) in inches.
        dpi: Resolution for saved images.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # Input validation
    if waveform is None or sample_rate is None:
        raise ValueError("waveform and sample_rate must be provided")

    audio_tensor = waveform
    sr = sample_rate
    audio_filename = audio_name or "tensor_audio"
    
    # Convert to mono if multi-channel
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    
    # Calculate audio properties
    duration = audio_tensor.shape[-1] / sr
    nyquist_freq = sr // 2
    
    print(f"Processing audio: {audio_filename}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Channels: {audio_tensor.shape[0]}")
    
    # Create mel spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_max=nyquist_freq,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=2.0
    )
    
    # Generate mel spectrogram
    print("Generating mel spectrogram...")
    mel_spec = mel_transform(audio_tensor)
    
    # Convert to dB scale
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # Convert to numpy for plotting (remove batch and channel dimensions)
    mel_spec_np = mel_spec_db.squeeze().numpy()
    
    # Create time axis
    n_frames = mel_spec_np.shape[1] 
    time_axis = np.linspace(0, duration, n_frames)
    
    # Create frequency axis (mel scale to Hz approximation)
    mel_frequencies = np.linspace(0, n_mels, n_mels)
    
    # Create publication-quality plot
    plt.figure(figsize=figure_size, dpi=dpi)
    
    # Plot spectrogram
    im = plt.imshow(
        mel_spec_np,
        aspect='auto',
        origin='lower', 
        extent=[0, duration, 0, nyquist_freq],
        cmap=colormap,
        interpolation='bilinear'
    )
    
    # Formatting for publication quality
    plt.title(title, fontfamily='Times New Roman', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time (s)', fontfamily='Times New Roman', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontfamily='Times New Roman', fontsize=18)
    
    # Configure ticks
    plt.xticks(fontfamily='Times New Roman', fontsize=18)
    plt.yticks(fontfamily='Times New Roman', fontsize=18)
    
    # Add colorbar
    cbar = plt.colorbar(im, format='%+2.0f dB', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # Grid for better readability (subtle)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Generate output filename
    output_path = Path(output_image_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if include_metadata_in_filename:
        metadata_parts = [
            audio_filename,
            f"SR{sr}",
            f"MELS{n_mels}",
            f"HOP{hop_length}", 
            f"FFT{n_fft}",
            f"DUR{duration:.1f}s"
        ]
        filename_with_metadata = f"{output_file_name}_{'_'.join(metadata_parts)}"
    else:
        filename_with_metadata = output_file_name
    
    # Save in multiple formats
    base_path = output_path / filename_with_metadata
    
    # Save as PNG for web/presentations
    png_path = base_path.with_suffix('.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi, 
                facecolor='white', edgecolor='none')
    
    # Save as PDF for publications
    pdf_path = base_path.with_suffix('.pdf') 
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Save as SVG for vector graphics
    svg_path = base_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.close()
    
    # Print summary
    print(f"\nSpectrogram generated successfully!")
    print(f"  Audio source: {audio_filename}")
    print(f"  Parameters: {n_mels} mels, {hop_length} hop, {n_fft} FFT")
    print(f"  Duration: {duration:.2f}s at {sr}Hz")
    print(f"  Max frequency: {nyquist_freq}Hz")
    print(f"  Output files:")
    print(f"    PNG: {png_path}")
    print(f"    PDF: {pdf_path}")
    print(f"    SVG: {svg_path}")

if __name__ == "__main__":
    # Example usage
    generate_and_save_spectrogram(
        audio_file="audio_files/speech.wav",
        output_image_path="spectrograms",
        output_file_name="example_spectrogram",
        title="Speech Mel Spectrogram"
    )