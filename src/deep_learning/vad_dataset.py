"""
Dataset preparation for GRU-VAD training.

This module creates labeled datasets for voice activity detection by:
1. Loading clean speech and noise audio
2. Creating noisy mixtures at various SNRs
3. Generating frame-level VAD labels (1 for speech, 0 for silence/noise)
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.audio_dataset_loader import (
    load_ears_dataset, 
    load_wham_dataset,
    create_audio_pairs,
    preprocess_audio
)


class VADDataset(Dataset):
    """PyTorch Dataset for Voice Activity Detection.
    
    This dataset generates training samples by:
    1. Loading clean speech and noise audio files
    2. Creating noisy mixtures at specified SNR levels
    3. Generating frame-level labels (1=speech, 0=noise/silence)
    
    The dataset uses energy-based VAD labeling: frames with energy above
    a threshold (relative to the maximum) are labeled as speech.
    
    Args:
        audio_pairs: List of (noise_path, clean_path) tuples
        target_sr: Target sampling rate in Hz
        frame_size: Frame size in samples (default: 128 = 8ms at 16kHz)
        hop_size: Hop size in samples (default: 64 = 50% overlap)
        snr_range: Tuple of (min_snr, max_snr) in dB for random SNR selection
        energy_threshold: Energy threshold for speech detection (0-1)
        max_frames_per_file: Maximum frames to use per audio file (for memory)
    """
    
    def __init__(
        self,
        audio_pairs: List[Tuple[Path, Path]],
        target_sr: int = 16000,
        frame_size: int = 128,
        hop_size: int = 64,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        energy_threshold: float = 0.02,
        max_frames_per_file: Optional[int] = 1000,
    ):
        self.audio_pairs = audio_pairs
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.snr_range = snr_range
        self.energy_threshold = energy_threshold
        self.max_frames_per_file = max_frames_per_file
        
        # Pre-compute dataset statistics
        self.samples = []
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        """Pre-process all audio pairs and create frame-level samples."""
        print(f"Preparing VAD dataset with {len(self.audio_pairs)} audio pairs...")
        
        for idx, (noise_path, clean_path) in enumerate(self.audio_pairs):
            try:
                # Random SNR for this pair
                snr_db = np.random.uniform(*self.snr_range)
                
                # Load and mix audio
                clean_waveform, noise_waveform, noisy_waveform, sr = preprocess_audio(
                    clean_speech=clean_path,
                    noisy_audio=noise_path,
                    target_sr=self.target_sr,
                    snr_db=snr_db,
                    output_dir=None,  # Don't save to disk
                )
                
                # Convert to mono if needed
                if clean_waveform.dim() > 1:
                    clean_waveform = clean_waveform.mean(dim=0)
                if noisy_waveform.dim() > 1:
                    noisy_waveform = noisy_waveform.mean(dim=0)
                
                # Create frames and labels
                frames, labels = self._create_frames_and_labels(
                    noisy_waveform, 
                    clean_waveform
                )
                
                # Limit frames if specified
                if self.max_frames_per_file is not None:
                    if len(frames) > self.max_frames_per_file:
                        # Randomly sample frames
                        indices = np.random.choice(
                            len(frames), 
                            self.max_frames_per_file, 
                            replace=False
                        )
                        frames = [frames[i] for i in indices]
                        labels = [labels[i] for i in indices]
                
                # Add to dataset
                for frame, label in zip(frames, labels):
                    self.samples.append((frame, label))
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(self.audio_pairs)} pairs, "
                          f"total frames: {len(self.samples)}")
                    
            except Exception as e:
                print(f"Error processing pair {idx} ({clean_path.name}): {e}")
                continue
        
        print(f"Dataset prepared: {len(self.samples)} total frames")
    
    def _create_frames_and_labels(
        self, 
        noisy_waveform: torch.Tensor,
        clean_waveform: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Create frames and corresponding VAD labels.
        
        Labels are based on the energy of the clean speech signal.
        
        Args:
            noisy_waveform: Noisy audio tensor
            clean_waveform: Clean speech tensor
            
        Returns:
            Tuple of (frames, labels) where frames are noisy audio segments
            and labels are 1 (speech) or 0 (non-speech)
        """
        frames = []
        labels = []
        
        # Ensure same length
        min_len = min(len(noisy_waveform), len(clean_waveform))
        noisy_waveform = noisy_waveform[:min_len]
        clean_waveform = clean_waveform[:min_len]
        
        # Calculate number of frames
        num_frames = (len(noisy_waveform) - self.frame_size) // self.hop_size + 1
        
        # Compute energy of clean signal for labeling
        clean_energy = clean_waveform ** 2
        
        # Frame-based processing
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            
            # Extract noisy frame (input)
            noisy_frame = noisy_waveform[start:end]
            
            # Extract clean frame for labeling
            clean_frame = clean_waveform[start:end]
            
            # Skip if frame is too short
            if len(noisy_frame) < self.frame_size:
                continue
            
            # Compute frame energy from clean signal
            frame_energy = torch.mean(clean_frame ** 2).item()
            
            # Label based on energy threshold
            # Normalize by maximum possible energy
            max_energy = 1.0  # Assuming normalized audio [-1, 1]
            normalized_energy = frame_energy / max_energy
            
            # Label: 1 if speech, 0 if non-speech
            label = 1 if normalized_energy > self.energy_threshold else 0
            
            frames.append(noisy_frame)
            labels.append(label)
        
        return frames, labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single frame and its label.
        
        Returns:
            Tuple of (frame, label):
                - frame: Tensor of shape (frame_size,)
                - label: Tensor of shape (1,) with value 0 or 1
        """
        frame, label = self.samples[idx]
        return frame, torch.tensor([label], dtype=torch.float32)


def create_vad_dataloaders(
    src_dir: Path,
    batch_size: int = 32,
    target_sr: int = 16000,
    frame_size: int = 128,
    hop_size: int = 64,
    snr_range: Tuple[float, float] = (-5.0, 20.0),
    num_workers: int = 0,
    max_train_pairs: Optional[int] = 50,
    max_val_pairs: Optional[int] = 10,
    max_test_pairs: Optional[int] = 10,
    max_frames_per_file: Optional[int] = 1000,
) -> Dict[str, DataLoader]:
    """Create training, validation, and test dataloaders.
    
    Args:
        src_dir: Root directory of the project
        batch_size: Batch size for training
        target_sr: Target sampling rate
        frame_size: Frame size in samples
        hop_size: Hop size in samples
        snr_range: SNR range for mixing
        num_workers: Number of dataloader workers
        max_train_pairs: Max audio pairs for training
        max_val_pairs: Max audio pairs for validation
        max_test_pairs: Max audio pairs for testing
        max_frames_per_file: Max frames per audio file
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    print("Creating VAD dataloaders...")
    
    # Load datasets
    train_clean = load_ears_dataset(src_dir, mode='train', max_files=max_train_pairs)
    train_noise = load_wham_dataset(src_dir, mode='train', max_files=max_train_pairs)
    train_pairs = create_audio_pairs(train_noise, train_clean, max_pairs=max_train_pairs)
    
    val_clean = load_ears_dataset(src_dir, mode='validation', max_files=max_val_pairs)
    val_noise = load_wham_dataset(src_dir, mode='validation', max_files=max_val_pairs)
    val_pairs = create_audio_pairs(val_noise, val_clean, max_pairs=max_val_pairs)
    
    test_clean = load_ears_dataset(src_dir, mode='test', max_files=max_test_pairs)
    test_noise = load_wham_dataset(src_dir, mode='test', max_files=max_test_pairs)
    test_pairs = create_audio_pairs(test_noise, test_clean, max_pairs=max_test_pairs)
    
    # Create datasets
    train_dataset = VADDataset(
        train_pairs, 
        target_sr=target_sr,
        frame_size=frame_size,
        hop_size=hop_size,
        snr_range=snr_range,
        max_frames_per_file=max_frames_per_file,
    )
    
    val_dataset = VADDataset(
        val_pairs,
        target_sr=target_sr,
        frame_size=frame_size,
        hop_size=hop_size,
        snr_range=snr_range,
        max_frames_per_file=max_frames_per_file,
    )
    
    test_dataset = VADDataset(
        test_pairs,
        target_sr=target_sr,
        frame_size=frame_size,
        hop_size=hop_size,
        snr_range=snr_range,
        max_frames_per_file=max_frames_per_file,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


if __name__ == "__main__":
    # Test dataset creation
    src_dir = Path(__file__).parent.parent.parent
    
    print("Testing VAD dataset creation...")
    
    dataloaders = create_vad_dataloaders(
        src_dir=src_dir,
        batch_size=32,
        max_train_pairs=5,
        max_val_pairs=2,
        max_test_pairs=2,
        max_frames_per_file=500,
    )
    
    # Test a batch
    for batch_frames, batch_labels in dataloaders['train']:
        print(f"\nBatch shapes:")
        print(f"Frames: {batch_frames.shape}")
        print(f"Labels: {batch_labels.shape}")
        print(f"Label distribution: {batch_labels.mean().item():.2%} speech frames")
        break
