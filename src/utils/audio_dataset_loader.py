"""Utility helpers to locate audio files and prepare noisy/clean waveform pairs.

This module contains small helpers used by data preparation scripts in the
repository. The functions are lightweight and intentionally do not mutate
filesystem state (except when delegating to `add_noise_over_speech` which may
write an output file if `output_dir` is provided).
"""

import pandas as pd
from pathlib import Path
import torchaudio
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import random

from utils.add_noise_over_speech import add_noise_over_speech

PathLike = Union[Path, str]

# ------------------------------
# ===== Helper functions =====
# ------------------------------
def get_ears_files(ears_dataset_path: Path, participant_ids: Iterable[Union[int, str]]) -> List[Dict[str, Path]]:
    """Collect WAV file paths for the EARS dataset participants.

    Parameters
    ----------
    ears_dataset_path
        Path to the root directory of the EARS dataset. The function expects
        participant subdirectories named like ``p1``, ``p2``, ... (numeric id
        appended to the letter ``p``).
    participant_ids
        Iterable of participant identifiers. Each identifier may be an
        ``int`` or a ``str``. The value is formatted into ``p{pid}`` when
        constructing the participant subdirectory.

    Returns
    -------
    List[Dict[str, Path]]
        A list of dictionaries with keys ``'participant'`` (e.g. ``'p1'``) and
        ``'file'`` (the absolute path to the WAV file as a :class:`pathlib.Path`).

    Notes
    -----
    The function does not raise if a participant directory is missing; it
    silently yields zero files for that participant. Caller code should
    validate the returned list if the presence of files is required.
    Files over 60 seconds are automatically skipped.
    """

    files: List[Dict[str, Path]] = []
    for pid in participant_ids:
        # Format participant ID with 3-digit zero padding (p001, p002, etc.)
        participant_dir = ears_dataset_path / f"p{pid:03d}"
        # If the participant folder does not exist, skip it (keeps behavior
        # backward-compatible with the previous implementation).
        if not participant_dir.exists():
            continue
        for f in sorted(participant_dir.glob("*.wav")):
            # Check audio duration - only include files less than 60 seconds
            try:
                info = torchaudio.info(f)
                duration_seconds = info.num_frames / info.sample_rate
                if duration_seconds < 60.0:
                    files.append({"participant": f"p{pid:03d}", "file": f})
                else:
                    print(f"Skipping {f.name} (duration: {duration_seconds:.2f}s > 60s)")
            except Exception as e:
                print(f"Warning: Could not load audio info for {f.name}: {e}")
                # Skip files that can't be loaded
    return files


def get_wham_noise_files(wham_dataset_path: Path, split: str) -> List[Dict[str, Path]]:
    """Collect WAV file paths from WHAM noise dataset.
    
    Parameters
    ----------
    wham_dataset_path
        Path to the root directory of the WHAM_NOISE_DATASET.
    split
        One of 'train', 'val', or 'test' to specify which split to load.
        
    Returns
    -------
    List[Dict[str, Path]]
        A list of dictionaries with keys 'split' and 'file'.
    """
    split_mapping = {
        'train': 'tr',
        'val': 'cv', 
        'test': 'tt'
    }
    
    if split not in split_mapping:
        raise ValueError(f"Split must be one of {list(split_mapping.keys())}, got {split}")
    
    wham_split = split_mapping[split]
    split_dir = wham_dataset_path / wham_split
    
    files: List[Dict[str, Path]] = []
    if not split_dir.exists():
        print(f"Warning: WHAM split directory does not exist: {split_dir}")
        return files
    
    for f in sorted(split_dir.glob("*.wav")):
        files.append({"split": split, "file": f})
    
    return files


def get_noizeus_files(noizeus_dataset_path: Path) -> List[Dict[str, Path]]:
    """Collect WAV file paths from NOIZEUS noise dataset.
    
    This dataset can only be used in test mode.
    
    Parameters
    ----------
    noizeus_dataset_path
        Path to the root directory of the NOIZEUS_NOISE_DATASET.
        
    Returns
    -------
    List[Dict[str, Path]]
        A list of dictionaries with keys 'dataset' and 'file'.
    """
    files: List[Dict[str, Path]] = []
    
    if not noizeus_dataset_path.exists():
        print(f"Warning: NOIZEUS dataset directory does not exist: {noizeus_dataset_path}")
        return files
    
    # Look for WAV files in the root directory and subdirectories
    for f in sorted(noizeus_dataset_path.rglob("*.wav")):
        files.append({"dataset": "noizeus", "file": f})
    
    return files



def preprocess_audio(
    clean_speech: Path,
    noisy_audio: Path,
    target_sr: int = 16000,
    snr_db: int = 0,
    output_dir: Optional[PathLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load, optionally resample, and mix a clean and noise audio file.

    This function performs the following steps:
    1. Loads the clean speech and noise audio from disk using ``torchaudio``.
    2. Resamples each signal to ``target_sr`` if their original sampling
       rates differ from ``target_sr``.
    3. Calls :func:`add_noise_over_speech` to produce a noisy mixture at the
       requested SNR. If ``output_dir`` is provided the helper may write the
       generated noisy file to disk.

    Parameters
    ----------
    clean_speech
        Path to the clean speech WAV file.
    noisy_audio
        Path to the noise WAV file that will be mixed over the clean speech.
    target_sr
        Desired output sample rate (Hz). Both input signals will be
        resampled to this rate if needed.
    snr_db
        Signal-to-noise ratio in decibels used when mixing the signals.
    output_dir
        Optional directory where the generated noisy waveform will be
        written. If ``None`` no file is written and the function only returns
        tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        A tuple containing (clean_waveform, noise_waveform, noisy_waveform,
        sample_rate). Waveforms are 2-D ``torch.Tensor`` objects with shape
        ``(channels, samples)`` as returned by ``torchaudio.load``.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    """

    if not clean_speech.exists():
        raise FileNotFoundError(f"Clean speech file does not exist: {clean_speech}")
    if not noisy_audio.exists():
        raise FileNotFoundError(f"Noisy audio file does not exist: {noisy_audio}")

    clean_waveform, clean_sr = torchaudio.load(clean_speech)
    noise_waveform, noise_sr = torchaudio.load(noisy_audio)

    if clean_sr != target_sr:
        resampler_clean = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=target_sr)
        clean_waveform = resampler_clean(clean_waveform)
        clean_sr = target_sr
    if noise_sr != target_sr:
        resampler_noise = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=target_sr)
        noise_waveform = resampler_noise(noise_waveform)
        noise_sr = target_sr

    clean_filename = f"{clean_speech.parent.name}_{clean_speech.stem}"
    noise_filename = f"{noisy_audio.parent.name}_{noisy_audio.stem}"

    if output_dir is None:
        noisy_speech, noisy_fs = add_noise_over_speech(
            clean_audio=clean_waveform,
            clean_sr=clean_sr,
            noise_audio=noise_waveform,
            noise_sr=noise_sr,
            snr_db=snr_db,
            output_dir=None,
        )
    else:
        noisy_speech, noisy_fs = add_noise_over_speech(
            clean_audio=clean_waveform,
            clean_sr=clean_sr,
            noise_audio=noise_waveform,
            noise_sr=noise_sr,
            snr_db=snr_db,
            output_dir=output_dir,
            clean_name=clean_filename,
            noise_name=noise_filename,
        )

    return clean_waveform, noise_waveform, noisy_speech, clean_sr

from itertools import cycle


def pair_sequentially(
    noise_files: Iterable[Dict[str, Any]],
    clean_files: Iterable[Dict[str, Any]],
) -> List[Tuple[Path, Path]]:
    """Pair noise entries with clean speech entries in-order, returning file paths.

    Each noise entry from ``noise_files`` is paired with the next
    entry from ``clean_files``. If there are more noise entries than clean
    entries the clean list is cycled (wrap-around) until every noise item
    is paired. The number of pairs is limited to the number of noise files.

    Parameters
    ----------
    noise_files
        Iterable of dictionaries describing noise files. Each dict
        is expected to have a ``'file'`` key whose value is a path-like
        (either a :class:`pathlib.Path` or a string path).
    clean_files
        Iterable of dictionaries describing clean speech files. Each dict is
        expected to have a ``'file'`` key whose value is a path-like.

    Returns
    -------
    List[Tuple[pathlib.Path, pathlib.Path]]
        A list of tuples ``(noise_path, clean_path)`` where both elements are
        :class:`pathlib.Path` objects pointing to the respective WAV files.
        The length is limited to the number of noise files.

    Raises
    ------
    ValueError
        If ``clean_files`` is empty because pairing/cycling would be undefined.
    KeyError
        If any entry in ``noise_files`` or ``clean_files`` does not contain a
        ``'file'`` key.
    """

    clean_list = list(clean_files)
    noise_list = list(noise_files)
    
    if len(clean_list) == 0:
        raise ValueError("clean_files must be a non-empty iterable")
    if len(noise_list) == 0:
        raise ValueError("noise_files must be a non-empty iterable")

    paired: List[Tuple[Path, Path]] = []
    
    # Create pairs up to the number of noise files (cycling clean files if needed)
    for i in range(len(noise_list)):
        noise = noise_list[i]
        # Use modulo to cycle through clean files
        clean_idx = i % len(clean_list)
        clean = clean_list[clean_idx]

        if "file" not in noise:
            raise KeyError("each noise entry must contain a 'file' key")
        if "file" not in clean:
            raise KeyError("each clean entry must contain a 'file' key")

        noise_path = Path(noise["file"]) if not isinstance(noise["file"], Path) else noise["file"]
        clean_path = Path(clean["file"]) if not isinstance(clean["file"], Path) else clean["file"]

        paired.append((noise_path, clean_path))
    return paired


def load_ears_dataset(
    src_dir: Path, 
    mode: str, 
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load EARS dataset files for the specified mode.

    Parameters
    ----------
    src_dir
        Root of the repository (or the parent directory that contains
        ``sound_data``). This is typically the repository root.
    mode
        One of ``'train'``, ``'validation'``, or ``'test'``. 
        Determines which EARS participant ids are returned.
        - 'train': participants 1-75
        - 'validation': participants 76-91  
        - 'test': participants 92-107
    max_files
        Optional maximum number of files to return. If None, all files are returned.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries describing EARS audio files. Each entry contains
        ``'participant'`` and ``'file'`` keys.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported options.
    """
    EARS_DATASET_PATH = src_dir / "sound_data" / "raw" / "EARS_DATASET"

    # Determine participant IDs based on mode
    if mode == "train":
        ears_ids = list(range(1, 76))  # p1 to p75
    elif mode == "validation":
        ears_ids = list(range(76, 92))  # p76 to p91
    elif mode == "test":
        ears_ids = list(range(92, 108))  # p92 to p107
    else:
        raise ValueError("mode must be one of ['train', 'validation', 'test']")

    ears_files = get_ears_files(EARS_DATASET_PATH, ears_ids)
    
    # Limit the number of files if max_files is specified
    if max_files is not None and len(ears_files) > max_files:
        # Shuffle to get random subset
        random.shuffle(ears_files)
        ears_files = ears_files[:max_files]
        print(f"Limited EARS dataset to {max_files} files for {mode} mode")

    return ears_files


def load_wham_dataset(
    src_dir: Path, 
    mode: str, 
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load WHAM noise dataset files for the specified mode.

    Parameters
    ----------
    src_dir
        Root of the repository (or the parent directory that contains
        ``sound_data``). This is typically the repository root.
    mode
        One of ``'train'``, ``'validation'``, or ``'test'``. 
        Determines which WHAM split is loaded.
    max_files
        Optional maximum number of files to return. If None, all files are returned.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries describing WHAM noise files. Each entry contains
        ``'split'`` and ``'file'`` keys.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported options.
    """
    WHAM_DATASET_PATH = src_dir / "sound_data" / "raw" / "WHAM_NOISE_DATASET"

    # Map mode to WHAM split
    if mode == "validation":
        wham_split = "val"
    else:
        wham_split = mode  # train or test

    wham_files = get_wham_noise_files(WHAM_DATASET_PATH, wham_split)
    
    # Limit the number of files if max_files is specified
    if max_files is not None and len(wham_files) > max_files:
        # Shuffle to get random subset
        random.shuffle(wham_files)
        wham_files = wham_files[:max_files]
        print(f"Limited WHAM dataset to {max_files} files for {mode} mode")

    return wham_files


def load_noizeus_dataset(
    src_dir: Path, 
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load NOIZEUS noise dataset files (test mode only).

    Parameters
    ----------
    src_dir
        Root of the repository (or the parent directory that contains
        ``sound_data``). This is typically the repository root.
    max_files
        Optional maximum number of files to return. If None, all files are returned.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries describing NOIZEUS noise files. Each entry contains
        ``'dataset'`` and ``'file'`` keys.
    """
    NOIZEUS_DATASET_PATH = src_dir / "sound_data" / "raw" / "NOIZEUS_NOISE_DATASET"

    noizeus_files = get_noizeus_files(NOIZEUS_DATASET_PATH)
    
    # Limit the number of files if max_files is specified
    if max_files is not None and len(noizeus_files) > max_files:
        # Shuffle to get random subset
        random.shuffle(noizeus_files)
        noizeus_files = noizeus_files[:max_files]
        print(f"Limited NOIZEUS dataset to {max_files} files")

    return noizeus_files


def create_audio_pairs(
    noise_files: List[Dict[str, Any]], 
    clean_files: List[Dict[str, Any]], 
    max_pairs: Optional[int] = None
) -> List[Tuple[Path, Path]]:
    """Create pairs of noise and clean audio files.

    The number of pairs is automatically limited by the number of noise files.
    Clean files will be cycled if there are fewer clean files than noise files.

    Parameters
    ----------
    noise_files
        List of dictionaries describing noise files.
    clean_files
        List of dictionaries describing clean speech files.
    max_pairs
        Optional maximum number of pairs to return. If None, defaults to the
        number of noise files available.

    Returns
    -------
    List[Tuple[Path, Path]]
        A list of tuples ``(noise_path, clean_path)`` where both elements are
        :class:`pathlib.Path` objects pointing to the respective WAV files.
        The length is limited to min(len(noise_files), max_pairs).
    """
    # Default max_pairs to the number of noise files if not specified
    if max_pairs is None:
        max_pairs = len(noise_files)
    
    # Limit noise files to max_pairs if needed
    if len(noise_files) > max_pairs:
        random.shuffle(noise_files)
        noise_files = noise_files[:max_pairs]
        print(f"Limited noise files to {max_pairs} files")
    
    # Create pairs (this will automatically limit to the number of noise files)
    pairs = pair_sequentially(noise_files, clean_files)
    
    print(f"Created {len(pairs)} audio pairs")
    return pairs
