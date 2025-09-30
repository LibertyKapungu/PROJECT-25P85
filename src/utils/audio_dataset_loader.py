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
    """

    files: List[Dict[str, Path]] = []
    for pid in participant_ids:
        participant_dir = ears_dataset_path / f"p{pid}"
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
                    files.append({"participant": f"p{pid}", "file": f})
                else:
                    print(f"Skipping {f.name} (duration: {duration_seconds:.2f}s > 60s)")
            except Exception as e:
                print(f"Warning: Could not load audio info for {f.name}: {e}")
                # Skip files that can't be loaded
    return files

def get_urban_files(
    urban_dataset_path: Path,
    urban_metadata: pd.DataFrame,
    folds: Optional[Iterable[int]] = None,
    sliceID_filter: Optional[Iterable[int]] = None,
) -> List[Dict[str, Union[int, Path]]]:
    """Build a list of UrbanSound8K audio file entries from metadata.

    Parameters
    ----------
    urban_dataset_path
        Path to the UrbanSound8K dataset root. The function expects subfolders
        named ``fold1``, ``fold2``, ... containing the WAV files.
    urban_metadata
        Pandas DataFrame loaded from ``UrbanSound8K.csv``. The function
        requires the DataFrame to contain columns ``'fold'`` and
        ``'slice_file_name'``. The caller may add a ``'sliceID'`` column used
        for filtering.
    folds
        Optional filter over fold numbers. If provided only rows whose
        ``'fold'`` column is in this iterable are returned.
    sliceID_filter
        Optional filter over the ``'sliceID'`` column. When provided only
        rows whose ``'sliceID'`` is in this iterable are returned.

    Returns
    -------
    List[Dict[str, Union[int, Path]]]
        Each entry is a dict with keys ``'fold'``, ``'sliceID'`` and ``'file'``
        where ``'file'`` is a :class:`pathlib.Path` pointing to the WAV file.
    """

    df = urban_metadata
    if folds is not None:
        df = df[df["fold"].isin(folds)]
    if sliceID_filter is not None:
        df = df[df["sliceID"].isin(sliceID_filter)]

    files: List[Dict[str, Union[int, Path]]] = []
    for row in df.itertuples():
        urban_file = urban_dataset_path / f"fold{row.fold}" / row.slice_file_name
        # Check audio duration - only include files less than 60 seconds
        try:
            info = torchaudio.info(urban_file)
            duration_seconds = info.num_frames / info.sample_rate
            if duration_seconds < 60.0:
                files.append({"fold": row.fold, "sliceID": row.sliceID, "file": urban_file})
            else:
                print(f"Skipping {urban_file.name} (duration: {duration_seconds:.2f}s > 60s)")
        except Exception as e:
            print(f"Warning: Could not load audio info for {urban_file.name}: {e}")
            # Skip files that can't be loaded
    return files

def prerocess_audio(
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
    urban_files: Iterable[Dict[str, Any]],
    ears_files: Iterable[Dict[str, Any]],
) -> List[Tuple[Path, Path]]:
    """Pair UrbanSound8K entries with EARS entries in-order, returning file paths.

    Each UrbanSound8K entry from ``urban_files`` is paired with the next
    entry from ``ears_files``. If there are more urban entries than EARS
    entries the EARS list is cycled (wrap-around) until every urban item
    is paired. The cycling ensures unique mapping where EARS participants
    cycle from p1-p100 back to p1, and test participants cycle from p101-p107
    back to p107.

    Parameters
    ----------
    urban_files
        Iterable of dictionaries describing UrbanSound8K files. Each dict
        is expected to have a ``'file'`` key whose value is a path-like
        (either a :class:`pathlib.Path` or a string path).
    ears_files
        Iterable of dictionaries describing EARS files. Each dict is
        expected to have a ``'file'`` key whose value is a path-like.

    Returns
    -------
    List[Tuple[pathlib.Path, pathlib.Path]]
        A list of tuples ``(urban_path, ears_path)`` where both elements are
        :class:`pathlib.Path` objects pointing to the respective WAV files.

    Raises
    ------
    ValueError
        If ``ears_files`` is empty because pairing/cycling would be undefined.
    KeyError
        If any entry in ``urban_files`` or ``ears_files`` does not contain a
        ``'file'`` key.
    """

    ears_list = list(ears_files)
    if len(ears_list) == 0:
        raise ValueError("ears_files must be a non-empty iterable")

    paired: List[Tuple[Path, Path]] = []
    urban_list = list(urban_files)
    
    # Create mapping with cycling
    for i, urban in enumerate(urban_list):
        # Use modulo to cycle through EARS files
        ears_idx = i % len(ears_list)
        ears = ears_list[ears_idx]

        if "file" not in urban:
            raise KeyError("each urban entry must contain a 'file' key")
        if "file" not in ears:
            raise KeyError("each ears entry must contain a 'file' key")

        urban_path = Path(urban["file"]) if not isinstance(urban["file"], Path) else urban["file"]
        ears_path = Path(ears["file"]) if not isinstance(ears["file"], Path) else ears["file"]

        paired.append((urban_path, ears_path))
    return paired


def load_dataset(src_dir: Path, mode: str = "all") -> Dict[str, List[Dict[str, Any]]]:
    """Discover dataset files required for experiments.

    The project layout this helper expects (relative to ``src_dir``) is::

        sound_data/raw/URBANSOUND8K_DATASET/...
        sound_data/raw/EARS_DATASET/...

    Parameters
    ----------
    src_dir
        Root of the repository (or the parent directory that contains
        ``sound_data``). This is typically the repository root.
    mode
        One of ``'all'``, ``'train'`` or ``'test'``. Determines which EARS
        participant ids and UrbanSound8K folds are returned.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A mapping with two keys: ``'urban'`` and ``'ears'``. Each value is a
        list of dictionaries describing audio files. Urban entries contain
        ``'fold'``, ``'sliceID'`` and ``'file'`` keys. EARS entries contain
        ``'participant'`` and ``'file'`` keys.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported options.
    """

    # Paths
    URBANSOUND8K_PATH = src_dir / "sound_data" / "raw" / "URBANSOUND8K_DATASET"
    URBANSOUND8K_METADATA = URBANSOUND8K_PATH / "UrbanSound8K.csv"
    EARS_DATASET_PATH = src_dir / "sound_data" / "raw" / "EARS_DATASET"

    # Load UrbanSound8K metadata
    urban_metadata = pd.read_csv(URBANSOUND8K_METADATA)

    urban_metadata["sliceID"] = urban_metadata["slice_file_name"].apply(
        lambda x: int(x.split("-")[-1].split(".")[0])
    )

    # Determine participant IDs and urban filters
    if mode == "all":
        ears_ids = list(range(1, 108))  # p1 to p107
        folds: Optional[Iterable[int]] = None
        sliceID_filter: Optional[Iterable[int]] = None

    elif mode == "test":
        ears_ids = list(range(101, 108))  # p101 to p107
        folds = [10]
        sliceID_filter = None

    elif mode == "train":
        ears_ids = list(range(1, 101))  # p1 to p100
        folds = list(range(1, 10))  # folds 1 to 9
        sliceID_filter = None

    else:
        raise ValueError("mode must be one of ['all', 'test', 'train']")

    ears_files = get_ears_files(EARS_DATASET_PATH, ears_ids)
    urban_files = get_urban_files(
        URBANSOUND8K_PATH, urban_metadata, folds=folds, sliceID_filter=sliceID_filter
    )

    return {"urban": urban_files, "ears": ears_files}
