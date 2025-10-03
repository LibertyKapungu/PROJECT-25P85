#!/usr/bin/env python3
"""
WHAM Dataset Splitter (Directory-Aware, Repo-Relative Paths)

Ensures exact counts per split by sorting files within each original split directory.

Requirements (ENFORCED - NO ADJUSTMENTS):
- Train (tr dir): 11,700 files (EXACTLY)
- Validation (cv dir): 2,496 files (EXACTLY)
- Test (tt dir): 2,496 files (EXACTLY)

Strategy:
- Within each directory (tr/cv/tt), sort files longest → shortest
- Keep the required number
- (Optionally) delete the rest
"""

import os
import librosa
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import argparse


# --- Resolve paths relative to repo root ---
REPO_ROOT = Path(__file__).resolve().parent.parent
WHAM_PATH = REPO_ROOT / "sound_data" / "raw" / "WHAM_NOISE_DATASET"
OUTPUT_PATH = WHAM_PATH / "datasplit"


def get_audio_duration(file_path: str) -> float:
    """Return duration of an audio file in seconds, or 0.0 if unreadable."""
    try:
        return librosa.get_duration(filename=file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0


def analyze_split(split_path: Path, split_name: str) -> pd.DataFrame:
    """Analyze all .wav files in a given split directory."""
    if not split_path.exists():
        raise ValueError(f"Split directory not found: {split_path}")

    audio_files = list(split_path.glob("*.wav"))
    data = []
    for audio_file in tqdm(audio_files, desc=f"Analyzing {split_name}"):
        duration = get_audio_duration(str(audio_file))
        data.append({
            'file_path': str(audio_file),
            'filename': audio_file.name,
            'split': split_name,
            'duration': duration
        })
    return pd.DataFrame(data)


def enforce_split(df: pd.DataFrame, required_count: int, split_name: str):
    """Sort by duration, keep required_count, discard rest."""
    df_sorted = df.sort_values('duration', ascending=False).reset_index(drop=True)

    if len(df_sorted) < required_count:
        raise ValueError(
            f"Not enough files in {split_name}: "
            f"Required {required_count}, found {len(df_sorted)}"
        )

    selected = df_sorted.iloc[:required_count].copy()
    unselected = df_sorted.iloc[required_count:].copy()
    selected['new_split'] = split_name
    return selected, unselected


def save_and_manage(splits, unselected, output_base_path, copy_files, delete_unselected):
    """Save CSVs, copy files (optional), and delete unselected (optional)."""
    output_path = Path(output_base_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save split CSVs
    for split_name, df in splits.items():
        csv_path = output_path / f"{split_name}_files.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} file list to {csv_path}")

        if copy_files:
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
                shutil.copy2(row['file_path'], split_dir / row['filename'])

    # Save summary
    summary = {
        'splits': {
            split: {
                'count': len(df),
                'min_dur': float(df['duration'].min()),
                'max_dur': float(df['duration'].max()),
                'mean_dur': float(df['duration'].mean()),
                'total_hours': float(df['duration'].sum() / 3600)
            }
            for split, df in splits.items()
        }
    }
    with open(output_path / "split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_path / 'split_summary.json'}")

    # Optionally delete unselected
    if delete_unselected:
        total_to_delete = sum(len(df) for df in unselected.values())
        print(f"\nDeleting {total_to_delete} unselected files (shortest ones in each split)...")
        for split_name, df in unselected.items():
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Deleting {split_name}"):
                try:
                    os.remove(row['file_path'])
                except Exception as e:
                    print(f"Failed to delete {row['file_path']}: {e}")


def main():
    parser = argparse.ArgumentParser(description='WHAM Dataset Splitter (Directory-Aware)')
    parser.add_argument('--wham_path',
        default=WHAM_PATH,
        help='Path to WHAM dataset root (with tr/cv/tt dirs)')
    parser.add_argument('--output_path',
        default=OUTPUT_PATH,
        help='Output path for split CSVs and summary')
    parser.add_argument('--copy_files', action='store_true',
        help='Copy selected files into new dirs')
    parser.add_argument('--delete_unselected', action='store_true',
        help='Delete unselected (shorter) files')
    args = parser.parse_args()

    base_path = Path(args.wham_path)

    # Analyze each split
    df_train = analyze_split(base_path / "tr", "train")
    df_val = analyze_split(base_path / "cv", "val")
    df_test = analyze_split(base_path / "tt", "test")

    # Enforce exact counts
    train_selected, train_unselected = enforce_split(df_train, 11700, "train")
    val_selected, val_unselected = enforce_split(df_val, 2496, "val")
    test_selected, test_unselected = enforce_split(df_test, 2496, "test")

    splits = {"train": train_selected, "val": val_selected, "test": test_selected}
    unselected = {"train": train_unselected, "val": val_unselected, "test": test_unselected}

    # Save and manage files
    save_and_manage(splits, unselected, args.output_path, args.copy_files, args.delete_unselected)

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    main()

    # Usage example:
    # python wham_dataset_splitter.py --copy_files --delete_unselected

    #Analysis Only:
    # python scripts/wham_dataset_splitter.py
