#!/usr/bin/env python3
"""
EARS Dataset Cleaner

This script cleans the EARS dataset by:
1. Deleting all speech files greater than 60 seconds
2. Deleting all files that contain "vegetative" in their filename

The script will scan all participant directories (p001-p107) and remove files
that meet the deletion criteria.

Usage:
    python scripts/ears_dataset_cleaner.py [--dry-run] [--confirm]
    
Options:
    --dry-run    Show what would be deleted without actually deleting
    --confirm    Skip confirmation prompt (automatic deletion)
"""

import os
import sys
import torchaudio
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
from typing import List, Tuple


# --- Resolve paths relative to repo root ---
REPO_ROOT = Path(__file__).resolve().parent.parent
EARS_PATH = REPO_ROOT / "sound_data" / "raw" / "EARS_DATASET"


def get_audio_duration(file_path: Path) -> float:
    """Get the duration of an audio file in seconds.
    
    Parameters
    ----------
    file_path : Path
        Path to the audio file
        
    Returns
    -------
    float
        Duration in seconds, or -1 if file cannot be processed
    """
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {file_path.name}: {e}")
        return -1


def scan_ears_dataset(ears_path: Path, dry_run: bool = True) -> Tuple[List[Path], List[Path]]:
    """Scan EARS dataset for files to delete.
    
    Parameters
    ----------
    ears_path : Path
        Path to the EARS dataset root directory
    dry_run : bool
        If True, only scan without deleting
        
    Returns
    -------
    Tuple[List[Path], List[Path]]
        (files_over_60s, vegetative_files) - lists of files to delete
    """
    if not ears_path.exists():
        print(f"Error: EARS dataset path does not exist: {ears_path}")
        return [], []
    
    files_over_60s = []
    vegetative_files = []
    
    # Get all participant directories (p001-p107)
    participant_dirs = sorted([d for d in ears_path.iterdir() 
                              if d.is_dir() and d.name.startswith('p')])
    
    print(f"Scanning {len(participant_dirs)} participant directories...")
    
    for participant_dir in tqdm(participant_dirs, desc="Scanning participants"):
        # Get all WAV files in this participant directory
        wav_files = list(participant_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            # Check if filename contains "vegetative"
            if "vegetative" in wav_file.name.lower():
                vegetative_files.append(wav_file)
                if not dry_run:
                    print(f"  Found vegetative file: {wav_file.name}")
                continue
            
            # Check duration for non-vegetative files
            duration = get_audio_duration(wav_file)
            if duration > 60.0:
                files_over_60s.append(wav_file)
                if not dry_run:
                    print(f"  Found long file: {wav_file.name} ({duration:.2f}s)")
    
    return files_over_60s, vegetative_files


def delete_files(files_to_delete: List[Path], category: str, dry_run: bool = True) -> int:
    """Delete a list of files.
    
    Parameters
    ----------
    files_to_delete : List[Path]
        List of file paths to delete
    category : str
        Description of the file category for logging
    dry_run : bool
        If True, only simulate deletion
        
    Returns
    -------
    int
        Number of files that would be/were deleted
    """
    if not files_to_delete:
        print(f"No {category} files to delete.")
        return 0
    
    print(f"\n{category} files to delete: {len(files_to_delete)}")
    
    if dry_run:
        print("DRY RUN - Files that would be deleted:")
        for file_path in files_to_delete:
            rel_path = file_path.relative_to(EARS_PATH)
            print(f"  {rel_path}")
        return len(files_to_delete)
    
    # Actually delete files
    deleted_count = 0
    failed_deletions = []
    
    for file_path in tqdm(files_to_delete, desc=f"Deleting {category}"):
        try:
            file_path.unlink()  # Delete the file
            deleted_count += 1
        except Exception as e:
            failed_deletions.append((file_path, str(e)))
            print(f"  Failed to delete {file_path.name}: {e}")
    
    print(f"Successfully deleted {deleted_count} {category} files.")
    
    if failed_deletions:
        print(f"Failed to delete {len(failed_deletions)} files:")
        for file_path, error in failed_deletions:
            print(f"  {file_path.name}: {error}")
    
    return deleted_count


def main():
    """Main function to clean EARS dataset."""
    parser = argparse.ArgumentParser(
        description="Clean EARS dataset by removing files >60s and vegetative files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--confirm", 
        action="store_true", 
        help="Skip confirmation prompt (automatic deletion)"
    )
    
    args = parser.parse_args()
    
    print("EARS Dataset Cleaner")
    print("=" * 50)
    print(f"EARS dataset path: {EARS_PATH}")
    
    if not EARS_PATH.exists():
        print(f"Error: EARS dataset not found at {EARS_PATH}")
        sys.exit(1)
    
    # Scan for files to delete
    print("\nScanning EARS dataset for files to delete...")
    files_over_60s, vegetative_files = scan_ears_dataset(EARS_PATH, dry_run=args.dry_run)
    
    total_files_to_delete = len(files_over_60s) + len(vegetative_files)
    
    print(f"\nScan Results:")
    print(f"Files over 60 seconds: {len(files_over_60s)}")
    print(f"Vegetative files: {len(vegetative_files)}")
    print(f"Total files to delete: {total_files_to_delete}")
    
    if total_files_to_delete == 0:
        print("\nNo files to delete. EARS dataset is already clean!")
        return
    
    if args.dry_run:
        print(f"\nDRY RUN MODE - No files will be deleted")
        delete_files(files_over_60s, "Files over 60 seconds", dry_run=True)
        delete_files(vegetative_files, "Vegetative files", dry_run=True)
        print(f"\nTo actually delete these files, run without --dry-run flag")
        return
    
    # Confirm deletion unless --confirm flag is used
    if not args.confirm:
        print(f"\nThis will permanently delete {total_files_to_delete} files from the EARS dataset.")
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Perform deletion
    print(f"\nDeleting files...")
    deleted_over_60s = delete_files(files_over_60s, "Files over 60 seconds", dry_run=False)
    deleted_vegetative = delete_files(vegetative_files, "Vegetative files", dry_run=False)
    
    total_deleted = deleted_over_60s + deleted_vegetative
    
    print(f"\nCleaning complete!")
    print(f"Total files deleted: {total_deleted}")
    print(f"EARS dataset has been cleaned.")


if __name__ == "__main__":
    main()