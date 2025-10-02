#!/usr/bin/env python3
"""
WHAM Dataset Splitter
This script analyzes WHAM noise audio files, sorts them by duration (longest to shortest),
and creates train/validation/test splits based on specified requirements.

Requirements (ENFORCED - NO ADJUSTMENTS):
- Train: 11,700 files (EXACTLY)
- Validation: 2,496 files (EXACTLY)
- Test: 2,496 files (EXACTLY)
- Total: 16,692 files (EXACTLY)

Strategy: Prioritize longer files first. Script will FAIL if insufficient files available.
"""

import os
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
import argparse


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds, or 0.0 if file cannot be read
    """
    try:
        duration = librosa.get_duration(filename=file_path)
        return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0


def analyze_wham_dataset(base_path: str) -> pd.DataFrame:
    """
    Analyze all WHAM audio files and get their durations.
    
    Args:
        base_path: Base path to WHAM dataset
        
    Returns:
        DataFrame with columns: file_path, original_split, duration, file_size
    """
    print("Analyzing WHAM dataset...")
    
    data = []
    splits = ['tr', 'cv', 'tt']  # train, validation, test
    
    for split in splits:
        split_path = Path(base_path) / split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist")
            continue
            
        print(f"Processing {split} directory...")
        audio_files = list(split_path.glob("*.wav"))
        
        for audio_file in tqdm(audio_files, desc=f"Analyzing {split}"):
            duration = get_audio_duration(str(audio_file))
            file_size = audio_file.stat().st_size
            
            data.append({
                'file_path': str(audio_file),
                'filename': audio_file.name,
                'original_split': split,
                'duration': duration,
                'file_size': file_size
            })
    
    df = pd.DataFrame(data)
    return df


def create_optimal_splits(df: pd.DataFrame, 
                         train_size: int = 11700,
                         val_size: int = 2496,
                         test_size: int = 2496,
                         min_duration_priority: float = 10.0) -> Dict[str, pd.DataFrame]:
    """
    Create optimal train/validation/test splits prioritizing longer files.
    
    Args:
        df: DataFrame with file information
        train_size: Number of files for training
        val_size: Number of files for validation
        test_size: Number of files for test
        min_duration_priority: Minimum duration to prioritize (seconds)
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    print(f"\nCreating optimal splits:")
    print(f"Target - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Total target files: {train_size + val_size + test_size}")
    print(f"Available files: {len(df)}")
    
    # Sort by duration (longest first)
    df_sorted = df.sort_values('duration', ascending=False).reset_index(drop=True)
    
    # Analyze duration distribution
    long_files = df_sorted[df_sorted['duration'] >= min_duration_priority]
    short_files = df_sorted[df_sorted['duration'] < min_duration_priority]
    
    print(f"\nDuration Analysis:")
    print(f"Files >= {min_duration_priority}s: {len(long_files)}")
    print(f"Files < {min_duration_priority}s: {len(short_files)}")
    print(f"Longest file: {df_sorted['duration'].max():.2f}s")
    print(f"Shortest file: {df_sorted['duration'].min():.2f}s")
    print(f"Mean duration: {df_sorted['duration'].mean():.2f}s")
    print(f"Median duration: {df_sorted['duration'].median():.2f}s")
    
    # Calculate total files needed and available
    total_needed = train_size + val_size + test_size
    total_available = len(df_sorted)
    
    # ENFORCE EXACT NUMBERS - NO ADJUSTMENTS ALLOWED
    print(f"ENFORCING EXACT SPLIT SIZES:")
    print(f"  Required Train: {train_size}")
    print(f"  Required Val: {val_size}")
    print(f"  Required Test: {test_size}")
    print(f"  Total Required: {total_needed}")
    print(f"  Total Available: {total_available}")
    
    if total_needed > total_available:
        raise ValueError(f"\n[ERROR] INSUFFICIENT FILES!\n"
                        f"Required: {total_needed} files ({train_size} train + {val_size} val + {test_size} test)\n"
                        f"Available: {total_available} files\n"
                        f"Missing: {total_needed - total_available} files\n\n"
                        f"Please ensure your dataset has at least {total_needed} audio files.")
    
    print(f"[INFO] Sufficient files available. Proceeding with exact split sizes.")
    
    # Select top files (longest first)
    selected_files = df_sorted.head(total_needed).copy()
    
    # Split proportionally while maintaining longest-first priority
    # We'll take files in order and distribute them to maintain the longest files in training
    
    # Assign splits in a way that training gets the longest files
    selected_files['new_split'] = ''
    
    # Strategy: Assign in blocks to ensure training gets longest files
    train_files = selected_files.iloc[:train_size].copy()
    train_files['new_split'] = 'train'
    
    val_files = selected_files.iloc[train_size:train_size + val_size].copy()
    val_files['new_split'] = 'val'
    
    test_files = selected_files.iloc[train_size + val_size:train_size + val_size + test_size].copy()
    test_files['new_split'] = 'test'
    
    # Combine results
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Print statistics for each split
    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()} Split Statistics:")
        print(f"  Files: {len(split_df)}")
        print(f"  Duration range: {split_df['duration'].min():.2f}s - {split_df['duration'].max():.2f}s")
        print(f"  Mean duration: {split_df['duration'].mean():.2f}s")
        print(f"  Files >= {min_duration_priority}s: {len(split_df[split_df['duration'] >= min_duration_priority])}")
        print(f"  Total duration: {split_df['duration'].sum():.2f}s ({split_df['duration'].sum()/3600:.2f} hours)")
    
    return splits


def save_split_files(splits: Dict[str, pd.DataFrame], 
                    output_base_path: str,
                    copy_files: bool = False,
                    delete_unselected: bool = False,
                    original_df: pd.DataFrame = None) -> None:
    """
    Save split information and optionally copy files to new directory structure.
    Also optionally delete files that weren't selected for any split.
    
    Args:
        splits: Dictionary containing split DataFrames
        output_base_path: Base path for output
        copy_files: Whether to copy actual audio files
        delete_unselected: Whether to delete files not selected for any split
        original_df: Original DataFrame with all files (needed for deletion)
    """
    output_path = Path(output_base_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV files with file lists for each split
    for split_name, split_df in splits.items():
        csv_path = output_path / f"{split_name}_files.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} file list to {csv_path}")
        
        # Optionally copy files
        if copy_files:
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"Copying {len(split_df)} files for {split_name} split...")
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
                src_path = Path(row['file_path'])
                dst_path = split_dir / src_path.name
                shutil.copy2(src_path, dst_path)
    
    # Save summary statistics
    summary = {
        'total_files_selected': sum(len(split_df) for split_df in splits.values()),
        'splits': {}
    }
    
    for split_name, split_df in splits.items():
        summary['splits'][split_name] = {
            'count': len(split_df),
            'duration_stats': {
                'min': float(split_df['duration'].min()),
                'max': float(split_df['duration'].max()),
                'mean': float(split_df['duration'].mean()),
                'median': float(split_df['duration'].median()),
                'total_hours': float(split_df['duration'].sum() / 3600)
            }
        }
    
    summary_path = output_path / "split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Optionally delete files not selected for any split
    if delete_unselected and original_df is not None:
        print(f"\nDeleting unselected files...")
        
        # Get all selected file paths
        selected_files = set()
        for split_df in splits.values():
            selected_files.update(split_df['file_path'].tolist())
        
        # Find files to delete (not in any split)
        all_files = set(original_df['file_path'].tolist())
        files_to_delete = all_files - selected_files
        
        print(f"Files selected for splits: {len(selected_files)}")
        print(f"Files to delete: {len(files_to_delete)}")
        
        if files_to_delete:
            # Ask for confirmation before deleting
            print(f"\nWARNING: About to delete {len(files_to_delete)} audio files!")
            print("These are the shortest duration files that weren't selected for train/val/test.")
            
            # Show some examples of files that will be deleted
            delete_df = original_df[original_df['file_path'].isin(files_to_delete)]
            delete_df_sorted = delete_df.sort_values('duration', ascending=True)
            
            print(f"\nSample files to be deleted (shortest first):")
            for i, (_, row) in enumerate(delete_df_sorted.head(10).iterrows()):
                print(f"  {i+1}. {Path(row['file_path']).name} - {row['duration']:.2f}s")
            if len(files_to_delete) > 10:
                print(f"  ... and {len(files_to_delete) - 10} more files")
            
            response = input(f"\nDo you want to proceed with deleting {len(files_to_delete)} files? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                deleted_count = 0
                failed_deletions = []
                
                for file_path in tqdm(files_to_delete, desc="Deleting files"):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        failed_deletions.append((file_path, str(e)))
                
                print(f"\nDeletion complete:")
                print(f"  Successfully deleted: {deleted_count} files")
                print(f"  Failed deletions: {len(failed_deletions)} files")
                
                if failed_deletions:
                    print("\nFailed deletions:")
                    for file_path, error in failed_deletions[:5]:
                        print(f"  {Path(file_path).name}: {error}")
                    if len(failed_deletions) > 5:
                        print(f"  ... and {len(failed_deletions) - 5} more failures")
                
                # Save deletion log
                deletion_log = {
                    'deleted_files': deleted_count,
                    'failed_deletions': len(failed_deletions),
                    'failed_files': [{'file': fp, 'error': err} for fp, err in failed_deletions]
                }
                
                deletion_log_path = output_path / "deletion_log.json"
                with open(deletion_log_path, 'w') as f:
                    json.dump(deletion_log, f, indent=2)
                print(f"Deletion log saved to {deletion_log_path}")
            else:
                print("Deletion cancelled.")
        else:
            print("No files to delete (all files were selected for splits).")


def main():
    parser = argparse.ArgumentParser(description='WHAM Dataset Splitter')
    parser.add_argument('--wham_path', 
                       default='C:/Users/kapun_63wn2un/Documents/ELEN4012 - Investigation/Repository/PROJECT-25P85/sound_data/raw/WHAM_NOISE_DATASET',
                       help='Path to WHAM dataset')
    parser.add_argument('--output_path',
                       default='C:/Users/kapun_63wn2un/Documents/ELEN4012 - Investigation/Repository/PROJECT-25P85/sound_data/raw/WHAM_NOISE_DATASET/datasplit',
                       help='Output path for splits')
    parser.add_argument('--copy_files', action='store_true',
                       help='Copy audio files to new directory structure')
    parser.add_argument('--train_size', type=int, default=11700,
                       help='Number of training files (FIXED: always 11,700)')
    parser.add_argument('--val_size', type=int, default=2496,
                       help='Number of validation files (FIXED: always 2,496)')
    parser.add_argument('--test_size', type=int, default=2496,
                       help='Number of test files (FIXED: always 2,496)')
    parser.add_argument('--delete_unselected', action='store_true',
                       help='Delete audio files that are not selected for any split (shortest duration files)')
    
    args = parser.parse_args()
    
    print("WHAM Dataset Splitter")
    print("=" * 50)
    
    # Analyze dataset
    df = analyze_wham_dataset(args.wham_path)
    print(f"Total files analyzed: {len(df)}")
    
    # Create optimal splits
    splits = create_optimal_splits(df, args.train_size, args.val_size, args.test_size)
    
    # Save results
    save_split_files(splits, args.output_path, args.copy_files, args.delete_unselected, df)
    
    print("\n" + "=" * 50)
    print("Dataset splitting completed successfully!")
    
    if not args.delete_unselected:
        print("\nNOTE: To delete unselected files (shortest duration ones), run with --delete_unselected flag:")
        print(f"python {__file__} --delete_unselected")

if __name__ == "__main__":
    main()

    # Example Usage to Analyse only:
    # python scripts/wham_dataset_splitter.py

    # Example usage to delete unselected files:
    # python scripts/wham_dataset_splitter.py --delete_unselected