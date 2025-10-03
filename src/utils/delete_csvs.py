"""
Utility to delete all CSV files within a specified directory.

Author: Generated Utility
Date: October 2025
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional

def delete_csvs_in_directory(
    input_directory: str,
    recursive: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """
    Delete all CSV files within a specified directory.
    
    Args:
        input_directory (str): Path to the directory containing CSV files to delete
        recursive (bool): If True, delete CSV files in subdirectories as well
        dry_run (bool): If True, only show what files would be deleted without actually deleting them
    
    Returns:
        List[str]: List of file paths that were deleted (or would be deleted in dry_run mode)
    
    Raises:
        ValueError: If the input directory doesn't exist
        PermissionError: If there are insufficient permissions to delete files
    """
    
    # Set up logging if not provided
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    directory_path = Path(input_directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {input_directory}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {input_directory}")
    
    # Determine search pattern based on recursive option
    if recursive:
        search_pattern = os.path.join(input_directory, "**", "*.csv")
        csv_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_directory, "*.csv")
        csv_files = glob.glob(search_pattern)
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    deleted_files = []
    
    if not csv_files:
        logger.info(f"[INFO] No CSV files found in directory: {input_directory}")
        return deleted_files
    
    logger.info(f"[INFO] Found {len(csv_files)} CSV files in directory: {input_directory}")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {csv_file}")
                deleted_files.append(csv_file)
            else:
                os.remove(csv_file)
                logger.info(f"Deleted: {csv_file}")
                deleted_files.append(csv_file)
                
        except PermissionError as e:
            logger.error(f"[ERROR] Permission denied when trying to delete {csv_file}: {e}")
            continue
        except FileNotFoundError as e:
            logger.warning(f"[WARNING] File not found (may have been deleted by another process): {csv_file}")
            continue
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error deleting {csv_file}: {e}")
            continue
    
    if not dry_run:
        logger.info(f"[INFO] Successfully deleted {len(deleted_files)} CSV files")
    else:
        logger.info(f"[DRY RUN] Would delete {len(deleted_files)} CSV files")
    
    return deleted_files


if __name__ == "__main__":
    # Example usage
    deleted_files = delete_csvs_in_directory(
        input_directory="path/to/your/directory",
        recursive=False,
        dry_run=False
    )