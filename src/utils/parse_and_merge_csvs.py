from pathlib import Path
from typing import Optional, Union
import pandas as pd


def merge_csvs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: str,
    keep_source: bool = True
) -> Optional[Path]:
    """
    Merge all CSV files in a directory into one CSV.

    Parameters:
        input_dir (str | Path): Path to the directory containing CSV files.
        output_dir (str | Path): Directory where the merged CSV will be saved.
        output_filename (str): Name of the output file ('.csv' will be added if missing).
        keep_source (bool): If True, adds a column with the source filename.

    Returns:
        Path | None: Path to the merged CSV if successful, else None.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    # Ensure .csv extension
    if not output_filename.lower().endswith(".csv"):
        output_filename += ".csv"

    output_file = output_dir / output_filename

    all_dfs = []

    for file in input_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if keep_source:
            df["source_file"] = file.name
        all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        return output_file
    else:
        print("[ERROR] No CSV files found in the directory.")
        return None
