"""
Collate Parameter Sweep Results
================================
Recursively finds all *_MERGED.csv files from parameter sweep and combines them
into a single analysis-ready CSV with extracted parameter values.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, Optional

# ====================================
# CONFIGURATION
# ====================================
MODE = "standalone"  # Options: "standalone" or "hybrid"
BASE_DIR = Path(f"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/{MODE}")
OUTPUT_FILE = BASE_DIR / f"COLLATED_ALL_RESULTS_{MODE}.csv"

print("="*80)
print(f"COLLATING PARAMETER SWEEP RESULTS - {MODE.upper()} MODE")
print("="*80)

# ====================================
# PARSE CONFIGURATION FROM FILENAME
# ====================================
def parse_config_from_path(csv_path: Path) -> Optional[Dict]:
    """
    Extract configuration parameters from file path.
    
    Example path:
    .../mband_linear_N4_F8_O25_FL0p001/mband_linear_N4_F8_O25_FL0p001_SNR-5dB_MERGED.csv
    
    Returns dict with: Freq_spacing, Nband, FRMSZ_ms, OVLP, Floor, Noisefr, SNR_dB, Config_ID
    """
    try:
        # Get config directory name (parent of the CSV)
        config_dir = csv_path.parent.name
        
        # Get SNR from filename
        filename = csv_path.stem  # Remove .csv
        snr_match = re.search(r'SNR(-?\d+)dB', filename)
        if not snr_match:
            print(f"  Warning: Could not extract SNR from {csv_path.name}")
            return None
        snr_db = int(snr_match.group(1))
        
        # Parse config string: mband_linear_N4_F8_O25_FL0p001
        # Pattern: mband_{freq}_{N}{nband}_{F}{frmsz}_{O}{ovlp}_{FL}{floor}
        # Optional: _NF{noisefr} if present
        
        # Extract frequency spacing
        freq_match = re.search(r'mband_(\w+)_N', config_dir)
        if not freq_match:
            print(f"  Warning: Could not extract Freq_spacing from {config_dir}")
            return None
        freq_spacing = freq_match.group(1)
        
        # Extract Nband
        nband_match = re.search(r'_N(\d+)_', config_dir)
        if not nband_match:
            return None
        nband = int(nband_match.group(1))
        
        # Extract FRMSZ
        frmsz_match = re.search(r'_F(\d+)_', config_dir)
        if not frmsz_match:
            return None
        frmsz = int(frmsz_match.group(1))
        
        # Extract OVLP
        ovlp_match = re.search(r'_O(\d+)_', config_dir)
        if not ovlp_match:
            return None
        ovlp = int(ovlp_match.group(1))
        
        # Extract Floor (format: FL0p001 means 0.001)
        floor_match = re.search(r'_FL(\d+)p(\d+)', config_dir)
        if not floor_match:
            # Try without decimal: FL0 means 0.0
            floor_match_simple = re.search(r'_FL(\d+)(?:_|$)', config_dir)
            if floor_match_simple:
                floor = float(floor_match_simple.group(1))
            else:
                return None
        else:
            integer_part = floor_match.group(1)
            decimal_part = floor_match.group(2)
            floor = float(f"{integer_part}.{decimal_part}")
        
        # Extract Noisefr (optional, default to 1 if not present)
        noisefr_match = re.search(r'_NF(\d+)', config_dir)
        noisefr = int(noisefr_match.group(1)) if noisefr_match else 1
        
        return {
            'Config_ID': config_dir,
            'Freq_spacing': freq_spacing,
            'Nband': nband,
            'FRMSZ_ms': frmsz,
            'OVLP': ovlp,
            'Floor': floor,
            'Noisefr': noisefr,
            'SNR_dB': snr_db
        }
        
    except Exception as e:
        print(f"  Error parsing {csv_path}: {e}")
        return None

# ====================================
# EXTRACT NOISE CATEGORY FROM ENHANCED_FILE
# ====================================
def categorize_noise_from_filename(enhanced_filename: str) -> str:
    """Extract noise category from enhanced_file column"""
    filename_lower = enhanced_filename.lower()
    
    if any(x in filename_lower for x in ['babble', 'cafeteria']):
        return 'Babble'
    elif any(x in filename_lower for x in ['train', 'inside_train']):
        return 'Train'
    elif 'street' in filename_lower:
        return 'Street'
    elif 'car' in filename_lower:
        return 'Car'
    elif any(x in filename_lower for x in ['construction', 'crane', 'drilling', 
                                            'jackhammer', 'trucks_unloading']):
        return 'Construction'
    elif any(x in filename_lower for x in ['fan', 'cooler', 'ssn', 'white', 'pc_fan']):
        return 'Stationary'
    elif 'flight' in filename_lower:
        return 'Flight'
    else:
        return 'Other'

# ====================================
# FIND ALL MERGED CSV FILES
# ====================================
print(f"\nSearching for *_MERGED.csv files in: {BASE_DIR}")

if not BASE_DIR.exists():
    print(f"ERROR: Directory does not exist: {BASE_DIR}")
    exit(1)

merged_csvs = list(BASE_DIR.glob("*/*_MERGED.csv"))
print(f"Found {len(merged_csvs)} merged CSV files\n")

if len(merged_csvs) == 0:
    print("ERROR: No merged CSV files found!")
    print("Expected structure: .../config_name/config_name_SNR-5dB_MERGED.csv")
    exit(1)

# ====================================
# LOAD AND COMBINE ALL CSVS
# ====================================
all_data = []
failed_files = []

for idx, csv_path in enumerate(merged_csvs, 1):
    print(f"[{idx}/{len(merged_csvs)}] Processing: {csv_path.name}")
    
    # Parse configuration
    config = parse_config_from_path(csv_path)
    if config is None:
        print(f"  ✗ Failed to parse configuration")
        failed_files.append(str(csv_path))
        continue
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded {len(df)} rows")
        
        # Add configuration columns
        for key, value in config.items():
            df[key] = value
        
        # Add noise category
        df['Noise_Category'] = df['enhanced_file'].apply(categorize_noise_from_filename)
        
        # Add mode
        df['Mode'] = MODE
        
        # Add status
        df['Status'] = 'Success'
        
        all_data.append(df)
        
    except Exception as e:
        print(f"  ✗ Error loading CSV: {e}")
        failed_files.append(str(csv_path))
        continue

# ====================================
# COMBINE AND SAVE
# ====================================
if len(all_data) == 0:
    print("\nERROR: No data could be loaded!")
    exit(1)

print(f"\n{'='*80}")
print("COMBINING DATA")
print(f"{'='*80}")

combined_df = pd.concat(all_data, ignore_index=True)

# Reorder columns for readability
column_order = [
    'Config_ID', 'Mode', 'Status',
    'Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr',
    'SNR_dB', 'Noise_Category',
    'clean_file', 'enhanced_file',
    'PESQ', 'STOI', 'SI_SDR', 
    'DNSMOS_p808_mos', 'DNSMOS_mos_sig', 'DNSMOS_mos_bak', 'DNSMOS_mos_ovr',
    'sampling_rate', 'source_file'
]

# Only include columns that exist
final_columns = [col for col in column_order if col in combined_df.columns]
combined_df = combined_df[final_columns]

# Save
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ Combined data saved to: {OUTPUT_FILE}")
print(f"  Total rows: {len(combined_df)}")
print(f"  Total configurations: {combined_df['Config_ID'].nunique()}")
print(f"  SNR levels: {sorted(combined_df['SNR_dB'].unique())}")
print(f"  Noise categories: {sorted(combined_df['Noise_Category'].unique())}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

print("\nParameter ranges:")
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
    if param in combined_df.columns:
        unique_vals = sorted(combined_df[param].unique())
        print(f"  {param}: {unique_vals}")

print("\nSamples per configuration:")
samples_per_config = combined_df.groupby('Config_ID').size()
print(f"  Min: {samples_per_config.min()}")
print(f"  Max: {samples_per_config.max()}")
print(f"  Mean: {samples_per_config.mean():.1f}")

print("\nMetric statistics (all data):")
for metric in ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']:
    if metric in combined_df.columns:
        print(f"  {metric}: {combined_df[metric].mean():.3f} ± {combined_df[metric].std():.3f}")

if failed_files:
    print(f"\n{'='*80}")
    print(f"WARNING: {len(failed_files)} files failed to process:")
    print(f"{'='*80}")
    for f in failed_files:
        print(f"  {f}")

print(f"\n{'='*80}")
print("COLLATION COMPLETE!")
print(f"{'='*80}")
print(f"\nNext step: Run analysis script with:")
print(f"  CSV_FILE = r\"{OUTPUT_FILE}\"")
print(f"{'='*80}\n")