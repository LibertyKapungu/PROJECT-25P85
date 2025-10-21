"""
SNR-Specific Parameter Analysis
Analyzes best configurations separately for each SNR level
"""

import pandas as pd
from pathlib import Path

# ====================================
# CONFIGURATION
# ====================================
CSV_FILE = r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\Random\OldCSVFiles\spectral_parameter_sweep_results7.csv"
OUTPUT_DIR = Path("snr_specific_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading CSV file...")
df = pd.read_csv(CSV_FILE)
df_success = df[df['Status'] == 'Success'].copy()

# Create weighted score
df_success['PESQ_norm'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_norm'] + 0.5 * df_success['STOI']

print(f"\n{'='*80}")
print("BEST CONFIGURATIONS BY SNR LEVEL")
print(f"{'='*80}")

# Analyze each SNR level separately
snr_levels = sorted(df_success['SNR_dB'].unique())
all_best_configs = []

for snr in snr_levels:
    df_snr = df_success[df_success['SNR_dB'] == snr]
    
    print(f"\n{'='*80}")
    print(f"SNR = {snr} dB")
    print(f"{'='*80}")
    
    # Find best configuration for this SNR
    best_idx = df_snr['Weighted_Score'].idxmax()
    best_config = df_snr.loc[best_idx]
    
    print(f"\nBest Configuration (Weighted Score: {best_config['Weighted_Score']:.4f}):")
    print(f"  Freq_spacing: {best_config['Freq_spacing']}")
    print(f"  Nband: {int(best_config['Nband'])}")
    print(f"  FRMSZ: {int(best_config['FRMSZ_ms'])} ms")
    print(f"  OVLP: {int(best_config['OVLP'])}%")
    print(f"  Floor: {best_config['Floor']:.4f}")
    print(f"\nMetrics:")
    print(f"  PESQ: {best_config['PESQ']:.4f}")
    print(f"  STOI: {best_config['STOI']:.4f}")
    print(f"  SI-SDR: {best_config['SI_SDR']:.2f} dB")
    print(f"  DNSMOS: {best_config['DNSMOS_mos_ovr']:.4f}")
    
    # Store for comparison
    all_best_configs.append({
        'SNR_dB': snr,
        'Freq_spacing': best_config['Freq_spacing'],
        'Nband': int(best_config['Nband']),
        'FRMSZ_ms': int(best_config['FRMSZ_ms']),
        'OVLP': int(best_config['OVLP']),
        'Floor': best_config['Floor'],
        'PESQ': best_config['PESQ'],
        'STOI': best_config['STOI'],
        'SI_SDR': best_config['SI_SDR'],
        'DNSMOS': best_config['DNSMOS_mos_ovr'],
        'Weighted_Score': best_config['Weighted_Score']
    })
    
    # Show top 5 for this SNR
    print(f"\nTop 5 Configurations at {snr} dB SNR:")
    top5 = df_snr.nlargest(5, 'Weighted_Score')[
        ['Config_ID', 'Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 
         'Floor', 'PESQ', 'STOI', 'Weighted_Score']
    ]
    print(top5.to_string(index=False))
    
    # Parameter trends at this SNR
    print(f"\nParameter Impact at {snr} dB SNR:")
    for param in ['Freq_spacing', 'Nband', 'OVLP']:
        avg_scores = df_snr.groupby(param)['Weighted_Score'].mean().sort_values(ascending=False)
        print(f"\n  {param}:")
        for val, score in avg_scores.items():
            print(f"    {val}: {score:.4f}")

# Save SNR-specific best configs
best_configs_df = pd.DataFrame(all_best_configs)
best_configs_df.to_csv(OUTPUT_DIR / 'best_configs_by_snr.csv', index=False)

# Create comparison table
print(f"\n{'='*80}")
print("BEST CONFIGURATION COMPARISON ACROSS SNR LEVELS")
print(f"{'='*80}\n")
print(best_configs_df.to_string(index=False))

# Check if same config works across SNRs
print(f"\n{'='*80}")
print("PARAMETER CONSISTENCY ACROSS SNR LEVELS")
print(f"{'='*80}")
print(f"\nFreq_spacing: {best_configs_df['Freq_spacing'].mode()[0]} (appears {(best_configs_df['Freq_spacing'] == best_configs_df['Freq_spacing'].mode()[0]).sum()}/{len(snr_levels)} times)")
print(f"Nband: {int(best_configs_df['Nband'].mode()[0])} (appears {(best_configs_df['Nband'] == best_configs_df['Nband'].mode()[0]).sum()}/{len(snr_levels)} times)")
print(f"OVLP: {int(best_configs_df['OVLP'].mode()[0])}% (appears {(best_configs_df['OVLP'] == best_configs_df['OVLP'].mode()[0]).sum()}/{len(snr_levels)} times)")

# Recommendation
most_common_config = best_configs_df.mode().iloc[0]
print(f"\n{'='*80}")
print("RECOMMENDED UNIVERSAL CONFIGURATION")
print(f"{'='*80}")
print(f"(Works well across all SNR levels)")
print(f"\n  Freq_spacing: {most_common_config['Freq_spacing']}")
print(f"  Nband: {int(most_common_config['Nband'])}")
print(f"  FRMSZ_ms: {int(most_common_config['FRMSZ_ms'])}")
print(f"  OVLP: {int(most_common_config['OVLP'])}%")
print(f"  Floor: {most_common_config['Floor']:.4f}")

print(f"\n✓ All results saved to: {OUTPUT_DIR.absolute()}")