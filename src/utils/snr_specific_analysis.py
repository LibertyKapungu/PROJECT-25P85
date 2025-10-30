"""
SNR-Specific Parameter Sweep Analysis
=====================================
Analyzes best configurations at EACH SNR level separately
instead of averaging across all SNRs (which biases toward high SNR)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# ====================================
# CONFIGURATION
# ====================================
MODE = "standalone"
CSV_FILE = rf"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/COLLATED_ALL_RESULTS_{MODE}.csv"
OUTPUT_DIR = Path(f"home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/snr_specific_analysis_{MODE}")
OUTPUT_DIR.mkdir(exist_ok=True, parents = True)

print("="*100)
print("SNR-SPECIFIC PARAMETER ANALYSIS")
print("="*100)

# ====================================
# LOAD DATA
# ====================================
print("\nLoading data...")
df = pd.read_csv(CSV_FILE)
df_success = df[df['Status'] == 'Success'].copy()

# Rename DNSMOS column if needed
if 'DNSMOS_mos_ovr' not in df_success.columns and 'DNSMOS_p808_mos' in df_success.columns:
    df_success['DNSMOS_mos_ovr'] = df_success['DNSMOS_p808_mos']

# Create weighted score
df_success['PESQ_normalized'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_normalized'] + 0.5 * df_success['STOI']

print(f"✓ Loaded {len(df_success)} successful tests")
print(f"  SNR levels: {sorted(df_success['SNR_dB'].unique())}")
print(f"  Noise categories: {sorted(df_success['Noise_Category'].unique())}")

# ====================================
# ANALYSIS BY SNR LEVEL
# ====================================
print("\n" + "="*100)
print("BEST PARAMETERS BY SNR LEVEL")
print("="*100)

snr_results = []

for snr in sorted(df_success['SNR_dB'].unique()):
    snr_data = df_success[df_success['SNR_dB'] == snr]
    
    print(f"\n{'='*100}")
    print(f"SNR = {snr} dB")
    print(f"{'='*100}")
    
    # Analyze each parameter
    for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
        print(f"\n{param}:")
        
        param_stats = snr_data.groupby(param).agg({
            'PESQ': ['mean', 'std'],
            'STOI': ['mean', 'std'],
            'SI_SDR': ['mean', 'std'],
            'DNSMOS_mos_ovr': ['mean', 'std']
        }).round(3)
        
        # Print stats
        for value in snr_data[param].unique():
            subset = snr_data[snr_data[param] == value]
            print(f"  {value}:")
            print(f"    PESQ:  {subset['PESQ'].mean():.3f} ± {subset['PESQ'].std():.3f}")
            print(f"    STOI:  {subset['STOI'].mean():.3f} ± {subset['STOI'].std():.3f}")
            print(f"    SI-SDR: {subset['SI_SDR'].mean():.2f} ± {subset['SI_SDR'].std():.2f} dB")
            print(f"    DNSMOS: {subset['DNSMOS_mos_ovr'].mean():.3f} ± {subset['DNSMOS_mos_ovr'].std():.3f}")
        
        # Find winners
        pesq_winner = snr_data.groupby(param)['PESQ'].mean().idxmax()
        stoi_winner = snr_data.groupby(param)['STOI'].mean().idxmax()
        
        print(f"\n  → Winner (PESQ): {pesq_winner}")
        print(f"  → Winner (STOI): {stoi_winner}")
        
        # Store results
        snr_results.append({
            'SNR_dB': snr,
            'Parameter': param,
            'PESQ_Winner': pesq_winner,
            'PESQ_Score': snr_data.groupby(param)['PESQ'].mean()[pesq_winner],
            'STOI_Winner': stoi_winner,
            'STOI_Score': snr_data.groupby(param)['STOI'].mean()[stoi_winner]
        })
    
    # Best overall config at this SNR
    print(f"\n{'='*100}")
    print(f"BEST OVERALL CONFIG AT {snr} dB:")
    print(f"{'='*100}")
    
    # Average by config at this SNR
    config_avg = snr_data.groupby('Config_ID').agg({
        'Freq_spacing': 'first',
        'Nband': 'first',
        'FRMSZ_ms': 'first',
        'OVLP': 'first',
        'Floor': 'first',
        'Noisefr': 'first',
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean',
        'DNSMOS_mos_ovr': 'mean',
        'Weighted_Score': 'mean'
    }).reset_index()
    
    # Best by PESQ
    best_pesq = config_avg.loc[config_avg['PESQ'].idxmax()]
    print(f"\nBest by PESQ: {best_pesq['Config_ID']}")
    print(f"  Freq={best_pesq['Freq_spacing']}, Nband={int(best_pesq['Nband'])}, "
          f"Frame={int(best_pesq['FRMSZ_ms'])}ms, Overlap={int(best_pesq['OVLP'])}%, "
          f"Floor={best_pesq['Floor']:.4f}, Noisefr={int(best_pesq['Noisefr'])}")
    print(f"  PESQ: {best_pesq['PESQ']:.3f} | STOI: {best_pesq['STOI']:.3f} | "
          f"SI-SDR: {best_pesq['SI_SDR']:.2f} dB | DNSMOS: {best_pesq['DNSMOS_mos_ovr']:.3f}")
    
    # Best by STOI
    best_stoi = config_avg.loc[config_avg['STOI'].idxmax()]
    print(f"\nBest by STOI: {best_stoi['Config_ID']}")
    print(f"  Freq={best_stoi['Freq_spacing']}, Nband={int(best_stoi['Nband'])}, "
          f"Frame={int(best_stoi['FRMSZ_ms'])}ms, Overlap={int(best_stoi['OVLP'])}%, "
          f"Floor={best_stoi['Floor']:.4f}, Noisefr={int(best_stoi['Noisefr'])}")
    print(f"  PESQ: {best_stoi['PESQ']:.3f} | STOI: {best_stoi['STOI']:.3f} | "
          f"SI-SDR: {best_stoi['SI_SDR']:.2f} dB | DNSMOS: {best_stoi['DNSMOS_mos_ovr']:.3f}")
    
    # Best by Weighted Score
    best_weighted = config_avg.loc[config_avg['Weighted_Score'].idxmax()]
    print(f"\nBest by Weighted Score: {best_weighted['Config_ID']}")
    print(f"  Freq={best_weighted['Freq_spacing']}, Nband={int(best_weighted['Nband'])}, "
          f"Frame={int(best_weighted['FRMSZ_ms'])}ms, Overlap={int(best_weighted['OVLP'])}%, "
          f"Floor={best_weighted['Floor']:.4f}, Noisefr={int(best_weighted['Noisefr'])}")
    print(f"  PESQ: {best_weighted['PESQ']:.3f} | STOI: {best_weighted['STOI']:.3f} | "
          f"SI-SDR: {best_weighted['SI_SDR']:.2f} dB | DNSMOS: {best_weighted['DNSMOS_mos_ovr']:.3f} | "
          f"Weighted: {best_weighted['Weighted_Score']:.4f}")

# Save SNR-specific results
snr_results_df = pd.DataFrame(snr_results)
snr_results_df.to_csv(OUTPUT_DIR / 'winners_by_snr.csv', index=False)
print(f"\n✓ Saved: winners_by_snr.csv")

# ====================================
# VISUALIZATIONS
# ====================================
print("\n" + "="*100)
print("GENERATING SNR-SPECIFIC VISUALIZATIONS")
print("="*100)

# 1. Parameter winners across SNR levels
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Parameter Winners Across SNR Levels', fontsize=16, fontweight='bold')

params_to_plot = ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']

for idx, (param, ax) in enumerate(zip(params_to_plot, axes.flat)):
    param_data = snr_results_df[snr_results_df['Parameter'] == param]
    
    # Plot PESQ winners
    ax.scatter(param_data['SNR_dB'], param_data['PESQ_Score'], 
               label='PESQ Winner', s=100, alpha=0.7, marker='o')
    
    # Plot STOI winners
    ax2 = ax.twinx()
    ax2.scatter(param_data['SNR_dB'], param_data['STOI_Score'], 
                color='orange', label='STOI Winner', s=100, alpha=0.7, marker='s')
    
    # Annotate winners
    for _, row in param_data.iterrows():
        ax.annotate(str(row['PESQ_Winner']), 
                   (row['SNR_dB'], row['PESQ_Score']),
                   textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax2.annotate(str(row['STOI_Winner']), 
                    (row['SNR_dB'], row['STOI_Score']),
                    textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color='orange')
    
    ax.set_xlabel('SNR (dB)', fontsize=10)
    ax.set_ylabel('PESQ', fontsize=10, color='blue')
    ax2.set_ylabel('STOI', fontsize=10, color='orange')
    ax.set_title(f'{param}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'parameter_winners_by_snr.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: parameter_winners_by_snr.png")
plt.close()

# 2. Performance trends by parameter value across SNR
for param in ['Freq_spacing', 'Floor', 'Nband']:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{param}: Performance Across SNR Levels', fontsize=16, fontweight='bold')
    
    metrics = ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']
    
    for metric, ax in zip(metrics, axes.flat):
        # Get unique values for this parameter
        param_values = sorted(df_success[param].unique())
        
        for val in param_values:
            subset = df_success[df_success[param] == val]
            snr_avg = subset.groupby('SNR_dB')[metric].agg(['mean', 'std']).reset_index()
            
            ax.plot(snr_avg['SNR_dB'], snr_avg['mean'], 'o-', label=str(val), linewidth=2, markersize=6)
            ax.fill_between(snr_avg['SNR_dB'],
                           snr_avg['mean'] - snr_avg['std'],
                           snr_avg['mean'] + snr_avg['std'],
                           alpha=0.2)
        
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} by {param}', fontsize=12, fontweight='bold')
        ax.legend(title=param, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'trends_by_{param}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: trends_by_{param}.png")
    plt.close()

# 3. Heatmap: Best parameter value at each SNR
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Best Parameter Values Across SNR Levels (PESQ)', fontsize=16, fontweight='bold')

for idx, (param, ax) in enumerate(zip(params_to_plot, axes.flat)):
    # Create matrix: SNR x Parameter_Value
    param_values = sorted(df_success[param].unique())
    snr_levels = sorted(df_success['SNR_dB'].unique())
    
    matrix = []
    for snr in snr_levels:
        row = []
        for val in param_values:
            subset = df_success[(df_success['SNR_dB'] == snr) & (df_success[param] == val)]
            if len(subset) > 0:
                row.append(subset['PESQ'].mean())
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                xticklabels=[str(v) for v in param_values],
                yticklabels=[f'{s}dB' for s in snr_levels],
                cbar_kws={'label': 'PESQ'})
    ax.set_title(f'{param}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Parameter Value', fontsize=10)
    ax.set_ylabel('SNR Level', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_pesq_by_snr_and_param.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_pesq_by_snr_and_param.png")
plt.close()

# 4. Summary table: Winner consistency
print("\n" + "="*100)
print("PARAMETER WINNER CONSISTENCY ACROSS SNR")
print("="*100)

consistency_analysis = []

for param in params_to_plot:
    param_winners = snr_results_df[snr_results_df['Parameter'] == param]
    
    # PESQ winners
    pesq_winners = param_winners['PESQ_Winner'].value_counts()
    most_common_pesq = pesq_winners.idxmax()
    pesq_consistency = pesq_winners.max() / len(param_winners) * 100
    
    # STOI winners
    stoi_winners = param_winners['STOI_Winner'].value_counts()
    most_common_stoi = stoi_winners.idxmax()
    stoi_consistency = stoi_winners.max() / len(param_winners) * 100
    
    print(f"\n{param}:")
    print(f"  PESQ: '{most_common_pesq}' wins {pesq_consistency:.0f}% of SNR levels")
    print(f"        Distribution: {dict(pesq_winners)}")
    print(f"  STOI: '{most_common_stoi}' wins {stoi_consistency:.0f}% of SNR levels")
    print(f"        Distribution: {dict(stoi_winners)}")
    
    consistency_analysis.append({
        'Parameter': param,
        'PESQ_Most_Common': most_common_pesq,
        'PESQ_Consistency_%': pesq_consistency,
        'STOI_Most_Common': most_common_stoi,
        'STOI_Consistency_%': stoi_consistency
    })

consistency_df = pd.DataFrame(consistency_analysis)
consistency_df.to_csv(OUTPUT_DIR / 'winner_consistency.csv', index=False)
print(f"\n✓ Saved: winner_consistency.csv")

# ====================================
# KEY INSIGHTS
# ====================================
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

# Find parameters that change winners across SNR
print("\n1. SNR-DEPENDENT PARAMETERS (different winners at different SNRs):")
for _, row in consistency_df.iterrows():
    if row['PESQ_Consistency_%'] < 60 or row['STOI_Consistency_%'] < 60:
        print(f"   • {row['Parameter']}: Winner changes across SNR levels")
        print(f"     → Consider SNR-adaptive parameter selection")

print("\n2. SNR-INDEPENDENT PARAMETERS (consistent winner across SNRs):")
for _, row in consistency_df.iterrows():
    if row['PESQ_Consistency_%'] >= 80 and row['STOI_Consistency_%'] >= 80:
        print(f"   • {row['Parameter']}: '{row['PESQ_Most_Common']}' for PESQ, '{row['STOI_Most_Common']}' for STOI")
        print(f"     → Can use fixed value")

# Low SNR recommendations
print("\n3. LOW SNR RECOMMENDATIONS (SNR ≤ 0 dB):")
low_snr_data = df_success[df_success['SNR_dB'] <= 0]
for param in ['Freq_spacing', 'Floor', 'Nband']:
    best_val = low_snr_data.groupby(param)['PESQ'].mean().idxmax()
    best_score = low_snr_data.groupby(param)['PESQ'].mean().max()
    print(f"   • Best {param}: {best_val} (avg PESQ: {best_score:.3f})")

# High SNR recommendations
print("\n4. HIGH SNR RECOMMENDATIONS (SNR ≥ 10 dB):")
high_snr_data = df_success[df_success['SNR_dB'] >= 10]
for param in ['Freq_spacing', 'Floor', 'Nband']:
    best_val = high_snr_data.groupby(param)['PESQ'].mean().idxmax()
    best_score = high_snr_data.groupby(param)['PESQ'].mean().max()
    print(f"   • Best {param}: {best_val} (avg PESQ: {best_score:.3f})")

print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR.absolute()}")
print("="*100)

print("\nGenerated files:")
print("  - winners_by_snr.csv (parameter winners at each SNR)")
print("  - winner_consistency.csv (consistency analysis)")
print("  - parameter_winners_by_snr.png (visual summary)")
print("  - trends_by_*.png (performance trends across SNR)")
print("  - heatmap_pesq_by_snr_and_param.png (comprehensive heatmap)")
print("="*100)