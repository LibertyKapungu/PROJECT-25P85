"""
Quick Parameter Sweep Analysis & Visualization
Loads large CSV file and creates summary statistics and graphs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ====================================
# CONFIGURATION
# ====================================
CSV_FILE = r"C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\OldCSVFiles\\spectral_parameter_sweep_results7.csv"  
OUTPUT_DIR = Path("parameter_sweep_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading CSV file...")
# Load in chunks if file is very large
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} rows successfully")
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Trying to load in chunks...")
    chunks = []
    for chunk in pd.read_csv(CSV_FILE, chunksize=10000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"✓ Loaded {len(df)} rows successfully (chunked)")

# ====================================
# DATA SUMMARY
# ====================================
print("\n" + "="*80)
print("DATA SUMMARY")
print("="*80)

print(f"\nTotal configurations tested: {len(df)}")
print(f"SNR levels: {sorted(df['SNR_dB'].unique())}")
print(f"Success rate: {(df['Status'] == 'Success').sum() / len(df) * 100:.1f}%")

# Parameter ranges
print("\nParameter ranges tested:")
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Averaging', 'Noisefr', 'VAD', 'Floor']:
    if param in df.columns:
        unique_vals = sorted(df[param].unique())
        print(f"  {param}: {unique_vals}")

# ====================================
# BEST CONFIGURATIONS
# ====================================
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY METRIC")
print("="*80)

# Filter only successful runs
df_success = df[df['Status'] == 'Success'].copy()

# Create weighted score (PESQ 50%, STOI 50%)
df_success['PESQ_normalized'] = (df_success['PESQ'] + 0.5) / 5.0  # Normalize [-0.5, 4.5] to [0, 1]
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_normalized'] + 0.5 * df_success['STOI']

metrics = ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', 'Weighted_Score']

for metric in metrics:
    print(f"\n--- Top 10 by {metric} ---")
    top10 = df_success.nlargest(10, metric)[['Config_ID', 'Freq_spacing', 'Nband', 
                                               'FRMSZ_ms', 'OVLP', 'Floor', 
                                               'PESQ', 'STOI', 'SI_SDR', metric]]
    print(top10.to_string(index=False))
    
    # Save to CSV
    top10.to_csv(OUTPUT_DIR / f'top10_by_{metric}.csv', index=False)

# ====================================
# AGGREGATE STATISTICS BY PARAMETER
# ====================================
print("\n" + "="*80)
print("AVERAGE METRICS BY PARAMETER VALUE")
print("="*80)

summary_stats = []

for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor']:
    if param in df_success.columns:
        print(f"\n--- {param} ---")
        grouped = df_success.groupby(param).agg({
            'PESQ': ['mean', 'std'],
            'STOI': ['mean', 'std'],
            'SI_SDR': ['mean', 'std'],
            'DNSMOS_mos_ovr': ['mean', 'std'],
            'Weighted_Score': ['mean', 'std']
        }).round(4)
        print(grouped)
        
        # Store for later
        for val in df_success[param].unique():
            subset = df_success[df_success[param] == val]
            summary_stats.append({
                'Parameter': param,
                'Value': val,
                'PESQ_mean': subset['PESQ'].mean(),
                'STOI_mean': subset['STOI'].mean(),
                'SI_SDR_mean': subset['SI_SDR'].mean(),
                'DNSMOS_mean': subset['DNSMOS_mos_ovr'].mean(),
                'Weighted_mean': subset['Weighted_Score'].mean(),
                'Count': len(subset)
            })

# Save summary
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(OUTPUT_DIR / 'parameter_summary_statistics.csv', index=False)

# ====================================
# VISUALIZATIONS
# ====================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Box plots for each parameter
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor']:
    if param not in df_success.columns:
        continue
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Impact of {param} on Speech Metrics', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [('PESQ', axes[0,0]), ('STOI', axes[0,1]), 
                       ('SI_SDR', axes[1,0]), ('DNSMOS_mos_ovr', axes[1,1])]
    
    for metric, ax in metrics_to_plot:
        if param == 'Floor':
            # Convert to string for better visualization
            df_plot = df_success.copy()
            df_plot[param] = df_plot[param].astype(str)
            sns.boxplot(data=df_plot, x=param, y=metric, ax=ax)
        else:
            sns.boxplot(data=df_success, x=param, y=metric, ax=ax)
        
        ax.set_title(f'{metric} by {param}')
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'boxplot_{param}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: boxplot_{param}.png")
    plt.close()

# 2. Heatmap of parameter interactions (Freq_spacing vs Nband)
print("\nGenerating interaction heatmaps...")
param_pairs = [
    ('Freq_spacing', 'Nband'),
    ('Nband', 'OVLP'),
    ('FRMSZ_ms', 'OVLP')
]

for param1, param2 in param_pairs:
    if param1 not in df_success.columns or param2 not in df_success.columns:
        continue
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Interaction: {param1} × {param2}', fontsize=16, fontweight='bold')
    
    for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], axes.flat)):
        pivot = df_success.pivot_table(values=metric, index=param1, columns=param2, aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': metric})
        ax.set_title(f'{metric}')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'heatmap_{param1}_vs_{param2}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: heatmap_{param1}_vs_{param2}.png")
    plt.close()

# 3. Distribution of metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Speech Quality Metrics', fontsize=16, fontweight='bold')

for metric, ax in zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], axes.flat):
    df_success[metric].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(df_success[metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_success[metric].mean():.3f}')
    ax.axvline(df_success[metric].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_success[metric].median():.3f}')
    ax.set_title(f'{metric} Distribution')
    ax.set_xlabel(metric)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metric_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: metric_distributions.png")
plt.close()

# 4. Scatter plot: PESQ vs STOI (colored by weighted score)
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df_success['PESQ'], df_success['STOI'], 
                     c=df_success['Weighted_Score'], cmap='viridis', 
                     alpha=0.6, s=50)
ax.set_xlabel('PESQ', fontsize=12)
ax.set_ylabel('STOI', fontsize=12)
ax.set_title('PESQ vs STOI (colored by Weighted Score)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weighted Score')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pesq_vs_stoi_scatter.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: pesq_vs_stoi_scatter.png")
plt.close()

# ====================================
# BEST OVERALL CONFIGURATION
# ====================================
print("\n" + "="*80)
print("BEST OVERALL CONFIGURATION (by Weighted Score)")
print("="*80)

best_config = df_success.loc[df_success['Weighted_Score'].idxmax()]
print("\nConfiguration:")
print(f"  Config_ID: {best_config['Config_ID']}")
print(f"  Freq_spacing: {best_config['Freq_spacing']}")
print(f"  Nband: {int(best_config['Nband'])}")
print(f"  FRMSZ_ms: {int(best_config['FRMSZ_ms'])}")
print(f"  OVLP: {int(best_config['OVLP'])}")
print(f"  Floor: {best_config['Floor']:.4f}")
print(f"  Averaging: {int(best_config['Averaging'])}")
print(f"  Noisefr: {int(best_config['Noisefr'])}")
print(f"  VAD: {int(best_config['VAD'])}")

print("\nMetrics:")
print(f"  PESQ: {best_config['PESQ']:.4f}")
print(f"  STOI: {best_config['STOI']:.4f}")
print(f"  SI-SDR: {best_config['SI_SDR']:.2f} dB")
print(f"  DNSMOS Overall: {best_config['DNSMOS_mos_ovr']:.4f}")
print(f"  Weighted Score: {best_config['Weighted_Score']:.4f}")

# Save best config
best_config_df = pd.DataFrame([best_config])
best_config_df.to_csv(OUTPUT_DIR / 'best_configuration.csv', index=False)

# ====================================
# FINAL SUMMARY REPORT
# ====================================
summary_report = f"""
PARAMETER SWEEP ANALYSIS SUMMARY
{'='*80}

Total Configurations Tested: {len(df)}
Successful Runs: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)

BEST CONFIGURATION (Weighted Score):
  Freq_spacing: {best_config['Freq_spacing']}
  Nband: {int(best_config['Nband'])}
  FRMSZ_ms: {int(best_config['FRMSZ_ms'])}
  OVLP: {int(best_config['OVLP'])}
  Floor: {best_config['Floor']:.4f}
  
  PESQ: {best_config['PESQ']:.4f}
  STOI: {best_config['STOI']:.4f}
  SI-SDR: {best_config['SI_SDR']:.2f} dB
  DNSMOS: {best_config['DNSMOS_mos_ovr']:.4f}

PARAMETER INSIGHTS:
"""

# Add parameter insights
for param in ['Freq_spacing', 'Nband', 'Floor']:
    if param in df_success.columns:
        best_val = df_success.groupby(param)['Weighted_Score'].mean().idxmax()
        best_score = df_success.groupby(param)['Weighted_Score'].mean().max()
        summary_report += f"\n  Best {param}: {best_val} (avg weighted score: {best_score:.4f})"

summary_report += f"\n\nAll results saved to: {OUTPUT_DIR.absolute()}\n"

print("\n" + summary_report)

# Save summary report
with open(OUTPUT_DIR / 'analysis_summary.txt', 'w') as f:
    f.write(summary_report)

print("="*80)
print("ANALYSIS COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print("="*80)
print("\nGenerated files:")
print("  - top10_by_*.csv (best configurations by each metric)")
print("  - parameter_summary_statistics.csv (average by parameter)")
print("  - best_configuration.csv (single best config)")
print("  - boxplot_*.png (parameter effect visualizations)")
print("  - heatmap_*.png (parameter interaction effects)")
print("  - metric_distributions.png (overall metric distributions)")
print("  - pesq_vs_stoi_scatter.png (metric relationship)")
print("  - analysis_summary.txt (text summary)")