"""
Parameter Sweep Analysis & Visualization
Analyzes GTCRN + Spectral Subtraction parameter sweep results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ====================================
# CONFIGURATION
# ====================================
# UPDATE THIS PATH to your collated results CSV
MODE = "hybrid"  # or "hybrid"
CSV_FILE = rf"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP3/COLLATED_ALL_RESULTS_{MODE}.csv"
OUTPUT_DIR = Path(f"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP3/parameter_sweep_analysis_{MODE}")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*100)
print("PARAMETER SWEEP ANALYSIS")
print("="*100)

# ====================================
# LOAD DATA
# ====================================
print("\nLoading CSV file...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} rows successfully")
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    exit(1)

# ====================================
# DATA SUMMARY
# ====================================
print("\n" + "="*100)
print("DATA SUMMARY")
print("="*100)

print(f"\nTotal configurations tested: {df['Config_ID'].nunique()}")
print(f"Total tests: {len(df)}")
# print(f"Noise files: {df['Noise_File'].nunique()}")
print(f"Noise categories: {df['Noise_Category'].unique().tolist()}")
print(f"Success rate: {(df['Status'] == 'Success').sum() / len(df) * 100:.1f}%")

# Parameter ranges
print("\nParameter ranges tested:")
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP']:
    if param in df.columns:
        unique_vals = sorted(df[param].unique())
        print(f"  {param}: {unique_vals}")

# ====================================
# FILTER SUCCESSFUL RUNS
# ====================================
df_success = df[df['Status'] == 'Success'].copy()
print(f"\nAnalyzing {len(df_success)} successful tests...")

if len(df_success) == 0:
    print("✗ No successful tests to analyze!")
    exit(1)

# Create weighted score
df_success['PESQ_normalized'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_normalized'] + 0.5 * df_success['STOI']

# ====================================
# OVERALL BEST CONFIGURATIONS
# ====================================
print("\n" + "="*100)
print("TOP 10 OVERALL CONFIGURATIONS")
print("="*100)

metrics_to_rank = ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', 'Weighted_Score']

for metric in metrics_to_rank:
    print(f"\n--- Top 10 by {metric} ---")
    top10 = df_success.nlargest(10, metric)[
        ['Config_ID', 'Noise_Category', 'Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 
         'PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', metric]
    ]
    print(top10.to_string(index=False))
    top10.to_csv(OUTPUT_DIR / f'top10_overall_by_{metric}.csv', index=False)

# ====================================
# BEST BY NOISE CATEGORY
# ====================================
print("\n" + "="*100)
print("BEST CONFIGURATIONS BY NOISE CATEGORY")
print("="*100)

for category in df_success['Noise_Category'].unique():
    print(f"\n{'='*100}")
    print(f"Category: {category}")
    print(f"{'='*100}")
    
    cat_data = df_success[df_success['Noise_Category'] == category]
    
    # Best by each metric
    for metric in ['PESQ', 'STOI', 'SI_SDR', 'Weighted_Score']:
        best = cat_data.loc[cat_data[metric].idxmax()]
        print(f"\nBest {metric}: {best[metric]:.4f}")
        print(f"  Config: Freq={best['Freq_spacing']}, Nband={int(best['Nband'])}, " +
              f"Frame={int(best['FRMSZ_ms'])}ms, Overlap={int(best['OVLP'])}%")
    
    # Save top 5 for this category
    top5 = cat_data.nlargest(5, 'Weighted_Score')[
        ['Config_ID', 'Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP',
         'PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', 'Weighted_Score']
    ]
    top5.to_csv(OUTPUT_DIR / f'top5_{category.lower()}.csv', index=False)

# ====================================
# AGGREGATE BY PARAMETERS
# ====================================
print("\n" + "="*100)
print("AVERAGE METRICS BY PARAMETER VALUE")
print("="*100)

summary_stats = []

for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP']:
    print(f"\n{'='*80}")
    print(f"Parameter: {param}")
    print(f"{'='*80}")
    
    grouped = df_success.groupby(param).agg({
        'PESQ': ['mean', 'std', 'min', 'max'],
        'STOI': ['mean', 'std', 'min', 'max'],
        'SI_SDR': ['mean', 'std', 'min', 'max'],
        'DNSMOS_mos_ovr': ['mean', 'std', 'min', 'max'],
        'Weighted_Score': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(grouped)
    
    # Store for CSV
    for val in df_success[param].unique():
        subset = df_success[df_success[param] == val]
        summary_stats.append({
            'Parameter': param,
            'Value': val,
            'PESQ_mean': subset['PESQ'].mean(),
            'PESQ_std': subset['PESQ'].std(),
            'STOI_mean': subset['STOI'].mean(),
            'STOI_std': subset['STOI'].std(),
            'SI_SDR_mean': subset['SI_SDR'].mean(),
            'SI_SDR_std': subset['SI_SDR'].std(),
            'DNSMOS_mean': subset['DNSMOS_mos_ovr'].mean(),
            'DNSMOS_std': subset['DNSMOS_mos_ovr'].std(),
            'Weighted_mean': subset['Weighted_Score'].mean(),
            'Weighted_std': subset['Weighted_Score'].std(),
            'Count': len(subset)
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(OUTPUT_DIR / 'parameter_summary_statistics.csv', index=False)

# ====================================
# VISUALIZATIONS
# ====================================
print("\n" + "="*100)
print("GENERATING VISUALIZATIONS")
print("="*100)

# 1. Box plots by parameter
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Impact of {param} on Speech Enhancement Metrics', 
                 fontsize=16, fontweight='bold')
    
    metrics_plot = [
        ('PESQ', axes[0,0], 'Perceptual Evaluation of Speech Quality'),
        ('STOI', axes[0,1], 'Short-Time Objective Intelligibility'),
        ('SI_SDR', axes[1,0], 'Scale-Invariant SDR (dB)'),
        ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall Quality')
    ]
    
    for metric, ax, ylabel in metrics_plot:
        df_plot = df_success.copy()
        if param in ['Nband', 'FRMSZ_ms', 'OVLP', 'Noisefr']:
            df_plot[param] = df_plot[param].astype(str)
        elif param == 'Floor':
            # Format floor values nicely
            df_plot[param] = df_plot[param].apply(lambda x: f"{x:.3f}")
        
        sns.boxplot(data=df_plot, x=param, y=metric, ax=ax, palette='Set2')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'boxplot_{param}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: boxplot_{param}.png")
    plt.close()

# 2. Box plots by noise category
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Performance by Noise Category', fontsize=16, fontweight='bold')

for (metric, ax, ylabel) in [
    ('PESQ', axes[0,0], 'PESQ'),
    ('STOI', axes[0,1], 'STOI'),
    ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
    ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
]:
    sns.boxplot(data=df_success, x='Noise_Category', y=metric, ax=ax, palette='Set1')
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Noise Category', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_by_noise_category.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: performance_by_noise_category.png")
plt.close()

# 3. Heatmap: Freq_spacing vs Nband
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parameter Interaction: Frequency Spacing × Number of Bands', 
             fontsize=16, fontweight='bold')

for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                        axes.flat)):
    pivot = df_success.pivot_table(values=metric, index='Freq_spacing', 
                                     columns='Nband', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': metric}, vmin=pivot.min().min(), vmax=pivot.max().max())
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Bands', fontsize=11)
    ax.set_ylabel('Frequency Spacing', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_freq_vs_nband.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_freq_vs_nband.png")
plt.close()

# 4. Heatmap: FRMSZ vs OVLP
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parameter Interaction: Frame Size × Overlap', 
             fontsize=16, fontweight='bold')

for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                        axes.flat)):
    pivot = df_success.pivot_table(values=metric, index='FRMSZ_ms', 
                                     columns='OVLP', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': metric})
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Overlap (%)', fontsize=11)
    ax.set_ylabel('Frame Size (ms)', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_frame_vs_overlap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_frame_vs_overlap.png")
plt.close()

# 4b. CRITICAL: Heatmap FLOOR vs Noise Category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CRITICAL: Floor Parameter × Noise Category', 
             fontsize=16, fontweight='bold')

for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                        axes.flat)):
    pivot = df_success.pivot_table(values=metric, index='Noise_Category', 
                                     columns='Floor', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': metric})
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Floor Parameter', fontsize=11)
    ax.set_ylabel('Noise Category', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_CRITICAL_floor_vs_noise.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_CRITICAL_floor_vs_noise.png")
plt.close()

# 4c. Heatmap: Noisefr vs Noise Category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Noisefr Parameter × Noise Category', 
             fontsize=16, fontweight='bold')

for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                        axes.flat)):
    pivot = df_success.pivot_table(values=metric, index='Noise_Category', 
                                     columns='Noisefr', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': metric})
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Noise Frames', fontsize=11)
    ax.set_ylabel('Noise Category', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_noisefr_vs_noise.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_noisefr_vs_noise.png")
plt.close()

# 5. Metric distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Distribution of Speech Enhancement Metrics', 
             fontsize=16, fontweight='bold')

for metric, ax in zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], axes.flat):
    df_success[metric].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(df_success[metric].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df_success[metric].mean():.3f}')
    ax.axvline(df_success[metric].median(), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {df_success[metric].median():.3f}')
    ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metric_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: metric_distributions.png")
plt.close()

# 6. PESQ vs STOI scatter (colored by noise category)
fig, ax = plt.subplots(figsize=(12, 8))
categories = df_success['Noise_Category'].unique()
colors = sns.color_palette('Set1', len(categories))

for category, color in zip(categories, colors):
    cat_data = df_success[df_success['Noise_Category'] == category]
    ax.scatter(cat_data['PESQ'], cat_data['STOI'], 
               label=category, alpha=0.6, s=50, color=color)

ax.set_xlabel('PESQ', fontsize=12, fontweight='bold')
ax.set_ylabel('STOI', fontsize=12, fontweight='bold')
ax.set_title('PESQ vs STOI by Noise Category', fontsize=14, fontweight='bold')
ax.legend(title='Noise Category', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pesq_vs_stoi_scatter.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: pesq_vs_stoi_scatter.png")
plt.close()

# 7. Grouped bar chart: Average metrics by frequency spacing
fig, ax = plt.subplots(figsize=(14, 8))
metrics_avg = df_success.groupby('Freq_spacing')[['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']].mean()
metrics_avg.plot(kind='bar', ax=ax, width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Average Metrics by Frequency Spacing', fontsize=14, fontweight='bold')
ax.set_xlabel('Frequency Spacing', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Metric', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'avg_metrics_by_freq_spacing.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: avg_metrics_by_freq_spacing.png")
plt.close()

# ====================================
# BEST CONFIGURATION SUMMARY
# ====================================
print("\n" + "="*100)
print("BEST OVERALL CONFIGURATION (Weighted Score)")
print("="*100)

best_overall = df_success.loc[df_success['Weighted_Score'].idxmax()]
print("\nConfiguration:")
print(f"  Config ID: {best_overall['Config_ID']}")
print(f"  Freq_spacing: {best_overall['Freq_spacing']}")
print(f"  Nband: {int(best_overall['Nband'])}")
print(f"  Frame Size: {int(best_overall['FRMSZ_ms'])} ms")
print(f"  Overlap: {int(best_overall['OVLP'])}%")
# print(f"  Noise: {best_overall['Noise_File']} ({best_overall['Noise_Category']})")

print("\nMetrics:")
print(f"  PESQ: {best_overall['PESQ']:.4f}")
print(f"  STOI: {best_overall['STOI']:.4f}")
print(f"  SI-SDR: {best_overall['SI_SDR']:.2f} dB")
print(f"  DNSMOS Overall: {best_overall['DNSMOS_mos_ovr']:.4f}")
print(f"  Weighted Score: {best_overall['Weighted_Score']:.4f}")

best_config_df = pd.DataFrame([best_overall])
best_config_df.to_csv(OUTPUT_DIR / 'best_overall_configuration.csv', index=False)

# ====================================
# SUMMARY REPORT
# ====================================
summary_report = f"""
PARAMETER SWEEP ANALYSIS SUMMARY
{'='*100}

Dataset Statistics:
  Total configurations: {df['Config_ID'].nunique()}
  Total tests: {len(df)}
  Successful tests: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)
  Noise categories: {', '.join(df_success['Noise_Category'].unique())}

Overall Performance (mean ± std):
  PESQ: {df_success['PESQ'].mean():.3f} ± {df_success['PESQ'].std():.3f}
  STOI: {df_success['STOI'].mean():.3f} ± {df_success['STOI'].std():.3f}
  SI-SDR: {df_success['SI_SDR'].mean():.2f} ± {df_success['SI_SDR'].std():.2f} dB
  DNSMOS: {df_success['DNSMOS_mos_ovr'].mean():.3f} ± {df_success['DNSMOS_mos_ovr'].std():.3f}

BEST OVERALL CONFIGURATION:
  Frequency Spacing: {best_overall['Freq_spacing']}
  Number of Bands: {int(best_overall['Nband'])}
  Frame Size: {int(best_overall['FRMSZ_ms'])} ms
  Overlap: {int(best_overall['OVLP'])}%
  
  Performance:
    PESQ: {best_overall['PESQ']:.4f}
    STOI: {best_overall['STOI']:.4f}
    SI-SDR: {best_overall['SI_SDR']:.2f} dB
    DNSMOS: {best_overall['DNSMOS_mos_ovr']:.4f}
    Weighted Score: {best_overall['Weighted_Score']:.4f}

PARAMETER INSIGHTS:
"""

# Add best values for each parameter
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP']:
    best_val = df_success.groupby(param)['Weighted_Score'].mean().idxmax()
    best_score = df_success.groupby(param)['Weighted_Score'].mean().max()
    summary_report += f"\n  Best {param}: {best_val} (avg weighted score: {best_score:.4f})"

# Add performance by noise category
summary_report += "\n\nPERFORMANCE BY NOISE CATEGORY (avg PESQ):"
for category in df_success['Noise_Category'].unique():
    cat_avg = df_success[df_success['Noise_Category'] == category]['PESQ'].mean()
    summary_report += f"\n  {category}: {cat_avg:.3f}"

# Add best floor by noise category
summary_report += "\n\nRECOMMENDED FLOOR BY NOISE TYPE:"
for category in df_success['Noise_Category'].unique():
    cat_data = df_success[df_success['Noise_Category'] == category]
    best_floor = cat_data.groupby('Floor')['Weighted_Score'].mean().idxmax()
    best_score = cat_data.groupby('Floor')['Weighted_Score'].mean().max()
    summary_report += f"\n  {category}: Floor={best_floor} (score: {best_score:.4f})"

summary_report += f"\n\nAll results saved to: {OUTPUT_DIR.absolute()}\n"

print("\n" + summary_report)

# Save report
with open(OUTPUT_DIR / 'analysis_summary_report.txt', 'w') as f:
    f.write(summary_report)

print("="*100)
print("ANALYSIS COMPLETE!")
print(f"Output directory: {OUTPUT_DIR.absolute()}")
print("="*100)
print("\nGenerated files:")
print("  CSV files:")
print("    - top10_overall_by_*.csv (top configs by metric)")
print("    - top5_*.csv (top configs per noise category)")
print("    - parameter_summary_statistics.csv")
print("    - best_overall_configuration.csv")
print("  Visualizations:")
print("    - boxplot_*.png (parameter effects)")
print("    - heatmap_*.png (parameter interactions)")
print("    - heatmap_CRITICAL_floor_vs_noise.png (MOST IMPORTANT)")
print("    - heatmap_noisefr_vs_noise.png")
print("    - performance_by_noise_category.png")
print("    - metric_distributions.png")
print("    - pesq_vs_stoi_scatter.png")
print("    - avg_metrics_by_freq_spacing.png")
print("  Report:")
print("    - analysis_summary_report.txt")
print("="*100)