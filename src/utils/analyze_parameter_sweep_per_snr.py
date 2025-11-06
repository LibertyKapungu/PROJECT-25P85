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
OUTPUT_DIR = Path(f"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP3/parameter_sweep_analysis_per_snrw_{MODE}")
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
    print(f" Loaded {len(df)} rows successfully")
except Exception as e:
    print(f" Error loading CSV: {e}")
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
# <--- MODIFIED: Added SNR_dB to the list
for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr', 'SNR_dB']:
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

# <--- MODIFIED: Added DNSMOS fallback check
# Rename DNSMOS column if needed
if 'DNSMOS_mos_ovr' not in df_success.columns and 'DNSMOS_p808_mos' in df_success.columns:
    print("Warning: 'DNSMOS_mos_ovr' not found. Using 'DNSMOS_p808_mos' as fallback.")
    df_success['DNSMOS_mos_ovr'] = df_success['DNSMOS_p808_mos']

# Create weighted score
df_success['PESQ_normalized'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_normalized'] + 0.5 * df_success['STOI']

# ====================================
# <--- NEW SECTION: MAIN SNR LOOP
# ====================================
# All analysis will now happen inside this loop

# --- IMPORTANT ---
# This is the critical change: using your 'SNR_dB' column
SNR_COLUMN = 'SNR_dB' 

if SNR_COLUMN not in df_success.columns:
    print("\n" + "!"*100)
    print(f" ERROR: SNR column '{SNR_COLUMN}' not found in the CSV.")
    print("  Cannot split analysis by SNR.")
    print("  Please check your CSV or update the 'SNR_COLUMN' variable in the script.")
    print("!"*100)
    exit(1)

# Get sorted, unique SNR levels
snr_levels = sorted(df_success[SNR_COLUMN].unique())
print(f"\nFound {len(snr_levels)} SNR levels to analyze: {snr_levels}")

for snr in snr_levels:
    print("\n" + "#"*100)
    print(f" PROCESSING SNR LEVEL: {snr} dB ")
    print("#"*100)
    
    # 1. Create an SNR-specific output directory
    snr_output_dir = OUTPUT_DIR / f"SNR_{snr}"
    snr_output_dir.mkdir(exist_ok=True)
    
    # 2. Filter the successful dataframe for *only* this SNR
    df_snr_success = df_success[df_success[SNR_COLUMN] == snr].copy()
    
    # 3. Check if there's any data for this SNR
    if len(df_snr_success) == 0:
        print(f"✗ No successful tests found for SNR {snr}. Skipping.")
        continue
        
    print(f" Analyzing {len(df_snr_success)} successful tests for SNR {snr}...")

    # ====================================
    # OVERALL BEST CONFIGURATIONS (FOR THIS SNR)
    # ====================================
    print("\n" + "="*100)
    # <--- MODIFIED: Updated print statements
    print(f"TOP 10 OVERALL CONFIGURATIONS (SNR: {snr} dB)")
    print("="*100)

    metrics_to_rank = ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', 'Weighted_Score']

    for metric in metrics_to_rank:
        # <--- MODIFIED: Updated print statements
        print(f"\n--- Top 10 by {metric} (SNR: {snr}) ---")
        top10 = df_snr_success.nlargest(10, metric)[  # <--- MODIFIED: Use df_snr_success
            ['Config_ID', 'Noise_Category', 'Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 
             'PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr', metric]
        ]
        print(top10.to_string(index=False))
        # <--- MODIFIED: Save to snr_output_dir
        top10.to_csv(snr_output_dir / f'top10_overall_by_{metric}.csv', index=False)

    # ====================================
    # BEST BY NOISE CATEGORY (FOR THIS SNR)
    # ====================================
    print("\n" + "="*100)
    # <--- MODIFIED: Updated print statements
    print(f"BEST CONFIGURATIONS BY NOISE CATEGORY (SNR: {snr} dB)")
    print("="*100)

    for category in df_snr_success['Noise_Category'].unique(): # <--- MODIFIED: Use df_snr_success
        print(f"\n{'='*100}")
        print(f"Category: {category} (SNR: {snr})")
        print(f"{'='*100}")
        
        # <--- MODIFIED: Use df_snr_success
        cat_data = df_snr_success[df_snr_success['Noise_Category'] == category]
        
        if len(cat_data) == 0:
            print("  No data for this category at this SNR. Skipping.")
            continue
            
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
        # <--- MODIFIED: Save to snr_output_dir
        top5.to_csv(snr_output_dir / f'top5_{category.lower()}.csv', index=False)

    # ====================================
    # AGGREGATE BY PARAMETERS (FOR THIS SNR)
    # ====================================
    print("\n" + "="*100)
    # <--- MODIFIED: Updated print statements
    print(f"AVERAGE METRICS BY PARAMETER VALUE (SNR: {snr} dB)")
    print("="*100)

    summary_stats = []

    # <--- MODIFIED: Added 'Floor' and 'Noisefr' to this list
    for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
        if param not in df_snr_success.columns: continue # Skip if param doesn't exist
        
        print(f"\n{'='*80}")
        print(f"Parameter: {param} (SNR: {snr})")
        print(f"{'='*80}")
        
        # <--- MODIFIED: Use df_snr_success
        grouped = df_snr_success.groupby(param).agg({
            'PESQ': ['mean', 'std', 'min', 'max'],
            'STOI': ['mean', 'std', 'min', 'max'],
            'SI_SDR': ['mean', 'std', 'min', 'max'],
            'DNSMOS_mos_ovr': ['mean', 'std', 'min', 'max'],
            'Weighted_Score': ['mean', 'std', 'min', 'max']
        }).round(4)
        print(grouped)
        
        # Store for CSV
        for val in df_snr_success[param].unique(): # <--- MODIFIED: Use df_snr_success
            subset = df_snr_success[df_snr_success[param] == val] # <--- MODIFIED: Use df_snr_success
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
    # <--- MODIFIED: Save to snr_output_dir
    summary_df.to_csv(snr_output_dir / 'parameter_summary_statistics.csv', index=False)

    # ====================================
    # VISUALIZATIONS (FOR THIS SNR)
    # ====================================
    print("\n" + "="*100)
    # <--- MODIFIED: Updated print statements
    print(f"GENERATING VISUALIZATIONS (SNR: {snr} dB)")
    print("="*100)

    # 1. Box plots by parameter
    # <--- MODIFIED: Added check for column existence
    for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
        if param not in df_snr_success.columns:
            print(f"  Skipping plot for param '{param}': not in columns.")
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        # <--- MODIFIED: Updated title
        fig.suptitle(f'Impact of {param} on Speech Enhancement Metrics (SNR: {snr} dB)', 
                     fontsize=16, fontweight='bold')
        
        metrics_plot = [
            ('PESQ', axes[0,0], 'Perceptual Evaluation of Speech Quality'),
            ('STOI', axes[0,1], 'Short-Time Objective Intelligibility'),
            ('SI_SDR', axes[1,0], 'Scale-Invariant SDR (dB)'),
            ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall Quality')
        ]
        
        for metric, ax, ylabel in metrics_plot:
            if metric not in df_snr_success.columns: continue
            df_plot = df_snr_success.copy() # <--- MODIFIED: Use df_snr_success
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
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / f'boxplot_{param}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: boxplot_{param}.png (SNR: {snr})")
        plt.close()

    # 2. Box plots by noise category
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # <--- MODIFIED: Updated title
    fig.suptitle(f'Performance by Noise Category (SNR: {snr} dB)', fontsize=16, fontweight='bold')

    for (metric, ax, ylabel) in [
        ('PESQ', axes[0,0], 'PESQ'),
        ('STOI', axes[0,1], 'STOI'),
        ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
        ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
    ]:
        if metric not in df_snr_success.columns: continue
        # <--- MODIFIED: Use df_snr_success
        sns.boxplot(data=df_snr_success, x='Noise_Category', y=metric, ax=ax, palette='Set1')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Category', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # <--- MODIFIED: Save to snr_output_dir
    plt.savefig(snr_output_dir / 'performance_by_noise_category.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_by_noise_category.png (SNR: {snr})")
    plt.close()

    # 3. Heatmap: Freq_spacing vs Nband
    if 'Freq_spacing' in df_snr_success.columns and 'Nband' in df_snr_success.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # <--- MODIFIED: Updated title
        fig.suptitle(f'Parameter Interaction: Frequency Spacing × Number of Bands (SNR: {snr} dB)', 
                     fontsize=16, fontweight='bold')

        for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                               axes.flat)):
            if metric not in df_snr_success.columns: continue
            # <--- MODIFIED: Use df_snr_success
            pivot = df_snr_success.pivot_table(values=metric, index='Freq_spacing', 
                                               columns='Nband', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                        cbar_kws={'label': metric}, vmin=pivot.min().min(), vmax=pivot.max().max())
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Bands', fontsize=11)
            ax.set_ylabel('Frequency Spacing', fontsize=11)

        plt.tight_layout()
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / 'heatmap_freq_vs_nband.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_freq_vs_nband.png (SNR: {snr})")
        plt.close()

    # 4. Heatmap: FRMSZ vs OVLP
    if 'FRMSZ_ms' in df_snr_success.columns and 'OVLP' in df_snr_success.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # <--- MODIFIED: Updated title
        fig.suptitle(f'Parameter Interaction: Frame Size × Overlap (SNR: {snr} dB)', 
                     fontsize=16, fontweight='bold')

        for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                               axes.flat)):
            if metric not in df_snr_success.columns: continue
            # <--- MODIFIED: Use df_snr_success
            pivot = df_snr_success.pivot_table(values=metric, index='FRMSZ_ms', 
                                               columns='OVLP', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                        cbar_kws={'label': metric})
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Overlap (%)', fontsize=11)
            ax.set_ylabel('Frame Size (ms)', fontsize=11)

        plt.tight_layout()
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / 'heatmap_frame_vs_overlap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_frame_vs_overlap.png (SNR: {snr})")
        plt.close()

    # 4b. CRITICAL: Heatmap FLOOR vs Noise Category
    if 'Floor' in df_snr_success.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # <--- MODIFIED: Updated title
        fig.suptitle(f'CRITICAL: Floor Parameter × Noise Category (SNR: {snr} dB)', 
                     fontsize=16, fontweight='bold')

        for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                               axes.flat)):
            if metric not in df_snr_success.columns: continue
            # <--- MODIFIED: Use df_snr_success
            pivot = df_snr_success.pivot_table(values=metric, index='Noise_Category', 
                                               columns='Floor', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                        cbar_kws={'label': metric})
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Floor Parameter', fontsize=11)
            ax.set_ylabel('Noise Category', fontsize=11)

        plt.tight_layout()
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / 'heatmap_CRITICAL_floor_vs_noise.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_CRITICAL_floor_vs_noise.png (SNR: {snr})")
        plt.close()

    # 4c. Heatmap: Noisefr vs Noise Category
    if 'Noisefr' in df_snr_success.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # <--- MODIFIED: Updated title
        fig.suptitle(f'Noisefr Parameter × Noise Category (SNR: {snr} dB)', 
                     fontsize=16, fontweight='bold')

        for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                               axes.flat)):
            if metric not in df_snr_success.columns: continue
            # <--- MODIFIED: Use df_snr_success
            pivot = df_snr_success.pivot_table(values=metric, index='Noise_Category', 
                                               columns='Noisefr', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                        cbar_kws={'label': metric})
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Noise Frames', fontsize=11)
            ax.set_ylabel('Noise Category', fontsize=11)

        plt.tight_layout()
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / 'heatmap_noisefr_vs_noise.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_noisefr_vs_noise.png (SNR: {snr})")
        plt.close()

    # 5. Metric distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # <--- MODIFIED: Updated title
    fig.suptitle(f'Distribution of Speech Enhancement Metrics (SNR: {snr} dB)', 
                 fontsize=16, fontweight='bold')

    for metric, ax in zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], axes.flat):
        if metric not in df_snr_success.columns: continue
        # <--- MODIFIED: Use df_snr_success
        df_snr_success[metric].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(df_snr_success[metric].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df_snr_success[metric].mean():.3f}')
        ax.axvline(df_snr_success[metric].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df_snr_success[metric].median():.3f}')
        ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # <--- MODIFIED: Save to snr_output_dir
    plt.savefig(snr_output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metric_distributions.png (SNR: {snr})")
    plt.close()

    # 6. PESQ vs STOI scatter (colored by noise category)
    fig, ax = plt.subplots(figsize=(12, 8))
    # <--- MODIFIED: Use df_snr_success
    categories = df_snr_success['Noise_Category'].unique()
    colors = sns.color_palette('Set1', len(categories))

    for category, color in zip(categories, colors):
        # <--- MODIFIED: Use df_snr_success
        cat_data = df_snr_success[df_snr_success['Noise_Category'] == category]
        ax.scatter(cat_data['PESQ'], cat_data['STOI'], 
                   label=category, alpha=0.6, s=50, color=color)

    ax.set_xlabel('PESQ', fontsize=12, fontweight='bold')
    ax.set_ylabel('STOI', fontsize=12, fontweight='bold')
    # <--- MODIFIED: Updated title
    ax.set_title(f'PESQ vs STOI by Noise Category (SNR: {snr} dB)', fontsize=14, fontweight='bold')
    ax.legend(title='Noise Category', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # <--- MODIFIED: Save to snr_output_dir
    plt.savefig(snr_output_dir / 'pesq_vs_stoi_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: pesq_vs_stoi_scatter.png (SNR: {snr})")
    plt.close()

    # 7. Grouped bar chart: Average metrics by frequency spacing
    if 'Freq_spacing' in df_snr_success.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        # <--- MODIFIED: Use df_snr_success
        metrics_avg = df_snr_success.groupby('Freq_spacing')[['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']].mean()
        metrics_avg.plot(kind='bar', ax=ax, width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        # <--- MODIFIED: Updated title
        ax.set_title(f'Average Metrics by Frequency Spacing (SNR: {snr} dB)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency Spacing', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Metric', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        # <--- MODIFIED: Save to snr_output_dir
        plt.savefig(snr_output_dir / 'avg_metrics_by_freq_spacing.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: avg_metrics_by_freq_spacing.png (SNR: {snr})")
        plt.close()

    # ====================================
    # BEST CONFIGURATION SUMMARY (FOR THIS SNR)
    # ====================================
    print("\n" + "="*100)
    # <--- MODIFIED: Updated print statements
    print(f"BEST OVERALL CONFIGURATION (Weighted Score) (SNR: {snr} dB)")
    print("="*100)

    # <--- MODIFIED: Use df_snr_success
    best_overall = df_snr_success.loc[df_snr_success['Weighted_Score'].idxmax()]
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
    # <--- MODIFIED: Save to snr_output_dir
    best_config_df.to_csv(snr_output_dir / 'best_overall_configuration.csv', index=False)

    # ====================================
    # SUMMARY REPORT (FOR THIS SNR)
    # ====================================
    
    # <--- MODIFIED: This entire report string is updated to use df_snr_success
    summary_report = f"""
PARAMETER SWEEP ANALYSIS SUMMARY (SNR: {snr} dB)
{'='*100}

Dataset Statistics:
  Total configurations: {df_snr_success['Config_ID'].nunique()}
  Total tests: {len(df_snr_success)}
  Successful tests: {len(df_snr_success)} (100.0% of this SNR subset)
  Noise categories: {', '.join(df_snr_success['Noise_Category'].unique())}

Overall Performance (mean ± std) for SNR {snr} dB:
  PESQ: {df_snr_success['PESQ'].mean():.3f} ± {df_snr_success['PESQ'].std():.3f}
  STOI: {df_snr_success['STOI'].mean():.3f} ± {df_snr_success['STOI'].std():.3f}
  SI-SDR: {df_snr_success['SI_SDR'].mean():.2f} ± {df_snr_success['SI_SDR'].std():.2f} dB
  DNSMOS: {df_snr_success['DNSMOS_mos_ovr'].mean():.3f} ± {df_snr_success['DNSMOS_mos_ovr'].std():.3f}

BEST OVERALL CONFIGURATION (for SNR {snr} dB):
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

PARAMETER INSIGHTS (for SNR {snr} dB):
"""

    # Add best values for each parameter
    for param in ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']:
        if param in df_snr_success.columns:
            best_val = df_snr_success.groupby(param)['Weighted_Score'].mean().idxmax()
            best_score = df_snr_success.groupby(param)['Weighted_Score'].mean().max()
            summary_report += f"\n  Best {param}: {best_val} (avg weighted score: {best_score:.4f})"

    # Add performance by noise category
    summary_report += f"\n\nPERFORMANCE BY NOISE CATEGORY (avg PESQ for SNR {snr} dB):"
    for category in df_snr_success['Noise_Category'].unique():
        cat_avg = df_snr_success[df_snr_success['Noise_Category'] == category]['PESQ'].mean()
        summary_report += f"\n  {category}: {cat_avg:.3f}"

    # Add best floor by noise category
    if 'Floor' in df_snr_success.columns:
        summary_report += f"\n\nRECOMMENDED FLOOR BY NOISE TYPE (for SNR {snr} dB):"
        for category in df_snr_success['Noise_Category'].unique():
            cat_data = df_snr_success[df_snr_success['Noise_Category'] == category]
            if len(cat_data) > 0:
                best_floor = cat_data.groupby('Floor')['Weighted_Score'].mean().idxmax()
                best_score = cat_data.groupby('Floor')['Weighted_Score'].mean().max()
                summary_report += f"\n  {category}: Floor={best_floor} (score: {best_score:.4f})"

    summary_report += f"\n\nAll results for SNR {snr} saved to: {snr_output_dir.absolute()}\n"

    print("\n" + summary_report)

    # Save report
    # <--- MODIFIED: Save to snr_output_dir
    with open(snr_output_dir / 'analysis_summary_report.txt', 'w') as f:
        f.write(summary_report)

# <--- MODIFIED: This is the end of the new 'for snr in snr_levels:' loop
# The final summary print block is now updated to reflect the new structure.

print("\n" + "="*100)
print("SNR-SPECIFIC ANALYSIS COMPLETE!")
print(f"Output saved to subfolders in: {OUTPUT_DIR.absolute()}")
print("="*100)
print("\nGenerated files in EACH 'SNR_X' subfolder:")
print("  CSV files:")
print("  - top10_overall_by_*.csv (top configs by metric)")
print("  - top5_*.csv (top configs per noise category)")
print("  - parameter_summary_statistics.csv")
print("  - best_overall_configuration.csv")
print("  Visualizations:")
print("  - boxplot_*.png (parameter effects)")
print("  - heatmap_*.png (parameter interactions)")
print("  - heatmap_CRITICAL_floor_vs_noise.png (MOST IMPORTANT)")
print("  - heatmap_noisefr_vs_noise.png")
print("  - performance_by_noise_category.png")
print("  - metric_distributions.png")
print("  - pesq_vs_stoi_scatter.png")
print("  - avg_metrics_by_freq_spacing.png")
print("  Report:")
print("  - analysis_summary_report.txt")
print("="*100)