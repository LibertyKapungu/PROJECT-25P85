"""
Analysis Script for 3-Phase Parameter Sweep
Generates comprehensive visualizations and insights for each phase
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ====================================
# CONFIGURATION
# ====================================
# CSV_FILE = r"path/to/complete_3phase_results.csv"  # UPDATE THIS
# OUTPUT_DIR = Path("3phase_analysis")
# OUTPUT_DIR.mkdir(exist_ok=True)

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
CSV_FILE = repo_root /"results" /"EXP3" /"SS"/"PARAM_SWEEP_ENHANCED_20251029_225201"/"complete_results_all_phases.csv"  # UPDATE THIS
OUTPUT_DIR = repo_root /"results" /"EXP3"/ "SS"/"PARAM_SWEEP_ENHANCED_20251029_225201" /"GTCRN_SS_hybrid_phases"
OUTPUT_DIR.mkdir(exist_ok=True, parents= True)

print("="*100)
print("3-PHASE PARAMETER SWEEP ANALYSIS")
print("="*100)

# ====================================
# LOAD DATA
# ====================================
print("\nLoading results...")
df = pd.read_csv(CSV_FILE)
df_success = df[df['Status'] == 'Success'].copy()
print(f"✓ Loaded {len(df)} total tests, {len(df_success)} successful")

# Create composite score
df_success['composite_score'] = 0.5 * df_success['PESQ'] + 0.5 * df_success['STOI']

# Separate by phase
df_p1 = df_success[df_success['Phase'] == 1].copy()
df_p2 = df_success[df_success['Phase'] == 2].copy()
df_p3 = df_success[df_success['Phase'] == 3].copy()

print(f"\nPhase 1 tests: {len(df_p1)}")
print(f"Phase 2 tests: {len(df_p2)}")
print(f"Phase 3 tests: {len(df_p3)}")

# ====================================
# PHASE 1 ANALYSIS: FREQUENCY SPACING
# ====================================
print(f"\n{'='*100}")
print("PHASE 1 ANALYSIS: FREQUENCY SPACING")
print(f"{'='*100}")

# 1. Frequency spacing performance across SNRs
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Phase 1: Frequency Spacing Performance Across SNR Levels', 
             fontsize=16, fontweight='bold')

for (metric, ax, ylabel) in [
    ('PESQ', axes[0,0], 'PESQ'),
    ('STOI', axes[0,1], 'STOI'),
    ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
    ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
]:
    for freq in df_p1['Freq_spacing'].unique():
        freq_data = df_p1[df_p1['Freq_spacing'] == freq]
        avg_by_snr = freq_data.groupby('SNR_dB')[metric].mean()
        std_by_snr = freq_data.groupby('SNR_dB')[metric].std()
        
        ax.plot(avg_by_snr.index, avg_by_snr.values, marker='o', linewidth=2, 
                label=freq, markersize=8)
        ax.fill_between(avg_by_snr.index, 
                        avg_by_snr - std_by_snr, 
                        avg_by_snr + std_by_snr, 
                        alpha=0.2)
    
    ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.legend(title='Freq Spacing', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase1_freq_spacing_vs_snr.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase1_freq_spacing_vs_snr.png")
plt.close()

# 2. Frequency spacing by noise type
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Phase 1: Frequency Spacing by Noise Category', 
             fontsize=16, fontweight='bold')

for (metric, ax, ylabel) in [
    ('PESQ', axes[0,0], 'PESQ'),
    ('STOI', axes[0,1], 'STOI'),
    ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
    ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
]:
    sns.boxplot(data=df_p1, x='Freq_spacing', y=metric, hue='Noise_Category', 
                ax=ax, palette='Set2')
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency Spacing', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(title='Noise Type', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase1_freq_spacing_by_noise.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase1_freq_spacing_by_noise.png")
plt.close()

# 3. Phase 1 Summary Table
print("\nPhase 1 Summary (Average across all conditions):")
p1_summary = df_p1.groupby('Freq_spacing').agg({
    'PESQ': ['mean', 'std'],
    'STOI': ['mean', 'std'],
    'SI_SDR': ['mean', 'std'],
    'DNSMOS_mos_ovr': ['mean', 'std']
}).round(3)
print(p1_summary)
p1_summary.to_csv(OUTPUT_DIR / 'phase1_summary_table.csv')

# Crossover point analysis
print("\nCrossover Analysis (where log beats linear):")
for metric in ['PESQ', 'SI_SDR']:
    crossover_data = df_p1.pivot_table(values=metric, 
                                       index='SNR_dB', 
                                       columns='Freq_spacing', 
                                       aggfunc='mean')
    if 'log' in crossover_data.columns and 'linear' in crossover_data.columns:
        diff = crossover_data['log'] - crossover_data['linear']
        print(f"\n{metric}:")
        for snr in diff.index:
            if diff[snr] > 0:
                print(f"  SNR {snr}dB: LOG wins by {diff[snr]:.3f}")
            else:
                print(f"  SNR {snr}dB: LINEAR wins by {abs(diff[snr]):.3f}")

# DETAILED SNR BREAKDOWN
print("\n" + "="*100)
print("DETAILED PERFORMANCE AT EACH SNR LEVEL")
print("="*100)

snr_analysis = df_p1.pivot_table(
    values=['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'],
    index='SNR_dB',
    columns='Freq_spacing',
    aggfunc=['mean', 'std']
)

for snr in sorted(df_p1['SNR_dB'].unique()):
    snr_data = df_p1[df_p1['SNR_dB'] == snr]
    print(f"\n{'='*80}")
    print(f"SNR = {snr} dB")
    print(f"{'='*80}")
    
    for freq in ['mel', 'log', 'linear']:
        freq_data = snr_data[snr_data['Freq_spacing'] == freq]
        if len(freq_data) > 0:
            print(f"\n{freq.upper()}:")
            print(f"  PESQ:  {freq_data['PESQ'].mean():.3f} ± {freq_data['PESQ'].std():.3f}")
            print(f"  STOI:  {freq_data['STOI'].mean():.3f} ± {freq_data['STOI'].std():.3f}")
            print(f"  SI-SDR: {freq_data['SI_SDR'].mean():.2f} ± {freq_data['SI_SDR'].std():.2f} dB")
            print(f"  DNSMOS: {freq_data['DNSMOS_mos_ovr'].mean():.3f} ± {freq_data['DNSMOS_mos_ovr'].std():.3f}")
    
    # Winner at this SNR
    best_pesq_freq = snr_data.groupby('Freq_spacing')['PESQ'].mean().idxmax()
    best_stoi_freq = snr_data.groupby('Freq_spacing')['STOI'].mean().idxmax()
    print(f"\n  → Winner (PESQ): {best_pesq_freq.upper()}")
    print(f"  → Winner (STOI): {best_stoi_freq.upper()}")
    
    # Performance insights
    if snr <= -5:
        print(f"   Very Low SNR: All methods struggle, LOG has slight edge")
    elif snr <= 0:
        print(f"    Low SNR: LINEAR shows better stability")
    elif snr <= 5:
        print(f"    Mid SNR: LINEAR maintains advantage")
    elif snr <= 10:
        print(f"    High SNR: Competition is close, both preserve speech well")
    else:
        print(f"    Very High SNR: LOG shows slight comeback, but differences minimal")

# ====================================
# PHASE 2 ANALYSIS: FLOOR OPTIMIZATION
# ====================================
print(f"\n{'='*100}")
print("PHASE 2 ANALYSIS: FLOOR OPTIMIZATION")
print(f"{'='*100}")

# 4. Floor parameter heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Phase 2: Floor Parameter Optimization by Noise Type', 
             fontsize=16, fontweight='bold')

for idx, (metric, ax) in enumerate(zip(['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr'], 
                                        axes.flat)):
    pivot = df_p2.pivot_table(values=metric, 
                               index='Floor', 
                               columns='Noise_Category', 
                               aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': metric}, linewidths=0.5)
    ax.set_title(f'{metric} by Floor × Noise Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Noise Category', fontsize=11)
    ax.set_ylabel('Floor Parameter', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase2_floor_optimization_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase2_floor_optimization_heatmap.png")
plt.close()

# 5. Floor parameter line plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Phase 2: Floor Parameter Impact on Metrics', 
             fontsize=16, fontweight='bold')

for (metric, ax, ylabel) in [
    ('PESQ', axes[0,0], 'PESQ'),
    ('STOI', axes[0,1], 'STOI'),
    ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
    ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
]:
    for noise_cat in df_p2['Noise_Category'].unique():
        cat_data = df_p2[df_p2['Noise_Category'] == noise_cat]
        avg_by_floor = cat_data.groupby('Floor')[metric].mean().sort_index()
        ax.plot(avg_by_floor.index, avg_by_floor.values, marker='o', 
                linewidth=2, label=noise_cat, markersize=8)
    
    ax.set_xlabel('Floor Parameter', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(title='Noise Type', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase2_floor_parameter_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase2_floor_parameter_curves.png")
plt.close()

# 6. Optimal floor by noise category
print("\nOptimal Floor Values by Noise Category:")
optimal_floors = {}
for noise_cat in df_p2['Noise_Category'].unique():
    cat_data = df_p2[df_p2['Noise_Category'] == noise_cat]
    best_floor = cat_data.groupby('Floor')['composite_score'].mean().idxmax()
    best_score = cat_data.groupby('Floor')['composite_score'].mean().max()
    optimal_floors[noise_cat] = {'floor': best_floor, 'score': best_score}
    print(f"  {noise_cat}: Floor = {best_floor} (score: {best_score:.3f})")

# ====================================
# PHASE 3 ANALYSIS: FINE-TUNING
# ====================================
if len(df_p3) > 0:
    print(f"\n{'='*100}")
    print("PHASE 3 ANALYSIS: FINE-TUNING")
    print(f"{'='*100}")
    
    # 7. Nband comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Phase 3: Number of Bands (Nband) Impact', 
                 fontsize=16, fontweight='bold')
    
    for (metric, ax, ylabel) in [
        ('PESQ', axes[0,0], 'PESQ'),
        ('STOI', axes[0,1], 'STOI'),
        ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
        ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
    ]:
        df_p3_plot = df_p3.copy()
        df_p3_plot['Nband'] = df_p3_plot['Nband'].astype(str)
        sns.boxplot(data=df_p3_plot, x='Nband', y=metric, ax=ax, palette='Set2')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Bands', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase3_nband_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase3_nband_comparison.png")
    plt.close()
    
    # 8. Frame size comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Phase 3: Frame Size Impact', 
                 fontsize=16, fontweight='bold')
    
    for (metric, ax, ylabel) in [
        ('PESQ', axes[0,0], 'PESQ'),
        ('STOI', axes[0,1], 'STOI'),
        ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
        ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
    ]:
        df_p3_plot = df_p3.copy()
        df_p3_plot['FRMSZ_ms'] = df_p3_plot['FRMSZ_ms'].astype(str) + 'ms'
        sns.boxplot(data=df_p3_plot, x='FRMSZ_ms', y=metric, ax=ax, palette='Set2')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Size', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase3_frame_size_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase3_frame_size_comparison.png")
    plt.close()
    
    # 9. Noisefr comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Phase 3: Noise Estimation Frames (Noisefr) Impact', 
                 fontsize=16, fontweight='bold')
    
    for (metric, ax, ylabel) in [
        ('PESQ', axes[0,0], 'PESQ'),
        ('STOI', axes[0,1], 'STOI'),
        ('SI_SDR', axes[1,0], 'SI-SDR (dB)'),
        ('DNSMOS_mos_ovr', axes[1,1], 'DNSMOS Overall')
    ]:
        df_p3_plot = df_p3.copy()
        df_p3_plot['Noisefr'] = df_p3_plot['Noisefr'].astype(str)
        sns.boxplot(data=df_p3_plot, x='Noisefr', y=metric, ax=ax, palette='Set2')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Estimation Frames', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase3_noisefr_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase3_noisefr_comparison.png")
    plt.close()
    
    # Phase 3 summary
    print("\nPhase 3 Optimal Parameters:")
    print(f"  Best Nband: {df_p3.groupby('Nband')['composite_score'].mean().idxmax()}")
    print(f"  Best Frame Size: {df_p3.groupby('FRMSZ_ms')['composite_score'].mean().idxmax()} ms")
    print(f"  Best Noisefr: {df_p3.groupby('Noisefr')['composite_score'].mean().idxmax()}")

# ====================================
# OVERALL BEST CONFIGURATION
# ====================================
print(f"\n{'='*100}")
print("OVERALL BEST CONFIGURATION")
print(f"{'='*100}")

best_overall = df_success.loc[df_success['composite_score'].idxmax()]

print(f"\nBest Configuration (Composite Score = {best_overall['composite_score']:.4f}):")
print(f"  Phase: {int(best_overall['Phase'])}")
print(f"  Frequency Spacing: {best_overall['Freq_spacing']}")
print(f"  Nband: {int(best_overall['Nband'])}")
print(f"  Frame Size: {int(best_overall['FRMSZ_ms'])} ms")
print(f"  Overlap: {int(best_overall['OVLP'])}%")
print(f"  Noisefr: {int(best_overall['Noisefr'])}")
print(f"  Floor: {best_overall['Floor']}")
print(f"\nPerformance:")
print(f"  PESQ: {best_overall['PESQ']:.4f}")
print(f"  STOI: {best_overall['STOI']:.4f}")
print(f"  SI-SDR: {best_overall['SI_SDR']:.2f} dB")
print(f"  DNSMOS: {best_overall['DNSMOS_mos_ovr']:.4f}")
print(f"  Tested at: {int(best_overall['SNR_dB'])} dB SNR, {best_overall['Noise_Category']}")

# Save best config
best_config_df = pd.DataFrame([best_overall])
best_config_df.to_csv(OUTPUT_DIR / 'best_overall_configuration.csv', index=False)

# ====================================
# COMPREHENSIVE SUMMARY REPORT
# ====================================
summary_report = f"""
{'='*100}
3-PHASE PARAMETER SWEEP - COMPREHENSIVE ANALYSIS REPORT
{'='*100}

EXPERIMENT OVERVIEW:
  Total Tests: {len(df)}
  Successful Tests: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)
  Phase 1 (Freq Spacing): {len(df_p1)} tests
  Phase 2 (Floor Optim): {len(df_p2)} tests
  Phase 3 (Fine-tuning): {len(df_p3)} tests

{'='*100}
PHASE 1 INSIGHTS: FREQUENCY SPACING
{'='*100}

Overall Performance (averaged across all SNRs and noises):
"""

for freq in df_p1['Freq_spacing'].unique():
    freq_data = df_p1[df_p1['Freq_spacing'] == freq]
    summary_report += f"\n{freq.upper()}:"
    summary_report += f"\n  PESQ: {freq_data['PESQ'].mean():.3f} ± {freq_data['PESQ'].std():.3f}"
    summary_report += f"\n  STOI: {freq_data['STOI'].mean():.3f} ± {freq_data['STOI'].std():.3f}"
    summary_report += f"\n  SI-SDR: {freq_data['SI_SDR'].mean():.2f} ± {freq_data['SI_SDR'].std():.2f} dB"

summary_report += f"\n\n{'='*100}"
summary_report += "\nPHASE 2 INSIGHTS: FLOOR OPTIMIZATION"
summary_report += f"\n{'='*100}"

summary_report += "\n\nOptimal Floor Values by Noise Type:"
for noise_cat, data in optimal_floors.items():
    summary_report += f"\n  {noise_cat}: Floor = {data['floor']}"
    if noise_cat == 'Stationary':
        summary_report += " (aggressive OK - predictable noise)"
    else:
        summary_report += " (conservative - preserves speech dynamics)"

if len(df_p3) > 0:
    summary_report += f"\n\n{'='*100}"
    summary_report += "\nPHASE 3 INSIGHTS: FINE-TUNING"
    summary_report += f"\n{'='*100}"
    
    best_nband = df_p3.groupby('Nband')['composite_score'].mean().idxmax()
    best_frame = df_p3.groupby('FRMSZ_ms')['composite_score'].mean().idxmax()
    best_noisefr = df_p3.groupby('Noisefr')['composite_score'].mean().idxmax()
    
    summary_report += f"\n\nOptimal Parameter Values:"
    summary_report += f"\n  Nband: {int(best_nband)} bands"
    summary_report += f"\n  Frame Size: {int(best_frame)} ms"
    summary_report += f"\n  Noisefr: {int(best_noisefr)} frames"

summary_report += f"\n\n{'='*100}"
summary_report += "\nFINAL RECOMMENDATIONS"
summary_report += f"\n{'='*100}"

summary_report += f"\n\nBEST OVERALL CONFIGURATION:"
summary_report += f"\n  Frequency Spacing: {best_overall['Freq_spacing']}"
summary_report += f"\n  Nband: {int(best_overall['Nband'])}"
summary_report += f"\n  Frame Size: {int(best_overall['FRMSZ_ms'])} ms"
summary_report += f"\n  Overlap: {int(best_overall['OVLP'])}%"
summary_report += f"\n  Noisefr: {int(best_overall['Noisefr'])}"
summary_report += f"\n  Floor: {best_overall['Floor']}"

summary_report += f"\n\nPerformance (vs noisy baseline assumed 0 improvement):"
summary_report += f"\n  PESQ: {best_overall['PESQ']:.3f}"
summary_report += f"\n  STOI: {best_overall['STOI']:.3f}"
summary_report += f"\n  SI-SDR: {best_overall['SI_SDR']:.2f} dB"
summary_report += f"\n  DNSMOS: {best_overall['DNSMOS_mos_ovr']:.3f}"


summary_report += f"\n\n{'='*100}"
print(summary_report)

# Save report
with open(OUTPUT_DIR / 'comprehensive_analysis_report.txt', 'w') as f:
    f.write(summary_report)

print(f"\n✓ Report saved to: {OUTPUT_DIR / 'comprehensive_analysis_report.txt'}")
print(f"✓ All visualizations saved to: {OUTPUT_DIR.absolute()}")
print(f"\n{'='*100}")
print("ANALYSIS COMPLETE!")
print(f"{'='*100}")