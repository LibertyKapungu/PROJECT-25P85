"""
SNR-Specific Parameter Sweep Analysis with Individual Metrics
==============================================================
Analyzes best configurations at EACH SNR level separately for:
1. Objective vs Subjective comparison
2. Individual metrics: PESQ, STOI, SI-SDR, DNSMOS
3. BEST IMPROVEMENT OVER GTCRN PER NOISE TYPE (Low SNR Focus: -5, 0, 5 dB)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# ====================================
# CONFIGURATION
# ====================================
MODE = "hybrid"

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
CSV_FILE = repo_root / "results" / "EXP3" / "spectral" / "PARAM_SWEEP3" / f"COLLATED_ALL_RESULTS_{MODE}.csv"
GTCRN_DIR = repo_root / "results" / "EXP3" / "GTCRN" / "GTCRN_EXP3p2a"
OUTPUT_DIR = repo_root / "results" / "EXP3" / "spectral" / "PARAM_SWEEP3" / f"snr_noise_{MODE}_all_metrics"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*100)
print("SNR-SPECIFIC ANALYSIS: OBJECTIVE vs SUBJECTIVE + INDIVIDUAL METRICS + GTCRN IMPROVEMENT")
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

# Create objective and subjective scores
df_success['PESQ_norm'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['STOI_norm'] = df_success['STOI']
df_success['SI_SDR_norm'] = (df_success['SI_SDR'] + 10) / 30.0
df_success['SI_SDR_norm'] = df_success['SI_SDR_norm'].clip(0, 1)

df_success['Objective_Score'] = 0.6*df_success['PESQ_norm'] + 0.3*df_success['STOI_norm'] + 0.1*df_success['SI_SDR_norm']
df_success['Subjective_Score'] = df_success['DNSMOS_mos_ovr'] / 5.0
df_success['Balanced_Score'] = 0.5 * df_success['Objective_Score'] + 0.5 * df_success['Subjective_Score']

print(f"✓ Loaded {len(df_success)} successful tests")
print(f"  SNR levels: {sorted(df_success['SNR_dB'].unique())}")
print(f"  Noise categories: {sorted(df_success['Noise_Category'].unique())}")

# ====================================
# LOAD GTCRN BASELINE DATA
# ====================================
print("\nLoading GTCRN baseline data...")

def parse_gtcrn(snr: int) -> pd.DataFrame:
    """Parse GTCRN csv by SNR."""
    file_path = GTCRN_DIR / f"GTCRN_EXP3p2a_merged_[{snr}]dB.csv"
    df = pd.read_csv(file_path)
    df['noise_type'] = df['enhanced_file'].str.extract(r'NOIZEUS_NOISE_DATASET_(.*?)_SNR')
    grouped = df.groupby('noise_type').mean(numeric_only=True)
    return grouped

# Load GTCRN data for low SNRs
gtcrn_data = {}
for snr in [-5, 0, 5]:
    gtcrn_data[snr] = parse_gtcrn(snr)
    print(f"  ✓ Loaded GTCRN data for SNR={snr}dB")

# ====================================
# DEFINE METRICS TO ANALYZE
# ====================================
metrics_to_analyze = {
    'Objective_Score': {'name': 'Objective', 'is_composite': True},
    'Subjective_Score': {'name': 'Subjective', 'is_composite': True},
    'PESQ': {'name': 'PESQ', 'is_composite': False},
    'STOI': {'name': 'STOI', 'is_composite': False},
    'SI_SDR': {'name': 'SI-SDR', 'is_composite': False},
    'DNSMOS_mos_ovr': {'name': 'DNSMOS', 'is_composite': False}
}

params_to_analyze = ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']

# ====================================
# NOISE TYPE IMPROVEMENT ANALYSIS
# ====================================
def analyze_noise_improvement(df_data, gtcrn_baseline):
    """Find best configuration for each noise type over GTCRN at low SNRs"""
    
    print("\n" + "="*100)
    print("BEST IMPROVEMENT OVER GTCRN PER NOISE TYPE (SNR: -5, 0, 5 dB)")
    print("="*100)
    
    noise_dir = OUTPUT_DIR / "noise_type_improvement"
    noise_dir.mkdir(exist_ok=True, parents=True)
    
    low_snr_data = df_data[df_data['SNR_dB'].isin([-5, 0, 5])]
    
    # Extract noise type from enhanced file
    low_snr_data['noise_type'] = low_snr_data['enhanced_file'].str.extract(r'NOIZEUS_NOISE_DATASET_(.*?)_SNR')
    
    metrics = ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']
    
    all_results = []
    
    for snr in [-5, 0, 5]:
        snr_data = low_snr_data[low_snr_data['SNR_dB'] == snr].copy()
        gtcrn_snr = gtcrn_baseline[snr]
        
        print(f"\n{'='*100}")
        print(f"SNR = {snr} dB")
        print(f"{'='*100}")
        
        noise_types = sorted(snr_data['noise_type'].dropna().unique())
        
        for noise in noise_types:
            noise_data = snr_data[snr_data['noise_type'] == noise]
            
            if len(noise_data) == 0 or noise not in gtcrn_snr.index:
                continue
            
            print(f"\n📊 Noise Type: {noise}")
            print("-" * 100)
            
            # Get GTCRN baseline for this noise
            gtcrn_vals = gtcrn_snr.loc[noise]
            
            print(f"  GTCRN Baseline:")
            print(f"    PESQ={gtcrn_vals['PESQ']:.4f} | STOI={gtcrn_vals['STOI']:.4f} | SI-SDR={gtcrn_vals['SI_SDR']:.2f} | DNSMOS={gtcrn_vals['DNSMOS_p808_mos']:.4f}")
            
            # For each metric, find best config
            for metric in metrics:
                metric_col = metric if metric != 'DNSMOS_mos_ovr' else 'DNSMOS_mos_ovr'
                gtcrn_metric = metric if metric != 'DNSMOS_mos_ovr' else 'DNSMOS_p808_mos'
                gtcrn_val = gtcrn_vals[gtcrn_metric]
                
                # Group by config and calculate mean improvement
                config_perf = noise_data.groupby('Config_ID').agg({
                    'Freq_spacing': 'first',
                    'Nband': 'first',
                    'FRMSZ_ms': 'first',
                    'OVLP': 'first',
                    'Floor': 'first',
                    'Noisefr': 'first',
                    metric_col: 'mean',
                    'PESQ': 'mean',
                    'STOI': 'mean',
                    'SI_SDR': 'mean',
                    'DNSMOS_mos_ovr': 'mean'
                }).reset_index()
                
                # Calculate improvement
                config_perf['improvement'] = ((config_perf[metric_col] - gtcrn_val) / gtcrn_val * 100)
                
                # Find best
                best_idx = config_perf['improvement'].idxmax()
                best = config_perf.loc[best_idx]
                
                print(f"\n  {metric}:")
                print(f"    Best Config: {best['Config_ID']}")
                print(f"    {best['Freq_spacing']} | Nband={int(best['Nband'])} | FRMSZ={int(best['FRMSZ_ms'])}ms | OVLP={int(best['OVLP'])}% | Floor={best['Floor']:.4f} | Noisefr={int(best['Noisefr'])}")
                print(f"    {metric}={best[metric_col]:.4f} (GTCRN={gtcrn_val:.4f}, Improvement={best['improvement']:+.2f}%)")
                
                all_results.append({
                    'SNR_dB': snr,
                    'Noise_Type': noise,
                    'Metric': metric,
                    'Best_Config': best['Config_ID'],
                    'Freq_spacing': best['Freq_spacing'],
                    'Nband': int(best['Nband']),
                    'FRMSZ_ms': int(best['FRMSZ_ms']),
                    'OVLP': int(best['OVLP']),
                    'Floor': best['Floor'],
                    'Noisefr': int(best['Noisefr']),
                    'GTCRN_Value': gtcrn_val,
                    'Best_Value': best[metric_col],
                    'Improvement_%': best['improvement'],
                    'PESQ': best['PESQ'],
                    'STOI': best['STOI'],
                    'SI_SDR': best['SI_SDR'],
                    'DNSMOS': best['DNSMOS_mos_ovr']
                })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(noise_dir / 'best_configs_per_noise_vs_gtcrn.csv', index=False)
    print(f"\n✓ Saved: best_configs_per_noise_vs_gtcrn.csv")
    
    # Create summary by noise type
    print("\n" + "="*100)
    print("SUMMARY: TOP IMPROVEMENTS BY NOISE TYPE")
    print("="*100)
    
    for noise in sorted(results_df['Noise_Type'].unique()):
        noise_results = results_df[results_df['Noise_Type'] == noise]
        
        print(f"\n{'='*100}")
        print(f"Noise Type: {noise}")
        print(f"{'='*100}")
        
        for snr in [-5, 0, 5]:
            snr_noise = noise_results[noise_results['SNR_dB'] == snr]
            if len(snr_noise) == 0:
                continue
            
            print(f"\n  SNR={snr}dB:")
            for _, row in snr_noise.iterrows():
                print(f"    {row['Metric']:15s}: {row['Improvement_%']:+7.2f}% | {row['Freq_spacing']:3s} Nband={row['Nband']:2d} FRMSZ={row['FRMSZ_ms']:2d}ms OVLP={row['OVLP']:2d}%")
    
    # Create heatmap of improvements
    for metric in metrics:
        metric_results = results_df[results_df['Metric'] == metric]
        
        pivot_data = metric_results.pivot_table(
            values='Improvement_%',
            index='Noise_Type',
            columns='SNR_dB',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    ax=ax, cbar_kws={'label': 'Improvement over GTCRN (%)'})
        ax.set_title(f'{metric}: Best Improvement over GTCRN by Noise Type and SNR', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Noise Type', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(noise_dir / f'heatmap_improvement_{metric}_vs_gtcrn.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_improvement_{metric}_vs_gtcrn.png")
        plt.close()
    
    return results_df

# Run noise improvement analysis
noise_improvement_results = analyze_noise_improvement(df_success, gtcrn_data)

# ====================================
# FUNCTION TO ANALYZE SINGLE METRIC
# ====================================
def analyze_metric(df_data, metric_col, metric_name, output_subdir):
    """Analyze a single metric across all SNRs and parameters"""
    
    print("\n" + "="*100)
    print(f"ANALYZING METRIC: {metric_name}")
    print("="*100)
    
    # Create subdirectory for this metric
    metric_dir = OUTPUT_DIR / output_subdir
    metric_dir.mkdir(exist_ok=True, parents=True)
    
    # ====================================
    # 1. PARAMETER ANALYSIS (ALL SNRs)
    # ====================================
    print(f"\n{'='*100}")
    print(f"PARAMETER ANALYSIS FOR {metric_name} (AVERAGED ACROSS ALL SNRs)")
    print(f"{'='*100}")
    
    param_results = []
    
    for param in params_to_analyze:
        param_stats = df_data.groupby(param).agg({
            metric_col: ['mean', 'std', 'count']
        }).round(4)
        
        param_stats.columns = ['_'.join(col).strip('_') for col in param_stats.columns.values]
        
        print(f"\n📊 {param} RANKING BY {metric_name}:")
        sorted_stats = param_stats.sort_values(f'{metric_col}_mean', ascending=False)
        print(sorted_stats.to_string())
        
        best_val = sorted_stats.index[0]
        best_score = sorted_stats[f'{metric_col}_mean'].iloc[0]
        
        print(f"\n🎯 BEST {param}: {best_val} (Score: {best_score:.4f})")
        
        param_results.append({
            'Parameter': param,
            'Best_Value': best_val,
            'Best_Score': best_score
        })
    
    # Save parameter results
    param_df = pd.DataFrame(param_results)
    param_df.to_csv(metric_dir / f'parameter_analysis_{metric_name}_all_snrs.csv', index=False)
    print(f"\n✓ Saved: parameter_analysis_{metric_name}_all_snrs.csv")
    
    # ====================================
    # 2. BEST OVERALL CONFIGURATION (ALL SNRs)
    # ====================================
    print(f"\n{'='*100}")
    print(f"BEST OVERALL CONFIGURATION FOR {metric_name} (ACROSS ALL SNRs)")
    print(f"{'='*100}")
    
    config_avg = df_data.groupby('Config_ID').agg({
        'Freq_spacing': 'first',
        'Nband': 'first',
        'FRMSZ_ms': 'first',
        'OVLP': 'first',
        'Floor': 'first',
        'Noisefr': 'first',
        metric_col: 'mean',
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean',
        'DNSMOS_mos_ovr': 'mean',
    }).round(4)
    
    best_config_id = config_avg[metric_col].idxmax()
    best_config = config_avg.loc[best_config_id]
    
    print(f"\n🏆 BEST CONFIGURATION FOR {metric_name}:")
    print(f"  Config: {best_config_id}")
    print(f"  {best_config['Freq_spacing']} | Nband={int(best_config['Nband'])} | FRMSZ={int(best_config['FRMSZ_ms'])}ms | OVLP={int(best_config['OVLP'])}% | Floor={best_config['Floor']:.4f} | Noisefr={int(best_config['Noisefr'])}")
    print(f"\n  {metric_name}: {best_config[metric_col]:.4f}")
    print(f"  PESQ: {best_config['PESQ']:.4f} | STOI: {best_config['STOI']:.4f} | SI-SDR: {best_config['SI_SDR']:.2f} dB | DNSMOS: {best_config['DNSMOS_mos_ovr']:.4f}")
    
    # Top 10 configs
    print(f"\n📊 TOP 10 CONFIGURATIONS BY {metric_name}:")
    top_configs = config_avg.nlargest(10, metric_col)[
        ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', metric_col, 'PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']
    ]
    print(top_configs.to_string())
    
    config_avg.to_csv(metric_dir / f'config_averaged_{metric_name}.csv')
    print(f"\n✓ Saved: config_averaged_{metric_name}.csv")
    
    # ====================================
    # 3. SNR-SPECIFIC ANALYSIS
    # ====================================
    print(f"\n{'='*100}")
    print(f"SNR-SPECIFIC ANALYSIS FOR {metric_name}")
    print(f"{'='*100}")
    
    snr_param_results = []
    snr_best_configs = []
    
    for snr in sorted(df_data['SNR_dB'].unique()):
        snr_data = df_data[df_data['SNR_dB'] == snr]
        
        print(f"\n{'='*100}")
        print(f"SNR = {snr} dB | {metric_name}")
        print(f"{'='*100}")
        
        # Analyze each parameter at this SNR
        for param in params_to_analyze:
            param_avg = snr_data.groupby(param)[metric_col].mean()
            best_val = param_avg.idxmax()
            best_score = param_avg[best_val]
            
            snr_param_results.append({
                'SNR_dB': snr,
                'Parameter': param,
                'Best_Value': best_val,
                'Score': best_score
            })
        
        # Best config at this SNR
        snr_config_avg = snr_data.groupby('Config_ID').agg({
            'Freq_spacing': 'first',
            'Nband': 'first',
            'FRMSZ_ms': 'first',
            'OVLP': 'first',
            'Floor': 'first',
            'Noisefr': 'first',
            metric_col: 'mean',
            'PESQ': 'mean',
            'STOI': 'mean',
            'SI_SDR': 'mean',
            'DNSMOS_mos_ovr': 'mean',
        }).reset_index()
        
        best_idx = snr_config_avg[metric_col].idxmax()
        snr_best = snr_config_avg.loc[best_idx]
        
        print(f"\n🏆 Best at {snr} dB:")
        print(f"  {snr_best['Freq_spacing']} | Nband={int(snr_best['Nband'])} | FRMSZ={int(snr_best['FRMSZ_ms'])}ms | OVLP={int(snr_best['OVLP'])}% | Floor={snr_best['Floor']:.4f}")
        print(f"  {metric_name}={snr_best[metric_col]:.4f} | PESQ={snr_best['PESQ']:.4f} | STOI={snr_best['STOI']:.4f} | SI-SDR={snr_best['SI_SDR']:.2f}")
        
        snr_best_configs.append({
            'SNR_dB': snr,
            'Best_Config': snr_best['Config_ID'],
            'Freq_spacing': snr_best['Freq_spacing'],
            'Nband': int(snr_best['Nband']),
            'FRMSZ_ms': int(snr_best['FRMSZ_ms']),
            'OVLP': int(snr_best['OVLP']),
            'Floor': snr_best['Floor'],
            'Noisefr': int(snr_best['Noisefr']),
            f'{metric_name}_Score': snr_best[metric_col],
            'PESQ': snr_best['PESQ'],
            'STOI': snr_best['STOI'],
            'SI_SDR': snr_best['SI_SDR'],
            'DNSMOS': snr_best['DNSMOS_mos_ovr']
        })
    
    # Save SNR-specific results
    snr_param_df = pd.DataFrame(snr_param_results)
    snr_param_df.to_csv(metric_dir / f'snr_param_analysis_{metric_name}.csv', index=False)
    print(f"\n✓ Saved: snr_param_analysis_{metric_name}.csv")
    
    snr_configs_df = pd.DataFrame(snr_best_configs)
    snr_configs_df.to_csv(metric_dir / f'snr_best_configs_{metric_name}.csv', index=False)
    print(f"\n✓ Saved: snr_best_configs_{metric_name}.csv")
    
    # ====================================
    # 4. CONSISTENCY ANALYSIS
    # ====================================
    print(f"\n{'='*100}")
    print(f"PARAMETER CONSISTENCY ACROSS SNR LEVELS FOR {metric_name}")
    print(f"{'='*100}")
    
    consistency_results = []
    
    for param in params_to_analyze:
        param_data = snr_param_df[snr_param_df['Parameter'] == param]
        winners = param_data['Best_Value'].value_counts()
        most_common = winners.idxmax()
        consistency_pct = winners.max() / len(param_data) * 100
        
        print(f"\n{param}:")
        print(f"  '{most_common}' wins {consistency_pct:.0f}% of SNR levels")
        print(f"  Distribution: {dict(winners)}")
        
        consistency_results.append({
            'Parameter': param,
            'Most_Common_Value': most_common,
            'Consistency_%': consistency_pct
        })
    
    consistency_df = pd.DataFrame(consistency_results)
    consistency_df.to_csv(metric_dir / f'parameter_consistency_{metric_name}.csv', index=False)
    print(f"\n✓ Saved: parameter_consistency_{metric_name}.csv")
    
    # ====================================
    # 5. VISUALIZATIONS
    # ====================================
    print(f"\n{'='*100}")
    print(f"GENERATING VISUALIZATIONS FOR {metric_name}")
    print(f"{'='*100}")
    
    # Plot each parameter across SNRs
    for param in ['Freq_spacing', 'Nband', 'OVLP']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        param_values = sorted(df_data[param].unique())
        
        for val in param_values:
            subset = df_data[df_data[param] == val]
            snr_avg = subset.groupby('SNR_dB')[metric_col].mean()
            ax.plot(snr_avg.index, snr_avg.values, 'o-', label=str(val), linewidth=2, markersize=8)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax.set_title(f'{metric_name} by {param} Across SNR', fontsize=14, fontweight='bold')
        ax.legend(title=param, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(metric_dir / f'{metric_name}_by_{param}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {metric_name}_by_{param}.png")
        plt.close()
    
    # Heatmap: Freq_spacing vs SNR
    freq_values = sorted(df_data['Freq_spacing'].unique())
    snr_levels = sorted(df_data['SNR_dB'].unique())
    
    metric_matrix = []
    for snr in snr_levels:
        row = []
        for freq in freq_values:
            subset = df_data[(df_data['SNR_dB'] == snr) & (df_data['Freq_spacing'] == freq)]
            if len(subset) > 0:
                row.append(subset[metric_col].mean())
            else:
                row.append(np.nan)
        metric_matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fmt = '.2f' if metric_col == 'SI_SDR' else '.3f'
    sns.heatmap(np.array(metric_matrix), annot=True, fmt=fmt, cmap='RdYlGn', ax=ax,
                xticklabels=freq_values, yticklabels=[f'{s}dB' for s in snr_levels],
                cbar_kws={'label': f'{metric_name} Score'})
    ax.set_title(f'{metric_name} by Freq_spacing and SNR', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Spacing', fontsize=12)
    ax.set_ylabel('SNR Level', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(metric_dir / f'heatmap_{metric_name}_freq_spacing.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: heatmap_{metric_name}_freq_spacing.png")
    plt.close()

# ====================================
# RUN ANALYSIS FOR ALL METRICS
# ====================================
for metric_col, metric_info in metrics_to_analyze.items():
    metric_name = metric_info['name']
    subdir = f"metric_{metric_name.lower().replace('-', '_').replace(' ', '_')}"
    analyze_metric(df_success, metric_col, metric_name, subdir)

# ====================================
# OBJECTIVE VS SUBJECTIVE COMPARISON
# ====================================
print("\n" + "="*100)
print("OBJECTIVE vs SUBJECTIVE COMPARISON")
print("="*100)

comparison_dir = OUTPUT_DIR / "objective_vs_subjective"
comparison_dir.mkdir(exist_ok=True, parents=True)

# Scatter plot
config_avg = df_success.groupby('Config_ID').agg({
    'Objective_Score': 'mean',
    'Subjective_Score': 'mean',
    'Balanced_Score': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(config_avg['Objective_Score'], config_avg['Subjective_Score'],
                    c=config_avg['Balanced_Score'], cmap='viridis', s=100, alpha=0.6)
ax.set_xlabel('Objective Score', fontsize=12)
ax.set_ylabel('Subjective Score', fontsize=12)
ax.set_title('Objective vs Subjective Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Agreement')

plt.colorbar(scatter, label='Balanced Score')
ax.legend()
plt.tight_layout()
plt.savefig(comparison_dir / 'objective_vs_subjective_scatter.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: objective_vs_subjective_scatter.png")
plt.close()

print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR.absolute()}")
print("="*100)