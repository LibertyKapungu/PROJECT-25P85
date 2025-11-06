"""
SNR-Specific Parameter Sweep Analysis with Objective vs Subjective Metrics
===========================================================================
Analyzes best configurations at EACH SNR level separately with detailed
comparison between objective (PESQ, STOI, SI-SDR) and subjective (DNSMOS) metrics
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
OUTPUT_DIR = repo_root / "results" / "EXP3" / "spectral" / "PARAM_SWEEP3" / f"snr_specific_analysis_{MODE}_subj_vs_obj_weighted_pesq"
# CSV_FILE = repo_root / "results" / "EXP3" / "spectral" / "PARAM_SWEEP2" / MODE / f"COLLATED_ALL_RESULTS_{MODE}.csv"
# OUTPUT_DIR = repo_root / "results" / "EXP3" / "spectral" / "PARAM_SWEEP2" / f"snr_specific_analysis_{MODE}_subj_vs_obj"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


print("="*100)
print("SNR-SPECIFIC OBJECTIVE vs SUBJECTIVE ANALYSIS")
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

# # Create objective and subjective scores
df_success['PESQ_norm'] = (df_success['PESQ'] + 0.5) / 5.0
df_success['STOI_norm'] = df_success['STOI']
df_success['SI_SDR_norm'] = (df_success['SI_SDR'] + 10) / 30.0
df_success['SI_SDR_norm'] = df_success['SI_SDR_norm'].clip(0, 1)

# # Objective score: Average of normalized PESQ, STOI, SI-SDR
# df_success['Objective_Score'] = (df_success['PESQ_norm'] + df_success['STOI_norm'] + df_success['SI_SDR_norm']) / 3

# CHANGE: Highlights PESQ (clarity) while still considering intelligibility (STOI) and distortion (SI-SDR). Avoids overfitting to PESQ alone, which could lead to unnatural artifacts.
# Gives researchers a clear optimization target that aligns with hearing aid goals: clarity first, intelligibility second, distortion control third

df_success['Objective_Score'] = 0.6*df_success['PESQ_norm'] + 0.3*df_success['STOI_norm'] + 0.1*df_success['SI_SDR_norm']

# Subjective score: DNSMOS normalized
df_success['Subjective_Score'] = df_success['DNSMOS_mos_ovr'] / 5.0

# # Weighted scores
df_success['Weighted_Score'] = 0.5 * df_success['PESQ_norm'] + 0.5 * df_success['STOI']
df_success['Balanced_Score'] = 0.5 * df_success['Objective_Score'] + 0.5 * df_success['Subjective_Score']

print(f"✓ Loaded {len(df_success)} successful tests")
print(f"  SNR levels: {sorted(df_success['SNR_dB'].unique())}")
print(f"  Noise categories: {sorted(df_success['Noise_Category'].unique())}")

# ====================================
# PARAMETER ANALYSIS: OBJECTIVE vs SUBJECTIVE (ALL SNRs)
# ====================================
print("\n" + "="*100)
print("PARAMETER ANALYSIS: OBJECTIVE vs SUBJECTIVE (AVERAGED ACROSS ALL SNRs)")
print("="*100)

params_to_analyze = ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Noisefr']
param_obj_subj_results = []

for param in params_to_analyze:
    print(f"\n{'='*100}")
    print(f"PARAMETER: {param}")
    print(f"{'='*100}")
    
    param_stats = df_success.groupby(param).agg({
        'Objective_Score': ['mean', 'std', 'count'],
        'Subjective_Score': ['mean', 'std'],
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean',
        'DNSMOS_mos_ovr': 'mean'
    }).round(4)
    
    # Flatten columns
    param_stats.columns = ['_'.join(col).strip('_') for col in param_stats.columns.values]
    
    print("\n📊 OBJECTIVE SCORE RANKING:")
    obj_sorted = param_stats.sort_values('Objective_Score_mean', ascending=False)
    print(obj_sorted[['Objective_Score_mean', 'Objective_Score_std', 'PESQ_mean', 'STOI_mean', 'SI_SDR_mean']].to_string())
    
    print("\n📊 SUBJECTIVE SCORE RANKING:")
    subj_sorted = param_stats.sort_values('Subjective_Score_mean', ascending=False)
    print(subj_sorted[['Subjective_Score_mean', 'Subjective_Score_std', 'DNSMOS_mos_ovr_mean']].to_string())
    
    # Find best for each
    best_obj_val = obj_sorted['Objective_Score_mean'].idxmax()
    best_obj_score = obj_sorted['Objective_Score_mean'].max()
    
    best_subj_val = subj_sorted['Subjective_Score_mean'].idxmax()
    best_subj_score = subj_sorted['Subjective_Score_mean'].max()
    
    agrees = (best_obj_val == best_subj_val)
    
    print(f"\n🎯 BEST {param}:")
    print(f"  Objective: {best_obj_val} (Score: {best_obj_score:.4f})")
    print(f"  Subjective: {best_subj_val} (Score: {best_subj_score:.4f})")
    if agrees:
        print(f"  ✅ AGREEMENT: Same value optimal for both!")
    else:
        print(f"  ⚠️  DISAGREEMENT: Different optimal values")
    
    param_obj_subj_results.append({
        'Parameter': param,
        'Best_Objective': best_obj_val,
        'Objective_Score': best_obj_score,
        'Best_Subjective': best_subj_val,
        'Subjective_Score': best_subj_score,
        'Agrees': agrees
    })

# Save parameter comparison
param_comparison_df = pd.DataFrame(param_obj_subj_results)
param_comparison_df.to_csv(OUTPUT_DIR / 'parameter_objective_vs_subjective_all_snrs.csv', index=False)
print(f"\n✓ Saved: parameter_objective_vs_subjective_all_snrs.csv")

# ====================================
# BEST OVERALL CONFIGURATIONS (ALL SNRs)
# ====================================
print("\n" + "="*100)
print("BEST OVERALL CONFIGURATIONS (ACROSS ALL SNRs)")
print("="*100)

# Average by configuration across all SNRs
config_avg = df_success.groupby('Config_ID').agg({
    'Freq_spacing': 'first',
    'Nband': 'first',
    'FRMSZ_ms': 'first',
    'OVLP': 'first',
    'Floor': 'first',
    'Noisefr': 'first',
    'Objective_Score': 'mean',
    'Subjective_Score': 'mean',
    'Balanced_Score': 'mean',
    'PESQ': 'mean',
    'STOI': 'mean',
    'SI_SDR': 'mean',
    'DNSMOS_mos_ovr': 'mean',
}).round(4)

# Best Objective
best_obj_config_id = config_avg['Objective_Score'].idxmax()
best_obj = config_avg.loc[best_obj_config_id]

print(f"\n🏆 BEST OBJECTIVE CONFIGURATION:")
print(f"  Config: {best_obj_config_id}")
print(f"  {best_obj['Freq_spacing']} | Nband={int(best_obj['Nband'])} | FRMSZ={int(best_obj['FRMSZ_ms'])}ms | OVLP={int(best_obj['OVLP'])}% | Floor={best_obj['Floor']:.4f} | Noisefr={int(best_obj['Noisefr'])}")
print(f"\n  Objective Score: {best_obj['Objective_Score']:.4f}")
print(f"  PESQ: {best_obj['PESQ']:.4f} | STOI: {best_obj['STOI']:.4f} | SI-SDR: {best_obj['SI_SDR']:.2f} dB")
print(f"  Subjective Score: {best_obj['Subjective_Score']:.4f} | DNSMOS: {best_obj['DNSMOS_mos_ovr']:.4f}")

# Best Subjective
best_subj_config_id = config_avg['Subjective_Score'].idxmax()
best_subj = config_avg.loc[best_subj_config_id]

print(f"\n🏆 BEST SUBJECTIVE CONFIGURATION:")
print(f"  Config: {best_subj_config_id}")
print(f"  {best_subj['Freq_spacing']} | Nband={int(best_subj['Nband'])} | FRMSZ={int(best_subj['FRMSZ_ms'])}ms | OVLP={int(best_subj['OVLP'])}% | Floor={best_subj['Floor']:.4f} | Noisefr={int(best_subj['Noisefr'])}")
print(f"\n  Subjective Score: {best_subj['Subjective_Score']:.4f} | DNSMOS: {best_subj['DNSMOS_mos_ovr']:.4f}")
print(f"  Objective Score: {best_subj['Objective_Score']:.4f}")
print(f"  PESQ: {best_subj['PESQ']:.4f} | STOI: {best_subj['STOI']:.4f} | SI-SDR: {best_subj['SI_SDR']:.2f} dB")

# Best Balanced
best_balanced_config_id = config_avg['Balanced_Score'].idxmax()
best_balanced = config_avg.loc[best_balanced_config_id]

print(f"\n🏆 BEST BALANCED CONFIGURATION (50% Obj + 50% Subj):")
print(f"  Config: {best_balanced_config_id}")
print(f"  {best_balanced['Freq_spacing']} | Nband={int(best_balanced['Nband'])} | FRMSZ={int(best_balanced['FRMSZ_ms'])}ms | OVLP={int(best_balanced['OVLP'])}% | Floor={best_balanced['Floor']:.4f} | Noisefr={int(best_balanced['Noisefr'])}")
print(f"\n  Balanced Score: {best_balanced['Balanced_Score']:.4f}")
print(f"  Objective: {best_balanced['Objective_Score']:.4f} | Subjective: {best_balanced['Subjective_Score']:.4f}")
print(f"  PESQ: {best_balanced['PESQ']:.4f} | STOI: {best_balanced['STOI']:.4f} | SI-SDR: {best_balanced['SI_SDR']:.2f} dB | DNSMOS: {best_balanced['DNSMOS_mos_ovr']:.4f}")

if best_obj_config_id == best_subj_config_id:
    print(f"\n✅ EXCELLENT: Same configuration excels at both objective AND subjective metrics!")

# Top 10 configs for each
print(f"\n📊 TOP 10 CONFIGURATIONS BY OBJECTIVE SCORE:")
top_obj = config_avg.nlargest(10, 'Objective_Score')[
    ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Objective_Score', 'PESQ', 'STOI', 'SI_SDR']
]
print(top_obj.to_string())

print(f"\n📊 TOP 10 CONFIGURATIONS BY SUBJECTIVE SCORE:")
top_subj = config_avg.nlargest(10, 'Subjective_Score')[
    ['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP', 'Floor', 'Subjective_Score', 'DNSMOS_mos_ovr']
]
print(top_subj.to_string())

# Save averaged configs
config_avg.to_csv(OUTPUT_DIR / 'config_averaged_objective_vs_subjective.csv')
print(f"\n✓ Saved: config_averaged_objective_vs_subjective.csv")

# ====================================
# SNR-SPECIFIC ANALYSIS
# ====================================
print("\n" + "="*100)
print("SNR-SPECIFIC OBJECTIVE vs SUBJECTIVE ANALYSIS")
print("="*100)

snr_results = []
snr_best_configs = []

for snr in sorted(df_success['SNR_dB'].unique()):
    snr_data = df_success[df_success['SNR_dB'] == snr]
    
    print(f"\n{'='*100}")
    print(f"SNR = {snr} dB")
    print(f"{'='*100}")
    
    # Analyze each parameter at this SNR
    print(f"\n📊 PARAMETER ANALYSIS AT {snr} dB:")
    for param in params_to_analyze:
        param_obj = snr_data.groupby(param)['Objective_Score'].mean()
        param_subj = snr_data.groupby(param)['Subjective_Score'].mean()
        
        best_obj_val = param_obj.idxmax()
        best_subj_val = param_subj.idxmax()
        
        print(f"\n  {param}:")
        print(f"    Best Objective: {best_obj_val} (Score: {param_obj[best_obj_val]:.4f})")
        print(f"    Best Subjective: {best_subj_val} (Score: {param_subj[best_subj_val]:.4f})")
        if best_obj_val == best_subj_val:
            print(f"     Agreement")
        else:
            print(f"      Disagreement")
        
        snr_results.append({
            'SNR_dB': snr,
            'Parameter': param,
            'Best_Objective': best_obj_val,
            'Obj_Score': param_obj[best_obj_val],
            'Best_Subjective': best_subj_val,
            'Subj_Score': param_subj[best_subj_val],
            'Agrees': best_obj_val == best_subj_val
        })
    
    # Best overall configs at this SNR
    print(f"\n{'='*100}")
    print(f"BEST CONFIGS AT {snr} dB:")
    print(f"{'='*100}")
    
    # Average by config at this SNR
    snr_config_avg = snr_data.groupby('Config_ID').agg({
        'Freq_spacing': 'first',
        'Nband': 'first',
        'FRMSZ_ms': 'first',
        'OVLP': 'first',
        'Floor': 'first',
        'Noisefr': 'first',
        'Objective_Score': 'mean',
        'Subjective_Score': 'mean',
        'Balanced_Score': 'mean',
        'PESQ': 'mean',
        'STOI': 'mean',
        'SI_SDR': 'mean',
        'DNSMOS_mos_ovr': 'mean',
    }).reset_index()
    
    # Best Objective at this SNR
    best_obj_idx = snr_config_avg['Objective_Score'].idxmax()
    snr_best_obj = snr_config_avg.loc[best_obj_idx]
    
    print(f"\n Best OBJECTIVE:")
    print(f"  {snr_best_obj['Freq_spacing']} | Nband={int(snr_best_obj['Nband'])} | FRMSZ={int(snr_best_obj['FRMSZ_ms'])}ms | OVLP={int(snr_best_obj['OVLP'])}% | Floor={snr_best_obj['Floor']:.4f}")
    print(f"  Obj={snr_best_obj['Objective_Score']:.4f} | PESQ={snr_best_obj['PESQ']:.4f} | STOI={snr_best_obj['STOI']:.4f} | SI-SDR={snr_best_obj['SI_SDR']:.2f}")
    print(f"  Subj={snr_best_obj['Subjective_Score']:.4f} | DNSMOS={snr_best_obj['DNSMOS_mos_ovr']:.4f}")
    
    # Best Subjective at this SNR
    best_subj_idx = snr_config_avg['Subjective_Score'].idxmax()
    snr_best_subj = snr_config_avg.loc[best_subj_idx]
    
    print(f"\n Best SUBJECTIVE:")
    print(f"  {snr_best_subj['Freq_spacing']} | Nband={int(snr_best_subj['Nband'])} | FRMSZ={int(snr_best_subj['FRMSZ_ms'])}ms | OVLP={int(snr_best_subj['OVLP'])}% | Floor={snr_best_subj['Floor']:.4f}")
    print(f"  Subj={snr_best_subj['Subjective_Score']:.4f} | DNSMOS={snr_best_subj['DNSMOS_mos_ovr']:.4f}")
    print(f"  Obj={snr_best_subj['Objective_Score']:.4f} | PESQ={snr_best_subj['PESQ']:.4f} | STOI={snr_best_subj['STOI']:.4f} | SI-SDR={snr_best_subj['SI_SDR']:.2f}")
    
    # Best Balanced at this SNR
    best_balanced_idx = snr_config_avg['Balanced_Score'].idxmax()
    snr_best_balanced = snr_config_avg.loc[best_balanced_idx]
    
    print(f"\n Best BALANCED:")
    print(f"  {snr_best_balanced['Freq_spacing']} | Nband={int(snr_best_balanced['Nband'])} | FRMSZ={int(snr_best_balanced['FRMSZ_ms'])}ms | OVLP={int(snr_best_balanced['OVLP'])}% | Floor={snr_best_balanced['Floor']:.4f}")
    print(f"  Balanced={snr_best_balanced['Balanced_Score']:.4f} | Obj={snr_best_balanced['Objective_Score']:.4f} | Subj={snr_best_balanced['Subjective_Score']:.4f}")
    
    if snr_best_obj['Config_ID'] == snr_best_subj['Config_ID']:
        print(f"\n Same configuration optimal for both!")
    
    # Store for comparison
    snr_best_configs.append({
        'SNR_dB': snr,
        'Best_Obj_Config': snr_best_obj['Config_ID'],
        'Best_Obj_Freq': snr_best_obj['Freq_spacing'],
        'Best_Obj_Nband': int(snr_best_obj['Nband']),
        'Best_Obj_FRMSZ': int(snr_best_obj['FRMSZ_ms']),
        'Best_Obj_OVLP': int(snr_best_obj['OVLP']),
        'Best_Obj_Floor': snr_best_obj['Floor'],
        'Best_Obj_Score': snr_best_obj['Objective_Score'],
        'Best_Obj_PESQ': snr_best_obj['PESQ'],
        'Best_Obj_STOI': snr_best_obj['STOI'],
        'Best_Obj_SI_SDR': snr_best_obj['SI_SDR'],
        'Best_Subj_Config': snr_best_subj['Config_ID'],
        'Best_Subj_Freq': snr_best_subj['Freq_spacing'],
        'Best_Subj_Nband': int(snr_best_subj['Nband']),
        'Best_Subj_FRMSZ': int(snr_best_subj['FRMSZ_ms']),
        'Best_Subj_OVLP': int(snr_best_subj['OVLP']),
        'Best_Subj_Floor': snr_best_subj['Floor'],
        'Best_Subj_Score': snr_best_subj['Subjective_Score'],
        'Best_Subj_DNSMOS': snr_best_subj['DNSMOS_mos_ovr'],
        'Same_Config': snr_best_obj['Config_ID'] == snr_best_subj['Config_ID']
    })

# Save SNR-specific results
snr_results_df = pd.DataFrame(snr_results)
snr_results_df.to_csv(OUTPUT_DIR / 'snr_param_objective_vs_subjective.csv', index=False)
print(f"\n✓ Saved: snr_param_objective_vs_subjective.csv")

snr_best_configs_df = pd.DataFrame(snr_best_configs)
snr_best_configs_df.to_csv(OUTPUT_DIR / 'snr_best_configs_objective_vs_subjective.csv', index=False)
print(f"\n✓ Saved: snr_best_configs_objective_vs_subjective.csv")

# ====================================
# CONSISTENCY ANALYSIS
# ====================================
print("\n" + "="*100)
print("PARAMETER CONSISTENCY ACROSS SNR LEVELS")
print("="*100)

consistency_analysis = []

for param in params_to_analyze:
    param_data = snr_results_df[snr_results_df['Parameter'] == param]
    
    # Objective winners
    obj_winners = param_data['Best_Objective'].value_counts()
    most_common_obj = obj_winners.idxmax()
    obj_consistency = obj_winners.max() / len(param_data) * 100
    
    # Subjective winners
    subj_winners = param_data['Best_Subjective'].value_counts()
    most_common_subj = subj_winners.idxmax()
    subj_consistency = subj_winners.max() / len(param_data) * 100
    
    # Agreement percentage
    agreement_pct = (param_data['Agrees'].sum() / len(param_data)) * 100
    
    print(f"\n{param}:")
    print(f"  Objective: '{most_common_obj}' wins {obj_consistency:.0f}% of SNR levels")
    print(f"             Distribution: {dict(obj_winners)}")
    print(f"  Subjective: '{most_common_subj}' wins {subj_consistency:.0f}% of SNR levels")
    print(f"              Distribution: {dict(subj_winners)}")
    print(f"  Agreement: {agreement_pct:.0f}% of SNR levels have same optimal value")
    
    consistency_analysis.append({
        'Parameter': param,
        'Obj_Most_Common': most_common_obj,
        'Obj_Consistency_%': obj_consistency,
        'Subj_Most_Common': most_common_subj,
        'Subj_Consistency_%': subj_consistency,
        'Agreement_%': agreement_pct
    })

consistency_df = pd.DataFrame(consistency_analysis)
consistency_df.to_csv(OUTPUT_DIR / 'parameter_consistency_obj_vs_subj.csv', index=False)
print(f"\n✓ Saved: parameter_consistency_obj_vs_subj.csv")

# ====================================
# VISUALIZATIONS
# ====================================
print("\n" + "="*100)
print("GENERATING OBJECTIVE vs SUBJECTIVE VISUALIZATIONS")
print("="*100)

# 1. Objective vs Subjective scatter for all configs
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(config_avg['Objective_Score'], config_avg['Subjective_Score'],
                    c=config_avg['Balanced_Score'], cmap='viridis', s=100, alpha=0.6)
ax.set_xlabel('Objective Score (PESQ + STOI + SI-SDR)', fontsize=12)
ax.set_ylabel('Subjective Score (DNSMOS)', fontsize=12)
ax.set_title('Configuration Performance: Objective vs Subjective', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add diagonal line
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Agreement')

# Annotate best configs
ax.scatter(best_obj['Objective_Score'], best_obj['Subjective_Score'], 
          color='red', s=300, marker='*', label='Best Objective', zorder=5)
ax.scatter(best_subj['Objective_Score'], best_subj['Subjective_Score'], 
          color='blue', s=300, marker='*', label='Best Subjective', zorder=5)

plt.colorbar(scatter, label='Balanced Score')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'objective_vs_subjective_scatter.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: objective_vs_subjective_scatter.png")
plt.close()

# 2. Parameter comparison: Objective vs Subjective across SNRs
for param in ['Freq_spacing', 'Nband', 'OVLP']:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{param}: Objective vs Subjective Across SNR', fontsize=14, fontweight='bold')
    
    param_values = sorted(df_success[param].unique())
    snr_levels = sorted(df_success['SNR_dB'].unique())
    
    # Objective scores
    for val in param_values:
        subset = df_success[df_success[param] == val]
        snr_avg = subset.groupby('SNR_dB')['Objective_Score'].mean()
        axes[0].plot(snr_avg.index, snr_avg.values, 'o-', label=str(val), linewidth=2, markersize=8)
    
    axes[0].set_xlabel('SNR (dB)', fontsize=11)
    axes[0].set_ylabel('Objective Score', fontsize=11)
    axes[0].set_title('Objective (PESQ+STOI+SI-SDR)', fontsize=12)
    axes[0].legend(title=param, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Subjective scores
    for val in param_values:
        subset = df_success[df_success[param] == val]
        snr_avg = subset.groupby('SNR_dB')['Subjective_Score'].mean()
        axes[1].plot(snr_avg.index, snr_avg.values, 'o-', label=str(val), linewidth=2, markersize=8)
    
    axes[1].set_xlabel('SNR (dB)', fontsize=11)
    axes[1].set_ylabel('Subjective Score', fontsize=11)
    axes[1].set_title('Subjective (DNSMOS)', fontsize=12)
    axes[1].legend(title=param, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'obj_vs_subj_by_{param}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: obj_vs_subj_by_{param}.png")
    plt.close()

# 3. Heatmap comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Objective vs Subjective Scores by Freq_spacing and SNR', fontsize=14, fontweight='bold')

freq_values = sorted(df_success['Freq_spacing'].unique())
snr_levels = sorted(df_success['SNR_dB'].unique())

# Objective heatmap
obj_matrix = []
for snr in snr_levels:
    row = []
    for freq in freq_values:
        subset = df_success[(df_success['SNR_dB'] == snr) & (df_success['Freq_spacing'] == freq)]
        if len(subset) > 0:
            row.append(subset['Objective_Score'].mean())
        else:
            row.append(np.nan)
    obj_matrix.append(row)

sns.heatmap(np.array(obj_matrix), annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0],
            xticklabels=freq_values, yticklabels=[f'{s}dB' for s in snr_levels],
            cbar_kws={'label': 'Objective Score'})
axes[0].set_title('Objective Score', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frequency Spacing', fontsize=10)
axes[0].set_ylabel('SNR Level', fontsize=10)

# Subjective heatmap
subj_matrix = []
for snr in snr_levels:
    row = []
    for freq in freq_values:
        subset = df_success[(df_success['SNR_dB'] == snr) & (df_success['Freq_spacing'] == freq)]
        if len(subset) > 0:
            row.append(subset['Subjective_Score'].mean())
        else:
            row.append(np.nan)
    subj_matrix.append(row)

sns.heatmap(np.array(subj_matrix), annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
            xticklabels=freq_values, yticklabels=[f'{s}dB' for s in snr_levels],
            cbar_kws={'label': 'Subjective Score'})
axes[1].set_title('Subjective Score', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frequency Spacing', fontsize=10)
axes[1].set_ylabel('SNR Level', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_obj_vs_subj_freq_spacing.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_obj_vs_subj_freq_spacing.png")
plt.close()

# ====================================
# SUMMARY RECOMMENDATIONS
# ====================================
print("\n" + "="*100)
print("🎯 SUMMARY RECOMMENDATIONS")
print("="*100)

print(f"\n1. PARAMETER-LEVEL RECOMMENDATIONS (All SNRs):")
print(f"{'='*100}")
for result in param_obj_subj_results:
    if result['Agrees']:
        print(f"  ✅ {result['Parameter']}: {result['Best_Objective']} (AGREES for both)")
    else:
        print(f"  ⚠️  {result['Parameter']}: Objective={result['Best_Objective']}, Subjective={result['Best_Subjective']}")

print(f"\n2. OVERALL BEST CONFIGURATIONS (Averaged across all SNRs):")
print(f"{'='*100}")
print(f"\n  🏆 For OBJECTIVE metrics (PESQ, STOI, SI-SDR):")
print(f"     Config: {best_obj_config_id}")
print(f"     {best_obj['Freq_spacing']} | Nband={int(best_obj['Nband'])} | FRMSZ={int(best_obj['FRMSZ_ms'])}ms | OVLP={int(best_obj['OVLP'])}% | Floor={best_obj['Floor']:.4f}")
print(f"     Score: {best_obj['Objective_Score']:.4f}")

print(f"\n  🏆 For SUBJECTIVE metrics (DNSMOS):")
print(f"     Config: {best_subj_config_id}")
print(f"     {best_subj['Freq_spacing']} | Nband={int(best_subj['Nband'])} | FRMSZ={int(best_subj['FRMSZ_ms'])}ms | OVLP={int(best_subj['OVLP'])}% | Floor={best_subj['Floor']:.4f}")
print(f"     Score: {best_subj['Subjective_Score']:.4f}")

print(f"\n  🏆 For BALANCED (50/50):")
print(f"     Config: {best_balanced_config_id}")
print(f"     {best_balanced['Freq_spacing']} | Nband={int(best_balanced['Nband'])} | FRMSZ={int(best_balanced['FRMSZ_ms'])}ms | OVLP={int(best_balanced['OVLP'])}% | Floor={best_balanced['Floor']:.4f}")
print(f"     Score: {best_balanced['Balanced_Score']:.4f}")

if best_obj_config_id == best_subj_config_id:
    print(f"\n  ✅ EXCELLENT: Same configuration excels at both!")

print(f"\n3. PARAMETER CONSISTENCY ACROSS SNRs:")
print(f"{'='*100}")
for result in consistency_analysis:
    print(f"\n  {result['Parameter']}:")
    print(f"    Objective:  '{result['Obj_Most_Common']}' (consistent {result['Obj_Consistency_%']:.0f}% of SNRs)")
    print(f"    Subjective: '{result['Subj_Most_Common']}' (consistent {result['Subj_Consistency_%']:.0f}% of SNRs)")
    print(f"    Agreement:  {result['Agreement_%']:.0f}% of SNRs have same optimal value")
    
    if result['Agreement_%'] >= 80:
        print(f"    ✅ HIGHLY CONSISTENT - Can use fixed value")
    elif result['Agreement_%'] < 50:
        print(f"    ⚠️  SNR-DEPENDENT - Consider adaptive selection")

print(f"\n4. LOW SNR RECOMMENDATIONS (SNR ≤ 0 dB):")
print(f"{'='*100}")
low_snr_data = df_success[df_success['SNR_dB'] <= 0]

if len(low_snr_data) > 0:
    for param in ['Freq_spacing', 'Nband', 'OVLP']:
        best_obj_val = low_snr_data.groupby(param)['Objective_Score'].mean().idxmax()
        best_obj_score = low_snr_data.groupby(param)['Objective_Score'].mean().max()
        best_subj_val = low_snr_data.groupby(param)['Subjective_Score'].mean().idxmax()
        best_subj_score = low_snr_data.groupby(param)['Subjective_Score'].mean().max()
        
        print(f"\n  {param}:")
        print(f"    Objective: {best_obj_val} (score: {best_obj_score:.4f})")
        print(f"    Subjective: {best_subj_val} (score: {best_subj_score:.4f})")

print(f"\n5. HIGH SNR RECOMMENDATIONS (SNR ≥ 10 dB):")
print(f"{'='*100}")
high_snr_data = df_success[df_success['SNR_dB'] >= 10]

if len(high_snr_data) > 0:
    for param in ['Freq_spacing', 'Nband', 'OVLP']:
        best_obj_val = high_snr_data.groupby(param)['Objective_Score'].mean().idxmax()
        best_obj_score = high_snr_data.groupby(param)['Objective_Score'].mean().max()
        best_subj_val = high_snr_data.groupby(param)['Subjective_Score'].mean().idxmax()
        best_subj_score = high_snr_data.groupby(param)['Subjective_Score'].mean().max()
        
        print(f"\n  {param}:")
        print(f"    Objective: {best_obj_val} (score: {best_obj_score:.4f})")
        print(f"    Subjective: {best_subj_val} (score: {best_subj_score:.4f})")

print(f"\n6. CONFIG AGREEMENT ACROSS SNRs:")
print(f"{'='*100}")
agreement_count = snr_best_configs_df['Same_Config'].sum()
total_snrs = len(snr_best_configs_df)
agreement_pct = (agreement_count / total_snrs) * 100
print(f"  Objective and Subjective agree on best config: {agreement_count}/{total_snrs} SNR levels ({agreement_pct:.0f}%)")

if agreement_pct >= 80:
    print(f"  ✅ HIGHLY CONSISTENT - Objective and subjective metrics align well")
elif agreement_pct < 50:
    print(f"  ⚠️  DIVERGENT - Objective and subjective prefer different configurations")

# Show SNRs where they differ
disagreement_snrs = snr_best_configs_df[~snr_best_configs_df['Same_Config']]['SNR_dB'].tolist()
if disagreement_snrs:
    print(f"\n  SNRs where best Objective ≠ best Subjective: {disagreement_snrs}")

print(f"\n" + "="*100)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR.absolute()}")
print("="*100)

print(f"\n Generated files:")
print(f"  ✓ parameter_objective_vs_subjective_all_snrs.csv")
print(f"  ✓ config_averaged_objective_vs_subjective.csv")
print(f"  ✓ snr_param_objective_vs_subjective.csv")
print(f"  ✓ snr_best_configs_objective_vs_subjective.csv")
print(f"  ✓ parameter_consistency_obj_vs_subj.csv")
print(f"  ✓ objective_vs_subjective_scatter.png")
print(f"  ✓ obj_vs_subj_by_*.png (for each parameter)")
print(f"  ✓ heatmap_obj_vs_subj_freq_spacing.png")
print("="*100 + "\n")