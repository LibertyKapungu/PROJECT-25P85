"""
Frequency Spacing Comparison Across SNR Levels
==============================================
Creates clean comparison graphs like your example showing
MEL vs LOG vs LINEAR performance at each SNR
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ====================================
# CONFIGURATION
# ====================================
MODE = "hybrid"
# CSV_FILE = rf"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/COLLATED_ALL_RESULTS_{MODE}.csv"
# OUTPUT_DIR = Path(f"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/frequency_space_analysis_{MODE}")
#CSV_FILE = rf"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/COLLATED_ALL_RESULTS_{MODE}.csv"
#OUTPUT_DIR = Path(f"/home/25p85/Gabi/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/frequency_space_analysis_{MODE}")

CSV_FILE = "C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/hybrid/COLLATED_ALL_RESULTS_hybrid.csv"
OUTPUT_DIR = "C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/results/EXP3/spectral/PARAM_SWEEP2/hybrid/frequency_space_analysis_{MODE}"

# OUTPUT_DIR.mkdir(exist_ok=True, parents = True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 5)
plt.rcParams['font.size'] = 10

print("="*100)
print("FREQUENCY SPACING COMPARISON ACROSS SNR")
print("="*100)

# Load data
df = pd.read_csv(CSV_FILE)
df_success = df[df['Status'] == 'Success'].copy()

if 'DNSMOS_mos_ovr' not in df_success.columns and 'DNSMOS_p808_mos' in df_success.columns:
    df_success['DNSMOS_mos_ovr'] = df_success['DNSMOS_p808_mos']

print(f"✓ Loaded {len(df_success)} successful tests\n")

# ====================================
# TEXT SUMMARY BY SNR
# ====================================
print("="*100)
print("FREQUENCY SPACING PERFORMANCE BY SNR")
print("="*100)

for snr in sorted(df_success['SNR_dB'].unique()):
    snr_data = df_success[df_success['SNR_dB'] == snr]
    
    print(f"\n{'='*100}")
    print(f"SNR = {snr} dB")
    print(f"{'='*100}")
    
    for freq in ['mel', 'log', 'linear']:
        freq_data = snr_data[snr_data['Freq_spacing'] == freq]
        
        if len(freq_data) > 0:
            print(f"\n{freq.upper()}:")
            print(f"  PESQ:  {freq_data['PESQ'].mean():.3f} ± {freq_data['PESQ'].std():.3f}")
            print(f"  STOI:  {freq_data['STOI'].mean():.3f} ± {freq_data['STOI'].std():.3f}")
            print(f"  SI-SDR: {freq_data['SI_SDR'].mean():.2f} ± {freq_data['SI_SDR'].std():.2f} dB")
            print(f"  DNSMOS: {freq_data['DNSMOS_mos_ovr'].mean():.3f} ± {freq_data['DNSMOS_mos_ovr'].std():.3f}")
    
    # Winners
    pesq_winner = snr_data.groupby('Freq_spacing')['PESQ'].mean().idxmax()
    stoi_winner = snr_data.groupby('Freq_spacing')['STOI'].mean().idxmax()
    
    print(f"\n  → Winner (PESQ): {pesq_winner.upper()}")
    print(f"  → Winner (STOI): {stoi_winner.upper()}")
    
    # Interpretation
    if snr <= 0:
        print(f"  💡 Low SNR: {stoi_winner.upper()} shows better stability")

# ====================================
# VISUALIZATION 1: LINE PLOTS
# ====================================
print("\n" + "="*100)
print("GENERATING VISUALIZATIONS")
print("="*100)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Frequency Spacing Performance Across SNR Levels', fontsize=16, fontweight='bold')

metrics = [
    ('PESQ', 'PESQ', 'blue'),
    ('STOI', 'STOI', 'green'),
    ('SI_SDR', 'SI-SDR (dB)', 'red'),
    ('DNSMOS_mos_ovr', 'DNSMOS Overall', 'purple')
]

colors = {'mel': '#FF6B6B', 'log': '#4ECDC4', 'linear': '#45B7D1'}

for (metric, ylabel, color), ax in zip(metrics, axes):
    for freq in ['mel', 'log', 'linear']:
        freq_data = df_success[df_success['Freq_spacing'] == freq]
        snr_avg = freq_data.groupby('SNR_dB')[metric].agg(['mean', 'std']).reset_index()
        
        ax.plot(snr_avg['SNR_dB'], snr_avg['mean'], 'o-', 
                label=freq.upper(), linewidth=2.5, markersize=8, color=colors[freq])
        ax.fill_between(snr_avg['SNR_dB'],
                       snr_avg['mean'] - snr_avg['std'],
                       snr_avg['mean'] + snr_avg['std'],
                       alpha=0.2, color=colors[freq])
    
    ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'freq_spacing_across_snr_lines.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: freq_spacing_across_snr_lines.png")
plt.close()

# ====================================
# VISUALIZATION 2: BAR CHART AT EACH SNR
# ====================================
snr_levels = sorted(df_success['SNR_dB'].unique())
n_snrs = len(snr_levels)

fig, axes = plt.subplots(1, n_snrs, figsize=(4*n_snrs, 5))
fig.suptitle('Frequency Spacing Comparison by SNR Level (PESQ & STOI)', 
             fontsize=16, fontweight='bold')

if n_snrs == 1:
    axes = [axes]

for snr, ax in zip(snr_levels, axes):
    snr_data = df_success[df_success['SNR_dB'] == snr]
    
    freq_stats = snr_data.groupby('Freq_spacing').agg({
        'PESQ': 'mean',
        'STOI': 'mean'
    }).reset_index()
    
    x = np.arange(len(freq_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, freq_stats['PESQ'], width, label='PESQ', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, freq_stats['STOI'], width, label='STOI', 
                    color='orange', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='orange')
    
    ax.set_xlabel('Frequency Spacing', fontsize=10, fontweight='bold')
    ax.set_ylabel('PESQ', fontsize=10, fontweight='bold')
    ax2.set_ylabel('STOI', fontsize=10, fontweight='bold', color='orange')
    ax.set_title(f'SNR = {snr} dB', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in freq_stats['Freq_spacing']])
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'freq_spacing_bars_by_snr.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: freq_spacing_bars_by_snr.png")
plt.close()

# ====================================
# VISUALIZATION 3: HEATMAP
# ====================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Frequency Spacing Performance Heatmap', fontsize=16, fontweight='bold')

for (metric, ylabel), ax in zip([('PESQ', 'PESQ'), 
                                  ('STOI', 'STOI'),
                                  ('SI_SDR', 'SI-SDR (dB)'),
                                  ('DNSMOS_mos_ovr', 'DNSMOS Overall')], 
                                 axes.flat):
    # Create pivot table
    pivot = df_success.pivot_table(values=metric, 
                                     index='Freq_spacing', 
                                     columns='SNR_dB', 
                                     aggfunc='mean')
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': ylabel})
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Frequency Spacing', fontsize=11)
    ax.set_yticklabels([t.get_text().upper() for t in ax.get_yticklabels()])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'freq_spacing_heatmap.png', dpi=300, bbox_inches='tight')
print(f" Saved: freq_spacing_heatmap.png")
plt.close()

# ====================================
# WINNER SUMMARY TABLE
# ====================================
winner_summary = []

for snr in sorted(df_success['SNR_dB'].unique()):
    snr_data = df_success[df_success['SNR_dB'] == snr]
    
    pesq_winner = snr_data.groupby('Freq_spacing')['PESQ'].mean().idxmax()
    pesq_score = snr_data.groupby('Freq_spacing')['PESQ'].mean().max()
    
    stoi_winner = snr_data.groupby('Freq_spacing')['STOI'].mean().idxmax()
    stoi_score = snr_data.groupby('Freq_spacing')['STOI'].mean().max()
    
    si_sdr_winner = snr_data.groupby('Freq_spacing')['SI_SDR'].mean().idxmax()
    si_sdr_score = snr_data.groupby('Freq_spacing')['SI_SDR'].mean().max()
    
    winner_summary.append({
        'SNR_dB': snr,
        'PESQ_Winner': pesq_winner.upper(),
        'PESQ_Score': pesq_score,
        'STOI_Winner': stoi_winner.upper(),
        'STOI_Score': stoi_score,
        'SI_SDR_Winner': si_sdr_winner.upper(),
        'SI_SDR_Score': si_sdr_score
    })

winner_df = pd.DataFrame(winner_summary)
winner_df.to_csv(OUTPUT_DIR / 'freq_spacing_winners_by_snr.csv', index=False)

print(f"\n{'='*100}")
print("WINNER SUMMARY TABLE")
print(f"{'='*100}\n")
print(winner_df.to_string(index=False))
print(f"\n✓ Saved: freq_spacing_winners_by_snr.csv")

# ====================================
# KEY RECOMMENDATIONS
# ====================================
print(f"\n{'='*100}")
print("KEY RECOMMENDATIONS")
print(f"{'='*100}")

# Overall winner
overall_pesq = df_success.groupby('Freq_spacing')['PESQ'].mean()
overall_stoi = df_success.groupby('Freq_spacing')['STOI'].mean()

print("\n1. OVERALL BEST (averaged across all SNR levels):")
print(f"   PESQ: {overall_pesq.idxmax().upper()} ({overall_pesq.max():.3f})")
print(f"   STOI: {overall_stoi.idxmax().upper()} ({overall_stoi.max():.3f})")

# Low SNR
low_snr = df_success[df_success['SNR_dB'] <= 0]
low_pesq = low_snr.groupby('Freq_spacing')['PESQ'].mean()
low_stoi = low_snr.groupby('Freq_spacing')['STOI'].mean()

print("\n2. LOW SNR (≤ 0 dB):")
print(f"   PESQ: {low_pesq.idxmax().upper()} ({low_pesq.max():.3f})")
print(f"   STOI: {low_stoi.idxmax().upper()} ({low_stoi.max():.3f})")
print(f"    Recommendation: Use {low_stoi.idxmax().upper()} for better intelligibility at low SNR")

# High SNR
high_snr = df_success[df_success['SNR_dB'] >= 10]
high_pesq = high_snr.groupby('Freq_spacing')['PESQ'].mean()
high_stoi = high_snr.groupby('Freq_spacing')['STOI'].mean()

print("\n3. HIGH SNR (≥ 10 dB):")
print(f"   PESQ: {high_pesq.idxmax().upper()} ({high_pesq.max():.3f})")
print(f"   STOI: {high_stoi.idxmax().upper()} ({high_stoi.max():.3f})")
print(f"    Recommendation: Use {high_pesq.idxmax().upper()} for best quality at high SNR")

# Consistency check
pesq_winners = winner_df['PESQ_Winner'].value_counts()
stoi_winners = winner_df['STOI_Winner'].value_counts()

print("\n4. CONSISTENCY ACROSS SNR:")
print(f"   PESQ winners: {dict(pesq_winners)}")
print(f"   STOI winners: {dict(stoi_winners)}")

if len(pesq_winners) == 1 and len(stoi_winners) == 1 and pesq_winners.index[0] == stoi_winners.index[0]:
    print(f"    {pesq_winners.index[0].upper()} consistently wins across all SNR levels")
    print(f"    Recommendation: Use {pesq_winners.index[0].upper()} for all conditions")
else:
    print(f"     Different winners at different SNR levels")
    print(f"    Recommendation: Consider SNR-adaptive frequency spacing")

print(f"\n{'='*100}")
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*100}\n")