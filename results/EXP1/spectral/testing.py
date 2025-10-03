import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\PARAM_SWEEP\spectral\spectral_parameter_sweep_results.csv")

# Check for bugs
print("AVRGING value counts:")
print(df['Averaging'].value_counts())

print("\nVAD value counts:")
print(df['VAD'].value_counts())

# Group by key parameters
summary = df.groupby(['Freq_spacing', 'Nband', 'FRMSZ_ms', 'OVLP'])[
    ['PESQ', 'STOI', 'SI_SDR', 'DNSMOS_mos_ovr']
].mean().sort_values('SI_SDR', ascending=False)

print("\nTop 30 configs by SI-SDR:")
print(summary.head(30))

# Check SNR dependence
if 'SNR_dB' in df.columns:
    snr_summary = df.groupby(['Freq_spacing', 'Nband', 'SNR_dB'])[
        ['PESQ', 'STOI']
    ].mean()
    print("\nPerformance by SNR:")
    print(snr_summary)