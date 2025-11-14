
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def estfilt1(nChannels, Srate):
    """Estimate filter bank for logarithmic spacing"""
    FS = Srate / 2
    UpperFreq = FS
    LowFreq = 1
    range_log = np.log10(UpperFreq / LowFreq)
    interval = range_log / nChannels
    
    lower = np.zeros(nChannels)
    upper = np.zeros(nChannels)
    
    for i in range(nChannels):
        upper[i] = LowFreq * 10**(interval * (i + 1))
        lower[i] = LowFreq * 10**(interval * i)
    
    return lower, upper

def mel(N, low, high):
    """Mel scale frequency mapping"""
    ac = 1100
    fc = 800
    
    LOW = ac * np.log(1 + low / fc)
    HIGH = ac * np.log(1 + high / fc)
    N1 = N + 1
    
    fmel = LOW + np.arange(1, N1 + 1) * (HIGH - LOW) / N1
    cen2 = fc * (np.exp(fmel / ac) - 1)
    
    lower = cen2[:N]
    upper = cen2[1:N+1]
    
    return lower, upper

def calculate_bands(spacing, nBands, fs, fftl):
    """Calculate frequency bands for different spacing methods"""
    if spacing == 'linear':
        bandsz = int(np.floor(fftl / (2 * nBands)))
        bands = []
        for i in range(nBands):
            lobin = i * bandsz
            hibin = lobin + bandsz - 1
            lof = lobin * fs / fftl
            hif = hibin * fs / fftl
            bands.append((lof, hif))
        return bands
    
    elif spacing == 'log':
        lof, hif = estfilt1(nBands, fs)
        return list(zip(lof, hif))
    
    elif spacing == 'mel':
        lof, hif = mel(nBands, 0, fs / 2)
        return list(zip(lof, hif))

def plot_frequency_bands(nBands=8, fs=16000, fftl=512, figsize=(12, 6)):
    """Plot horizontal frequency band strips for different spacing methods"""
    
    max_freq = fs / 2
    
    # Calculate bands for each method
    linear_bands = calculate_bands('linear', nBands, fs, fftl)
    log_bands = calculate_bands('log', nBands, fs, fftl)
    mel_bands = calculate_bands('mel', nBands, fs, fftl)
    
    # Colors for bands
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#6366f1', '#f97316']
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(f'Frequency Band Scaling Methods (fs={fs/1000:.0f} kHz, {nBands} bands)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    spacing_methods = [
        ('Linear', linear_bands),
        ('Logarithmic', log_bands),
        ('Mel', mel_bands)
    ]
    
    for idx, (name, bands) in enumerate(spacing_methods):
        ax = axes[idx]
        
        # Draw each band
        for i, (lower, upper) in enumerate(bands):
            width = upper - lower
            color = colors[i % len(colors)]
            
            # Draw rectangle for band
            rect = mpatches.Rectangle((lower, 0), width, 1, 
                                     facecolor=color, edgecolor='white', 
                                     linewidth=2)
            ax.add_patch(rect)
            
            # Add band number in center
            center = (lower + upper) / 2
            ax.text(center, 0.5, str(i+1), 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Configure axis
        ax.set_xlim(0, max_freq)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=11, fontweight='bold')
        ax.set_yticks([])
        
        # Only show x-axis on bottom plot
        if idx < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
            # Set major ticks
            major_ticks = np.arange(0, max_freq + 1, 1000)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f'{int(x)}' for x in major_ticks])
            
            # Add minor ticks
            minor_ticks = np.arange(0, max_freq + 1, 500)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', axis='x', alpha=0.3)
            ax.grid(True, which='minor', axis='x', alpha=0.1)
        
        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

# Create and display the plot
fig = plot_frequency_bands(nBands=8, fs=16000, fftl=512, figsize=(12, 6))
plt.savefig('frequency_bands.png', dpi=300, bbox_inches='tight')
plt.show()