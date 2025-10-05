import sys
from pathlib import Path

# Add the SS folder to Python path
ss_folder = Path(__file__).parent
sys.path.insert(0, str(ss_folder))

# Now import mband from the original translation (file-based interface)
#from mband_causalChange_original_translation import mband
from mband_non_causal_original_translation import mband

# Base paths
base_input_dir = Path('C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS/noisy_speech')
base_output_dir = Path('C:/Users/gabi/Documents/University/Uni2025/Investigation/PROJECT-25P85/Random/Matlab2025Files/SS')

# Parameters
noise_types = ['babble']
snr_levels = [5]
nbands = [4]
freq_spacings = ['linear']

# Loop through all combinations
for noise in noise_types:
    for snr in snr_levels:
        input_filename = f"sp21_{noise}_sn{snr}.wav"
        input_path = base_input_dir / input_filename
        input_path = base_input_dir / input_filename
        if not input_path.exists():
            input_path = base_input_dir / noise / input_filename
        
        for Nband in nbands:
            for Freq_spacing in freq_spacings:
                # Create output folder based on spacing
                output_dir = base_output_dir / f"mband_python_c_{noise}_{Freq_spacing}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create output filename
                output_filename = f"sp21_{noise}_sn{snr}_B{Nband}_{Freq_spacing}.wav"
                output_path = output_dir / output_filename
                
                # Run mband with file paths (original interface)
                mband(
                    filename=str(input_path),
                    outfile=str(output_path),
                    Nband=Nband,
                    Freq_spacing=Freq_spacing
                )
                
                print(f"Saved: {output_path}")

print("\nAll files processed successfully!")
