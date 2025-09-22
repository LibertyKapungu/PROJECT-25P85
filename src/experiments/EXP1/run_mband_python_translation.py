import os
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsp_algorithms.mband import mband

Nbands = 8
freq_spacing = 'log'

script_dir = os.path.dirname(os.path.abspath(__file__))
# input_folder = "C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/loizou_code_verification_OLD_STUFF_DELETE/validation_dataset/noisy_speech"
# output_folder = os.path.join(script_dir, "python_mband_processed_output/")


# if not os.path.isdir(input_folder):
#     raise FileNotFoundError(f'Input directory "{input_folder}" does not exist.')

# if not os.path.isdir(output_folder):
#     raise FileNotFoundError(f'Output directory "{output_folder}" does not exist.')

input_folder = "C:/Users/E7440/Documents/Uni2025/Investigation/PROJECT-25P85/loizou_code_verification_OLD_STUFF_DELETE/validation_dataset/noisy_speech"
#output_folder = os.path.join(script_dir, "python_mband_processed_output/")
output_folder = os.path.join(script_dir, "python_mband_processed_output/")


if not os.path.isdir(input_folder):
    raise FileNotFoundError(f'Input directory "{input_folder}" does not exist.')

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

files = glob.glob(os.path.join(input_folder, "*.wav"))

if not files:
    raise FileNotFoundError(f'No .wav files found in input directory "{input_folder}".')

# --- Process each file using the python translation of mband ---
print("Processing noisy speech with mband...")
for i, infile in enumerate(files, start=1):
    filename = os.path.basename(infile)
    outfile = os.path.join(output_folder, f"python_mband_{filename}")
    
    mband(infile, outfile, Nbands, freq_spacing )
    
    print(f"Processed {i}/{len(files)}: mband_{filename}")

print("mband processing complete\n")
print("=== All mband processing complete ===")
print(f"Total files processed: {len(files)}")
print(f"Output directory: {output_folder}")