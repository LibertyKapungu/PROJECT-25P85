import os
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dsp_algorithms.specsub import specsub

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "validation_dataset/noisy_speech/")
output_folder = os.path.join(script_dir, "python_specsub_processed_output/")

if not os.path.isdir(input_folder):
    raise FileNotFoundError(f'Input directory "{input_folder}" does not exist.')

if not os.path.isdir(output_folder):
    raise FileNotFoundError(f'Output directory "{output_folder}" does not exist.')

files = glob.glob(os.path.join(input_folder, "*.wav"))

if not files:
    raise FileNotFoundError(f'No .wav files found in input directory "{input_folder}".')

# --- Process each file using the python translation of specsub ---
print("Processing noisy speech with specsub...")
for i, infile in enumerate(files, start=1):
    filename = os.path.basename(infile)
    outfile = os.path.join(output_folder, f"python_specsub_{filename}")
    
    specsub(infile, outfile)
    
    print(f"Processed {i}/{len(files)}: specsub_{filename}")

print("specsub processing complete\n")
print("=== All specsub processing complete ===")
print(f"Total files processed: {len(files)}")
print(f"Output directory: {output_folder}")