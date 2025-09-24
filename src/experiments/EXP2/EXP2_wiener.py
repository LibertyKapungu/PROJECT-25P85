import pandas as pd
import torchaudio
from pathlib import Path
import sys

current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

output_dir = repo_root / 'sound_data' / 'processed' / 'wiener_processed_outputs' / 'EXP1_output'

import utils.audio_dataset_loader as loader

dataset = loader.load_dataset(repo_root, mode="test")
paired_files = loader.pair_sequentially(dataset["urban"], dataset["ears"])

for urban_path, ears_path in paired_files:

    participant = ears_path.parent.name
    print(f"Urban: {urban_path.name} | EARS: {ears_path.name} | Participant: {participant}")

    loader.prerocess_audio(noisy_audio=urban_path, clean_speech=ears_path, snr_db=5)

    try:
        _ = torchaudio.load(urban_path)
    except Exception as e:
        print(f"Warning loading Urban file {urban_path}: {e}")
    try:
        _ = torchaudio.load(ears_path)
    except Exception as e:
        print(f"Warning loading EARS file {ears_path}: {e}")
