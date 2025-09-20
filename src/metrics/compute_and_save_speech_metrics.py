import torch
import torchaudio
import csv
import os
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

def compute_and_save_speech_metrics(clean_file: str, enhanced_file: str, save_csv: str = None):

    clean, fs_clean = torchaudio.load(clean_file)
    enhanced, fs_enhanced = torchaudio.load(enhanced_file)

    # Ensure same sampling rate
    if fs_clean != fs_enhanced:
        raise ValueError(f"Sampling rates do not match: {fs_clean} vs {fs_enhanced}")
    fs = fs_clean

    # Trim to the same length
    min_len = min(clean.shape[1], enhanced.shape[1])
    clean = clean[:, :min_len]
    enhanced = enhanced[:, :min_len]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = clean.to(device)
    enhanced = enhanced.to(device)

    # Ensure batch dimension (N, L)
    if clean.ndim == 1:
        clean = clean.unsqueeze(0)
    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)

    # Choose PESQ mode based on sampling rate
    if fs == 8000:
        pesq_mode = 'nb'  # narrowband
    elif fs == 16000:
        pesq_mode = 'wb'  # wideband
    else:
        raise ValueError("PESQ supports only 8000 Hz (nb) or 16000 Hz (wb)")

    # Initialize metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=pesq_mode).to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=fs).to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=fs, personalized=False).to(device)

    # Compute metrics
    pesq_score = pesq_metric(enhanced, clean).item()
    stoi_score = stoi_metric(enhanced, clean).item()
    si_sdr_score = si_sdr_metric(enhanced, clean).item()

    dnsmos_scores = dnsmos_metric(enhanced).squeeze().cpu().numpy()
    dnsmos_dict = {
        "DNSMOS_p808_mos": dnsmos_scores[0],
        "DNSMOS_mos_sig": dnsmos_scores[1],
        "DNSMOS_mos_bak": dnsmos_scores[2],
        "DNSMOS_mos_ovr": dnsmos_scores[3]
    }

    # Combine metrics
    metrics_dict = {
        "clean_file": clean_file,
        "enhanced_file": enhanced_file,
        "PESQ": pesq_score,
        "SI-SDR": si_sdr_score,
        "STOI": stoi_score
    }
    metrics_dict.update(dnsmos_dict)

    # Save to CSV if path is provided
    if save_csv is not None:
        base_clean = os.path.splitext(os.path.basename(clean_file))[0]
        base_enhanced = os.path.splitext(os.path.basename(enhanced_file))[0]
        csv_filename = os.path.join(save_csv, f"ENHANCED_{base_enhanced}_Clean_{base_clean}.csv")

        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        header = list(metrics_dict.keys())
        row = list(metrics_dict.values())

        # Always create a new CSV
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

        print(f"Metrics saved to {csv_filename}")

    return metrics_dict

# Example usage
# compute_and_save_speech_metrics(
#     "audio_stuff/S_56_02.wav",
#     "audio_stuff/wiener_as_sp21_station_sn0.wav",
#     save_csv="yoh/metrics/"
# )
