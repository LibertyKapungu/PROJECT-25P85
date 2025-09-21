import torch
import torchaudio
import csv
import os
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

def compute_and_save_speech_metrics(
    clean_filename: str,
    enhanced_filename: str,
    clean_dir: str = None,
    enhanced_dir: str = None,
    clean_tensor: torch.Tensor = None,
    enhanced_tensor: torch.Tensor = None,
    fs: int = None,
    csv_dir: str = None,
    csv_filename: str = None
):
    # Load signals from file if tensors are not provided
    if clean_tensor is None or enhanced_tensor is None:
        if clean_dir is None or enhanced_dir is None:
            raise ValueError("clean_dir and enhanced_dir must be provided when using file input mode.")
        clean_file = os.path.join(clean_dir, clean_filename)
        enhanced_file = os.path.join(enhanced_dir, enhanced_filename)
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
    else:
        clean = clean_tensor
        enhanced = enhanced_tensor
        if fs is None:
            raise ValueError("Sampling rate (fs) must be provided when using tensor inputs.")
        clean_file = None
        enhanced_file = None


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


    # Save to CSV only if both csv_dir and csv_filename are provided
    if csv_dir is not None and csv_filename is not None:
        os.makedirs(csv_dir, exist_ok=True)

        base_csv, ext = os.path.splitext(csv_filename)
        if ext == "":
            ext = ".csv" 

        base_clean = os.path.splitext(clean_filename)[0]
        base_enhanced = os.path.splitext(enhanced_filename)[0]
        final_csv_name = f"{base_csv}_ENH-{base_enhanced}_CLN-{base_clean}{ext}"
        csv_path = os.path.join(csv_dir, final_csv_name)

        header = list(metrics_dict.keys())
        row = list(metrics_dict.values())

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

        print(f"Metrics saved to {csv_path}")

    return metrics_dict


# Example usage
# compute_and_save_speech_metrics(
#     clean_dir="audio_stuff", clean_filename="S_56_02.wav",
#     enhanced_dir="audio_stuff", enhanced_filename="wiener_as_sp21_station_sn0.wav",
#     csv_dir="yoh/metrics", csv_filename="wiener_results.csv"
# )
