"""
Speech enhancement metrics computation module.

Computes PESQ, STOI, SI-SDR, and DNSMOS metrics for speech enhancement evaluation.
"""

import torch
import torchaudio
import csv
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any
from torchmetrics.audio import (
    DeepNoiseSuppressionMeanOpinionScore,
    PerceptualEvaluationSpeechQuality, 
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalDistortionRatio
)


def compute_and_save_speech_metrics(
    clean_tensor: torch.Tensor,
    enhanced_tensor: torch.Tensor,
    fs: int,
    clean_name: Optional[str] = None,
    enhanced_name: Optional[str] = None,
    csv_dir: Optional[Union[str, Path]] = None,
    csv_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute speech enhancement metrics between clean and enhanced speech tensors.

    Args:
        clean_tensor: Required. Clean speech tensor (channels, samples) or (samples,).
        enhanced_tensor: Required. Enhanced speech tensor (channels, samples) or (samples,).
        fs: Required. Sampling rate in Hz (must be 8000 or 16000 for PESQ).
        clean_name: Optional base name for the clean signal (used when saving CSV).
        enhanced_name: Optional base name for the enhanced signal (used when saving CSV).
        csv_dir: Optional directory to save CSV metrics file. If provided together
            with `csv_filename`, metrics will be written to disk.
        csv_filename: Optional base name for the CSV file.

    Returns:
        Dictionary containing computed metrics.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # Input validation: tensors and sample rate are required
    if clean_tensor is None or enhanced_tensor is None:
        raise ValueError("clean_tensor and enhanced_tensor must be provided")
    if fs is None:
        raise ValueError("Sampling rate 'fs' must be provided")

    clean = clean_tensor
    enhanced = enhanced_tensor
    enhanced_file_path = None
    # Use provided names for CSV/saving metadata
    clean_filename = clean_name if clean_name is not None else "clean"
    enhanced_filename = enhanced_name if enhanced_name is not None else "enhanced"
    
    # Ensure same length (truncate to shortest)
    min_length = min(clean.shape[-1], enhanced.shape[-1])
    clean = clean[..., :min_length]
    enhanced = enhanced[..., :min_length]
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = clean.to(device)
    enhanced = enhanced.to(device)
    
    # Ensure proper tensor dimensions for metrics (batch dimension required)
    if clean.ndim == 1:
        clean = clean.unsqueeze(0)  # Add batch dimension
    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)
    
    # Convert multi-channel to mono if needed
    if clean.shape[0] > 1:
        clean = torch.mean(clean, dim=0, keepdim=True)
    if enhanced.shape[0] > 1:
        enhanced = torch.mean(enhanced, dim=0, keepdim=True)
    
    # Validate sampling rate for PESQ
    if fs not in [8000, 16000]:
        raise ValueError(f"PESQ requires sampling rate of 8000Hz or 16000Hz, got {fs}Hz")

    if fs == 8000:
        pesq_mode = 'nb'
    else:
        pesq_mode = 'wb'  # narrowband vs wideband
    
    # Initialize metrics with proper device placement
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=pesq_mode).to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=fs).to(device) 
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=fs, personalized=True).to(device)
    
    # Compute metrics
    print("Computing PESQ...")
    pesq_score = pesq_metric(enhanced, clean).item()
    
    print("Computing STOI...")
    stoi_score = stoi_metric(enhanced, clean).item()
    
    print("Computing SI-SDR...")
    si_sdr_score = si_sdr_metric(enhanced, clean).item()
    
    print("Computing DNSMOS...")
    dnsmos_scores = dnsmos_metric(enhanced)
    
    # Handle DNSMOS output format
    if dnsmos_scores.dim() > 1:
        dnsmos_scores = dnsmos_scores.squeeze()
    dnsmos_numpy = dnsmos_scores.cpu().numpy()
    
    # DNSMOS returns 4 scores: [p808_mos, mos_sig, mos_bak, mos_ovr]  
    dnsmos_dict = {
        "DNSMOS_p808_mos": float(dnsmos_numpy[0]),
        "DNSMOS_mos_sig": float(dnsmos_numpy[1]),
        "DNSMOS_mos_bak": float(dnsmos_numpy[2]),
        "DNSMOS_mos_ovr": float(dnsmos_numpy[3])
    }
    
    # Compile all metrics
    metrics_dict = {
        "clean_file": str(clean_filename),
        "enhanced_file": str(enhanced_filename),
        "sampling_rate": fs,
        "PESQ": float(pesq_score),
        "SI_SDR": float(si_sdr_score),
        "STOI": float(stoi_score),
        **dnsmos_dict
    }
    
    # Save to CSV if requested
    if csv_dir is not None and csv_filename is not None:
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive CSV filename
        clean_base = Path(clean_filename).stem
        enhanced_base = Path(enhanced_filename).stem
        csv_base = Path(csv_filename).stem

        final_csv_name = f"{csv_base}_ENH-{enhanced_base}_CLN-{clean_base}.csv"
        csv_path = csv_dir / final_csv_name
        
        # Write metrics to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            writer.writeheader()
            writer.writerow(metrics_dict)
        
        print(f"Metrics saved to: {csv_path}")
    
    # Print summary
    print("\n--- Speech Enhancement Metrics ---")
    print(f"PESQ: {metrics_dict['PESQ']:.3f}")
    print(f"STOI: {metrics_dict['STOI']:.3f}")
    print(f"SI-SDR: {metrics_dict['SI_SDR']:.2f} dB")
    print(f"DNSMOS Overall: {metrics_dict['DNSMOS_mos_ovr']:.3f}")
    print("------------------------------------")
    
    return metrics_dict


if __name__ == "__main__":
    # Example usage
    # Load clean and enhanced files and call the tensor-based API
    clean_path = 'audio_files/clean_speech.wav'
    enhanced_path = 'enhanced_audio/wiener_enhanced.wav'
    clean_tensor, clean_sr = torchaudio.load(clean_path)
    enhanced_tensor, enh_sr = torchaudio.load(enhanced_path)

    if clean_sr != enh_sr:
        raise ValueError(f"Sampling rates do not match: clean={clean_sr}, enhanced={enh_sr}")

    metrics = compute_and_save_speech_metrics(
        clean_tensor=clean_tensor,
        enhanced_tensor=enhanced_tensor,
        fs=clean_sr,
        clean_name=Path(clean_path).name,
        enhanced_name=Path(enhanced_path).name,
        csv_dir="results/metrics",
        csv_filename="enhancement_results.csv"
    )