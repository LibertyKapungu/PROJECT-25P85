import torch
import numpy as np
import torchaudio


def create_filtered_dataset_with_vad(vad_model, audio_pairs, config):
    """
    Use VAD to intelligently sample training frames
    Keep some "edge cases" for robustness
    """
    filtered_samples = []
    
    # Thresholds
    CONFIDENT_SILENCE = 0.2   # VAD prob < 0.2 = definitely silence
    CONFIDENT_SPEECH = 0.8    # VAD prob > 0.8 = definitely speech
    UNCERTAIN_LOW = 0.35      # 0.2-0.35 = uncertain silence
    UNCERTAIN_HIGH = 0.65     # 0.65-0.8 = uncertain speech
    
    for clean, noise, noisy in audio_pairs:
        noisy_mag = compute_stft(noisy)    # add actual function 
        noise_psd_true = compute_stft(noise) ** 2
        
        # Get VAD predictions
        with torch.no_grad():
            vad_prob = vad_model(noisy_mag.unsqueeze(0)).squeeze()
        
        # Categorize frames
        confident_silence = vad_prob < CONFIDENT_SILENCE
        confident_speech = vad_prob > CONFIDENT_SPEECH
        uncertain_silence = (vad_prob >= CONFIDENT_SILENCE) & (vad_prob < UNCERTAIN_LOW)
        uncertain_speech = (vad_prob >= UNCERTAIN_HIGH) & (vad_prob <= CONFIDENT_SPEECH)
        ambiguous = (vad_prob >= UNCERTAIN_LOW) & (vad_prob < UNCERTAIN_HIGH)
        
        # Sample strategy:
        # 60% confident silence (pure noise - easy learning)
        # 15% confident speech (speech + noise - harder)
        # 15% uncertain frames (where VAD struggles - robustness)
        # 10% ambiguous frames (edge cases - generalization)
        
        n_frames_total = len(vad_prob)
        
        # Sample from each category
        silence_idx = torch.where(confident_silence)[0]
        speech_idx = torch.where(confident_speech)[0]
        uncertain_idx = torch.where(uncertain_silence | uncertain_speech)[0]
        ambiguous_idx = torch.where(ambiguous)[0]
        
        n_silence = min(len(silence_idx), int(n_frames_total * 0.60))
        n_speech = min(len(speech_idx), int(n_frames_total * 0.15))
        n_uncertain = min(len(uncertain_idx), int(n_frames_total * 0.15))
        n_ambiguous = min(len(ambiguous_idx), int(n_frames_total * 0.10))
        
        # Random sampling
        selected_silence = silence_idx[torch.randperm(len(silence_idx))[:n_silence]]
        selected_speech = speech_idx[torch.randperm(len(speech_idx))[:n_speech]]
        selected_uncertain = uncertain_idx[torch.randperm(len(uncertain_idx))[:n_uncertain]]
        selected_ambiguous = ambiguous_idx[torch.randperm(len(ambiguous_idx))[:n_ambiguous]]
        
        selected_frames = torch.cat([
            selected_silence, 
            selected_speech, 
            selected_uncertain, 
            selected_ambiguous
        ])
        
        # Store samples
        for frame_idx in selected_frames:
            filtered_samples.append({
                'noisy_mag': noisy_mag[frame_idx],
                'noise_psd_target': noise_psd_true[frame_idx],
                'vad_prob': vad_prob[frame_idx],
                'frame_category': categorize_frame(vad_prob[frame_idx])
            })
    
    print(f"Dataset size: {len(audio_pairs) * n_frames_total} → {len(filtered_samples)} frames")
    print(f"Data reduction: {(1 - len(filtered_samples)/(len(audio_pairs)*n_frames_total))*100:.1f}%")
    
    return filtered_samples

def categorize_frame(vad_prob):
    """Helper to label frame type"""
    if vad_prob < 0.2: return 'confident_silence'
    elif vad_prob < 0.35: return 'uncertain_silence'
    elif vad_prob < 0.65: return 'ambiguous'
    elif vad_prob < 0.8: return 'uncertain_speech'
    else: return 'confident_speech'