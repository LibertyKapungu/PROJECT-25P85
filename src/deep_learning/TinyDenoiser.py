"""
TinyDenoiser: Speech enhancement using RNN-based denoising with ONNX inference.

Based on the GreenWaves Technologies tiny_denoiser_v2 repository:
https://github.com/GreenWaves-Technologies/tiny_denoiser_v2

Reference:
    Rusci et al. (2022). "Accelerating RNN-based Speech Enhancement on a Multi-Core MCU 
    with Mixed FP16-INT8 Post-Training Quantization"
"""

import numpy as np
import torch
import onnxruntime
from pathlib import Path
from typing import Union, Tuple
import librosa


class TinyDenoiser:
    """
    TinyDenoiser speech enhancement using ONNX models.
    
    This class implements RNN-based noise suppression using pretrained ONNX models.
    The enhancement is performed by:
    1. Computing STFT of noisy audio
    2. Processing magnitude spectrum through RNN to get suppression mask
    3. Applying mask to STFT
    4. Reconstructing clean audio via iSTFT
    """
    
    # Default STFT parameters (from tiny_denoiser_v2)
    WIN_LENGTH = 400  # 25ms at 16kHz
    HOP_LENGTH = 100  # 6.25ms at 16kHz
    N_FFT = 512
    SAMPLERATE = 16000
    WIN_FUNC = "hann"
    
    @staticmethod
    def preprocessing(
        audio: Union[np.ndarray, torch.Tensor],
        frame_size: int = WIN_LENGTH,
        frame_step: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        win_func: str = WIN_FUNC
    ) -> np.ndarray:
        """
        Compute STFT of input audio.
        
        Args:
            audio: Input audio waveform (1D array/tensor)
            frame_size: Window length for STFT
            frame_step: Hop length for STFT
            n_fft: FFT size
            win_func: Window function type
            
        Returns:
            Complex STFT matrix of shape (n_fft//2 + 1, n_frames)
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Compute STFT
        stft = librosa.stft(
            audio,
            win_length=frame_size,
            n_fft=n_fft,
            hop_length=frame_step,
            window=win_func,
            center=True
        )
        
        return stft
    
    @staticmethod
    def postprocessing(
        stft: np.ndarray,
        frame_size: int = WIN_LENGTH,
        frame_step: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        win_func: str = WIN_FUNC
    ) -> np.ndarray:
        """
        Reconstruct audio from STFT using inverse STFT.
        
        Args:
            stft: Complex STFT matrix of shape (n_fft//2 + 1, n_frames)
            frame_size: Window length for iSTFT
            frame_step: Hop length for iSTFT
            n_fft: FFT size
            win_func: Window function type
            
        Returns:
            Reconstructed audio waveform (1D array)
        """
        audio = librosa.istft(
            stft,
            win_length=frame_size,
            hop_length=frame_step,
            n_fft=n_fft,
            window=win_func,
            center=True
        )
        
        return audio
    
    @staticmethod
    def inference_onnx(
        model_path: Union[str, Path],
        in_features_frames: np.ndarray
    ) -> np.ndarray:
        """
        Run ONNX inference on STFT frames.
        
        Args:
            model_path: Path to ONNX model file
            in_features_frames: STFT frames transposed to (n_frames, n_features)
            
        Returns:
            Masked STFT features of same shape as input
        """
        # Load ONNX model
        onnx_session = onnxruntime.InferenceSession(str(model_path))
        
        # Get input/output info
        input_info = onnx_session.get_inputs()
        output_tensor_names = [output.name for output in onnx_session.get_outputs()]
        
        # Determine RNN state shapes from model
        ordered_rnn_states_shapes = []
        for node in input_info + onnx_session.get_outputs():
            if "state_in" in node.name:
                ordered_rnn_states_shapes.append(node.shape)
        
        # Initialize RNN states
        rnn_states = [np.zeros(shape, dtype=np.float32) for shape in ordered_rnn_states_shapes]
        
        # Process each frame
        masked_features = np.empty_like(in_features_frames)
        
        for i, in_features in enumerate(in_features_frames):
            # Compute magnitude
            in_features_mag = np.abs(in_features)
            
            # Prepare inputs
            inputs = {
                'input': in_features_mag.reshape((1, -1, 1)).astype(np.float32),
            }
            
            # Add RNN states to inputs
            if len(rnn_states) >= 4:  # LSTM states (h and c for 2 layers)
                inputs.update({
                    'rnn1_h_state_in': rnn_states[0],
                    'rnn1_c_state_in': rnn_states[1],
                    'rnn2_h_state_in': rnn_states[2],
                    'rnn2_c_state_in': rnn_states[3]
                })
            elif len(rnn_states) >= 2:  # GRU states (h only for 2 layers)
                inputs.update({
                    'rnn1_h_state_in': rnn_states[0],
                    'rnn2_h_state_in': rnn_states[1]
                })
            
            # Run inference
            outputs = onnx_session.run(output_tensor_names, inputs)
            
            # Extract mask and update states
            feat_mask = outputs[0]
            rnn_states = outputs[1:]
            
            # Apply mask to features
            in_features_out = in_features * feat_mask[0, :, 0]
            masked_features[i] = in_features_out
        
        return masked_features
    
    @classmethod
    def enhance(
        cls,
        noisy_audio: Union[np.ndarray, torch.Tensor],
        fs: int,
        onnx_model: Union[str, Path],
        frame_size: int = WIN_LENGTH,
        frame_step: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        win_func: str = WIN_FUNC
    ) -> Tuple[torch.Tensor, int]:
        """
        Enhance noisy audio using ONNX-based TinyDenoiser model.
        
        This is the main entry point for speech enhancement.
        
        Args:
            noisy_audio: Input noisy audio (1D array or tensor)
            fs: Sample rate of input audio
            onnx_model: Path to ONNX model file
            frame_size: Window length for STFT (default: 400 samples = 25ms @ 16kHz)
            frame_step: Hop length for STFT (default: 100 samples = 6.25ms @ 16kHz)
            n_fft: FFT size (default: 512)
            win_func: Window function type (default: "hann")
            
        Returns:
            Tuple of (enhanced_audio, sample_rate)
                - enhanced_audio: Enhanced audio as torch.Tensor of shape (1, n_samples)
                - sample_rate: Output sample rate (16000 Hz)
        
        Example:
            >>> enhanced, fs = TinyDenoiser.enhance(
            ...     noisy_audio=noisy_waveform,
            ...     fs=16000,
            ...     onnx_model="models/denoiser_GRU_dns.onnx"
            ... )
        """
        # Resample if needed
        if fs != cls.SAMPLERATE:
            if isinstance(noisy_audio, torch.Tensor):
                noisy_audio_np = noisy_audio.cpu().numpy().squeeze()
            else:
                noisy_audio_np = np.array(noisy_audio).squeeze()
            
            noisy_audio_np = librosa.resample(
                noisy_audio_np,
                orig_sr=fs,
                target_sr=cls.SAMPLERATE
            )
        else:
            if isinstance(noisy_audio, torch.Tensor):
                noisy_audio_np = noisy_audio.cpu().numpy().squeeze()
            else:
                noisy_audio_np = np.array(noisy_audio).squeeze()
        
        # Step 1: Preprocessing - Compute STFT
        stft = cls.preprocessing(
            noisy_audio_np,
            frame_size=frame_size,
            frame_step=frame_step,
            n_fft=n_fft,
            win_func=win_func
        )
        
        # Step 2: Inference - Apply RNN model
        # Transpose to (n_frames, n_features) for frame-by-frame processing
        stft_transposed = stft.T
        stft_masked = cls.inference_onnx(onnx_model, stft_transposed)
        
        # Step 3: Postprocessing - Reconstruct audio
        # Transpose back to (n_features, n_frames)
        enhanced_audio = cls.postprocessing(
            stft_masked.T,
            frame_size=frame_size,
            frame_step=frame_step,
            n_fft=n_fft,
            win_func=win_func
        )
        
        # Convert to torch tensor with batch dimension
        enhanced_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0)
        
        return enhanced_tensor, cls.SAMPLERATE
