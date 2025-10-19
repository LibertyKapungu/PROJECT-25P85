import os
import torch
import soundfile as sf
from gtcrn import GTCRN
import torchaudio

## load model
base_dir = os.path.dirname(__file__)  # directory of the current script
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join(base_dir,'checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## load data
mix, fs = sf.read(os.path.join(base_dir,'test_wavs', 'noisy_input.wav'), dtype='float32')
# assert fs == 16000

if fs != 16000:
    mix_tensor = torch.from_numpy(mix).unsqueeze(0)  # shape: [1, num_samples]
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
    mix_tensor = resampler(mix_tensor)
    mix = mix_tensor.squeeze(0).numpy()
    fs = 16000  # update sampling rate

## inference
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input[None])[0]


real = output[..., 0]
imag = output[..., 1]
complex_output = torch.complex(real, imag)

enh = torch.istft(complex_output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

## save enhanced wav
sf.write(os.path.join(base_dir, 'test_wavs', 'enh_noisy_input2.wav'), enh.detach().cpu().numpy(), fs)
