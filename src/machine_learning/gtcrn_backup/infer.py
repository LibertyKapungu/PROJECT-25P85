import os
import torch
import soundfile as sf
from gtcrn import GTCRN


## load model
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## load data and convert to STFT
mix, fs = sf.read(os.path.join('test_wavs', 'mix.wav'), dtype='float32')
assert fs == 16000

## inference
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input[None])[0]
## convert back to time domain
enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

## save enhanced wav
sf.write(os.path.join('test_wavs', 'enh.wav'), enh.detach().cpu().numpy(), fs)


# input = torch.stft(torch.from_numpy(mix), 
#                   n_fft=512,         # FFT size
#                   hop_length=256,     # Hop size
#                   win_length=512,     # Window length
#                   window=torch.hann_window(512).pow(0.5),
#                   return_complex=False)