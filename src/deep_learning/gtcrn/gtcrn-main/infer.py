import os
import torch
import soundfile as sf
from gtcrn import GTCRN

## load model
base_dir = os.path.dirname(__file__)  # directory of the current script
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join(base_dir,'checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## load data
mix, fs = sf.read(os.path.join(base_dir,'test_wavs', 'mband_normal.wav'), dtype='float32')
assert fs == 16000

## inference
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input[None])[0]


real = output[..., 0]
imag = output[..., 1]
complex_output = torch.complex(real, imag)

enh = torch.istft(complex_output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

## save enhanced wav
sf.write(os.path.join(base_dir, 'test_wavs', 'enh_mband_normal.wav'), enh.detach().cpu().numpy(), fs)
