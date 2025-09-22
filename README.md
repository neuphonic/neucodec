# NeuCodec 🎧

HuggingFace 🤗: [Model](https://huggingface.co/neuphonic/neucodec), [Distilled Model](https://huggingface.co/neuphonic/distill-neucodec)


[![NeuCodec Intro](NeuCodec-Thumbnail.jpg)](https://www.youtube.com/watch?v=O7XH1lGZyYY)
<br>
*Click the image above to see NeuCodec in action on Youtube!*

*Created by Neuphonic - building faster, smaller, on-device voice AI*

A lightweight neural codec that encodes audio at just 0.8 kbps - perfect for researchers and builders who need something that *just works* for training high quality text-to-speech models.

# Key Features

🔊 Low bit-rate compression - a speech codec that compresses and reconstructs audio with near-inaudible reconstruction loss
<br>
🎼 Upsamples from 16kHz → 24kHz
<br>
🌍 Ready for real-world use - train your own SpeechLMs without needing to build your own codec
<br>
🏢 Commercial use permitted - use it in your own tools or products
<br>
📊 Released with large pre-encoded datasets - we’ve compressed Emilia-YODAS from 1.7TB to 41GB using NeuCodec, significantly reducing the compute requirements needed for training 
<br>

# Model Details

NeuCodec is a Finite Scalar Quantisation (FSQ) based 0.8kbps audio codec for speech tokenization.
It takes advantage of the following features:

* It uses both audio (BigCodec) and semantic ([Wav2Vec2-BERT](https://huggingface.co/facebook/w2v-bert-2.0)) encoders. 
* We make use of Finite Scalar Quantisation (FSQ) resulting in a single vector for the quantised output, which makes it ideal for downstream modeling with Speech Language Models.
* At 50 tokens/sec and 16 bits per token, the overall bit-rate is 0.8kbps.
* The codec takes in 16kHz input and outputs 24kHz using an upsampling decoder.

Our work largely based on extending the work of [X-Codec2.0](https://huggingface.co/HKUSTAudio/xcodec2).

- **Developed by:** Neuphonic
- **Model type:** Neural Audio Codec
- **License:** apache-2.0
- **Repository:** https://github.com/neuphonic/neucodec
- **Paper:** *Coming soon!*
- **Pre-encoded Datasets**:
  - [Emilia-YODAS-EN](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec)

## Get Started

Use the code below to get started with the model.

To install from pypi in a dedicated environment:

**Using conda + pip:**
```bash
conda create -n neucodec python>3.9
conda activate neucodec
pip install neucodec
```

**Using uv:**
```bash
uv venv neucodec --python 3.10
source neucodec/bin/activate  # On Windows: neucodec\Scripts\activate
uv pip install neucodec
```

If you would like to use the onnx decoder, also install `onnxruntime`:
```bash
pip install onnxruntime
```
Then, to use the regular codec in python:

```python
import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from neucodec import NeuCodec
 
model = NeuCodec.from_pretrained("neuphonic/neucodec")
model.eval().cuda()   
 
y, sr = torchaudio.load(librosa.ex("libri1"))
if sr != 16_000:
    y = T.Resample(sr, 16_000)(y)[None, ...] # (B, 1, T_16)

with torch.no_grad():
    fsq_codes = model.encode_code(y)
    # fsq_codes = model.encode_code(librosa.ex("libri1")) # or directly pass your filepath!
    print(f"Codes shape: {fsq_codes.shape}")  
    recon = model.decode_code(fsq_codes).cpu() # (B, 1, T_24)

torchaudio.save("reconstructed.wav", recon[0, :, :], 24_000)
```

## Training Details

The model was trained using the following data: 
* Emilia-YODAS
* MLS
* LibriTTS
* Fleurs
* CommonVoice
* HUI
* Additional proprietary set

All publically available data was covered by either the CC-BY-4.0 or CC0 license.
