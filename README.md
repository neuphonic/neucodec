# NeuCodec

HuggingFace: [model](https://huggingface.co/neuphonic/neucodec) / [distilled model](https://huggingface.co/neuphonic/distill-neucodec)

NeuCodec is an FSQ-based audio codec for speech tokenization.

## Model Details

<!-- Provide a longer summary of what this model is. -->

NeuCodec is an ultra low bit-rate audio codec which takes advantage of the following advances;

* It uses both audio (BigCodec encoder) and semantic (Wav2Vec2-BERT-large) information in the encoding process. 
* We make use of Finite Scalar Quantisation (FSQ) resulting in a single vector for the quantised output, which makes it ideal for downstream modeling in SpeechLMs.
* At 50 tokens/sec and 16 bits per token, the overall bit-rate is 0.8kbps.

Our work largely based on extending the work of [X-Codec2.0](https://huggingface.co/HKUSTAudio/xcodec2).

- **Developed by:** Neuphonic
- **Model type:** Neural Audio Codec
- **Language(s):** English
- **License:** apache-2.0

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/neuphonic/neucodec
- **Paper:** *Coming soon*

## Get Started

Use the code below to get started with the model.

To install from pypi in a dedicated environment:

```bash
conda create -n neucodec python>3.9
conda activate neucodec
pip install neucodec
```
Then, to use in python:

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

sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
```

## Training Details

The model was trained on a mix of publicly available and proprietary data. The publicly available data includes the English segments of Emilia-YODAS, MLS, LibriTTS, Fleurs, CommonVoice, and HUI. All publically available data was covered by either the CC-BY-4.0 or CC0 license.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

CMU-Arctic

<!-- This should link to a Dataset Card if possible. -->

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

As we are interested the the degree of distortion from the unencoded to reconstructed audio, our evaluation metrics include. PESQ, STOI, SI-SDR, Mel-Spectrogram MSE, and diff WER.

### Results

| Codec	| Quantizer Token Rate |	Tokens Per Second |	Bitrate |	Codebook size |	Quantizers |	Params |	Autoencoding RTF	| Decoding RTF |	WER (%) |	CER (%) |
| -------- | ------- | -------- | ------- | -------- | ------- | -------- | ------- | -------- | ------- | -------- |
| DAC 	|	75 |	600 |	6kbps |	1024 |	8 |	74.7 |	0.015 |	0.007 |	1.9 |	0.06 |
| Mimi 	|	12.5 |	150	|1.1kbps |	2k	| 8| 	79.3| 	0.012|	0.006|	3.0|	1.4 |
| NeuCodec |	50 |	50|	0.8kbps |	65k|	1|	800|	0.030|	0.003|	2.5|	1.0|

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

Coming Soon