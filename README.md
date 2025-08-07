# NeuCodec

NeuCodec is a Python SDK for working with the Neuphonic codec. 
```
import librosa 
import torchaudio
from neucodec import NeuCodec, DistillNeuCodec

neucodec = NeuCodec.from_pretrained("neuphonic/neucodec")

y, sr = torchaudio.load(librosa.ex("libri1"))
y = T.Resample(sr, 16_000)(y)[None, :]
y, _ = example_audio

y_true = neucodec._prepare_audio(y)

vq_codes = neucodec.encode_code(audio)

recon = neucodec.decode_code(vq_codes)

recon_16 = T.Resample(neucodec.sample_rate, 16_000)(recon)
```