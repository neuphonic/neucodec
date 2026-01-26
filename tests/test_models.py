import pytest
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
import librosa
from ..neucodec import NeuCodec, DistillNeuCodec, NeuCodecOnnxDecoder

@pytest.fixture
def example_audio():
    y, sr = torchaudio.load(librosa.ex("libri1"))
    y = T.Resample(sr, 16_000)(y)[None, :]
    return y, 16_000


@pytest.fixture
def example_fpath():
    y, sr = torchaudio.load(librosa.ex("libri1"))
    y = T.Resample(sr, 16_000)(y)[None, :]
    return librosa.ex("libri1"), y, 16_000


@pytest.fixture
def neucodec_fixture():
    return NeuCodec.from_pretrained("neuphonic/neucodec").eval()


@pytest.fixture
def distill_neucodec_fixture():
    return DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").eval()


@pytest.fixture
def onnx_decoder_fixture():
    return NeuCodecOnnxDecoder.from_pretrained("neuphonic/neucodec-onnx-decoder")


@torch.inference_mode()
def test_neucodec_audio(neucodec_fixture, example_audio):
    y, _ = example_audio
    y_true = neucodec_fixture._prepare_audio(y)
    vq_codes = neucodec_fixture.encode_code(y)
    recon = neucodec_fixture.decode_code(vq_codes)
    recon_16 = T.Resample(neucodec_fixture.sample_rate, 16_000)(recon)
    min_len = min(y_true.shape[-1], recon_16.shape[-1])
    assert F.mse_loss(y_true[..., :min_len], recon_16[..., :min_len]) < 0.02


@torch.inference_mode()
def test_distill_neucodec_audio(distill_neucodec_fixture, example_audio):
    y, _ = example_audio
    y_true = distill_neucodec_fixture._prepare_audio(y)
    vq_codes = distill_neucodec_fixture.encode_code(y)
    recon = distill_neucodec_fixture.decode_code(vq_codes)
    recon_16 = T.Resample(distill_neucodec_fixture.sample_rate, 16_000)(recon)
    min_len = min(y_true.shape[-1], recon_16.shape[-1])
    assert F.mse_loss(y_true[..., :min_len], recon_16[..., :min_len]) < 0.02


@torch.inference_mode()
def test_neucodec_fpath(neucodec_fixture, example_fpath):
    fpath, y, _ = example_fpath
    y_true = neucodec_fixture._prepare_audio(fpath)
    vq_codes = neucodec_fixture.encode_code(y)
    recon = neucodec_fixture.decode_code(vq_codes)
    recon_16 = T.Resample(neucodec_fixture.sample_rate, 16_000)(recon)
    min_len = min(y_true.shape[-1], recon_16.shape[-1])
    assert F.mse_loss(y_true[..., :min_len], recon_16[..., :min_len]) < 0.02


@torch.inference_mode()
def test_distill_neucodec_fpath(distill_neucodec_fixture, example_fpath):
    fpath, y, _ = example_fpath
    y_true = distill_neucodec_fixture._prepare_audio(y)
    vq_codes = distill_neucodec_fixture.encode_code(fpath)
    recon = distill_neucodec_fixture.decode_code(vq_codes)
    recon_16 = T.Resample(distill_neucodec_fixture.sample_rate, 16_000)(recon)
    min_len = min(y_true.shape[-1], recon_16.shape[-1])
    assert F.mse_loss(y_true[..., :min_len], recon_16[..., :min_len]) < 0.02


@torch.inference_mode()
def test_onnx_decoder(neucodec_fixture, onnx_decoder_fixture, example_fpath):
    fpath, y, _ = example_fpath
    y_true = neucodec_fixture._prepare_audio(y)
    vq_codes = neucodec_fixture.encode_code(fpath)
    recon = neucodec_fixture.decode_code(vq_codes)
    recon_onnx = onnx_decoder_fixture.decode_code(vq_codes.numpy()) 
    recon_onnx_16 = T.Resample(onnx_decoder_fixture.sample_rate, 16_000)(torch.tensor(recon_onnx))
    min_len = min(recon.shape[-1], recon_onnx.shape[-1])
    assert F.mse_loss(recon[..., :min_len], torch.tensor(recon_onnx[..., :min_len])) < 0.03 # check torch and onnx recons don't deviate
    min_len = min(y_true.shape[-1], recon_onnx_16.shape[-1])
    assert F.mse_loss(y_true[..., :min_len], recon_onnx_16[..., :min_len]) < 0.03 # check deviation from sample


@torch.inference_mode()
def test_neucodec_batched_autoencode(neucodec_fixture, example_audio):
    y, _ = example_audio
    y_batched = torch.vstack([y for _ in range(2)])
    vq_codes = neucodec_fixture.encode_code(y_batched)
    recon = neucodec_fixture.decode_code(vq_codes)
    assert torch.allclose(recon[0, :, :], recon[1, :, :], atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_distill_neucodec_batched_autoencode(distill_neucodec_fixture, example_audio):
    y, _ = example_audio
    y_batched = torch.vstack([y for _ in range(2)])
    vq_codes = distill_neucodec_fixture.encode_code(y_batched)
    recon = distill_neucodec_fixture.decode_code(vq_codes)
    assert torch.allclose(recon[0, :, :], recon[1, :, :], atol=1e-4, rtol=1e-4)