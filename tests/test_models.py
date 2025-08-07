import pytest
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
import librosa
from ..neucodec import NeuCodec, DistillNeuCodec


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
    return NeuCodec.from_pretrained("neuphonic/neucodec")


@pytest.fixture
def distill_neucodec_fixture():
    return DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")


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
