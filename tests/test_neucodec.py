import pytest
import torch
import torchaudio
import librosa
from xcodec2 import XCodec2, MiniXCodec2Encoder


@pytest.fixture
def model_16khz():
    return XCodec2.from_cache("16khz")


@pytest.fixture
def model_24khz():
    return XCodec2.from_cache("24khz")


@pytest.fixture
def model_asr_encoder():
    return MiniXCodec2Encoder.from_cache()


@pytest.fixture
def example_audio():
    y, sr = torchaudio.load(librosa.ex("libri1"))
    return y, sr


@pytest.fixture
def example_fpath():
    return librosa.ex("libri1")


@pytest.fixture
def batch_fpaths():
    return [librosa.ex("libri1"), librosa.ex("libri2")]


def load_and_validate_audio(save_path, sample_rate):
    _, sr = torchaudio.load(save_path)
    assert sr == sample_rate


def test_16khz_autoencode(example_fpath, tmp_path, model_16khz):
    save_path = str(tmp_path / "0.wav")
    model_16khz.autoencode(example_fpath, save_path)
    load_and_validate_audio(save_path, 16_000)


def test_24khz_autoencode(example_fpath, tmp_path, model_24khz):
    save_path = str(tmp_path / "0.wav")
    model_24khz.autoencode(example_fpath, save_path)
    load_and_validate_audio(save_path, 24_000)


def test_24khz_encode_decode_single(example_audio, model_24khz):
    y, sr = example_audio
    if sr != 16_000:
        y = torchaudio.transforms.Resample(sr, 16_000)(y)
        sr = 16_000

    # encode
    vq_codes = model_24khz.encode_code(y, sample_rate=sr)
    assert isinstance(vq_codes, torch.Tensor)
    assert vq_codes.dim() == 3  # [batch, channels, time]
    
    # decode
    reconstructed = model_24khz.decode_code(vq_codes)
    assert isinstance(reconstructed, torch.Tensor)
    assert reconstructed.dim() == 3  # [batch, channels, time]


def test_24khz_batch_encode(batch_fpaths, model_24khz):
    vq_codes_list, token_durations = model_24khz.batch_encode(batch_fpaths, return_tensor=False)
    assert isinstance(vq_codes_list, list)
    assert token_durations is None
    assert len(vq_codes_list) == 2
    
    for codes in vq_codes_list:
        assert isinstance(codes, torch.Tensor)
        assert codes.dim() == 2  # [channels, time]


def test_24khz_batch_encode_tensor(batch_fpaths, model_24khz):
    vq_codes_tensor, token_durations = model_24khz.batch_encode(batch_fpaths, return_tensor=True)
    assert isinstance(vq_codes_tensor, torch.Tensor)
    assert isinstance(token_durations, list)
    assert vq_codes_tensor.dim() == 3  # [batch, channels, time]
    assert len(token_durations) == 2
    assert len(set(token_durations)) == 2 # ensure we get two different durations back


def test_24khz_batch_decode(batch_fpaths, model_24khz):
    vq_codes_tensor, token_durations = model_24khz.batch_encode(batch_fpaths, return_tensor=True)
    reconstructed_list = model_24khz.batch_decode(vq_codes_tensor, token_durations)
    assert isinstance(reconstructed_list, list)
    assert len(reconstructed_list) == 2
    for recon in reconstructed_list:
        assert isinstance(recon, torch.Tensor)
        assert recon.dim() == 2  # [channels, time]


def test_24khz_batch_decode_list_input(batch_fpaths, model_24khz):
    vq_codes_list, _ = model_24khz.batch_encode(batch_fpaths, return_tensor=False)
    reconstructed_list = model_24khz.batch_decode(vq_codes_list)
    assert isinstance(reconstructed_list, list)
    assert len(reconstructed_list) == 2
    for recon in reconstructed_list:
        assert isinstance(recon, torch.Tensor)
        assert recon.dim() == 2  # [channels, time]


def test_24khz_batch_autoencode(batch_fpaths, tmp_path, model_24khz):
    output_paths = [str(tmp_path / f"{i}.wav") for i in range(len(batch_fpaths))]
    reconstructed_list = model_24khz.batch_autoencode(batch_fpaths, output_paths)
    assert isinstance(reconstructed_list, list)
    assert len(reconstructed_list) == 2
    for i, output_path in enumerate(output_paths):
        load_and_validate_audio(output_path, 24_000)


def test_asr_encoder_encode(example_audio, model_asr_encoder):
    y, sr = example_audio
    if sr != model_asr_encoder.sample_rate:
        y = torchaudio.transforms.Resample(sr, model_asr_encoder.sample_rate)(y)
    vq_codes = model_asr_encoder.encode_code(y)
    assert isinstance(vq_codes, torch.Tensor)
    assert vq_codes.dim() == 3