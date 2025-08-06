from typing import Optional, Dict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2BertModel

from .codec_encoder import CodecEncoder
from .codec_encoder_distill import DistillCodecEncoder
from .codec_decoder_vocos import CodecDecoderVocos
from .module import SemanticEncoder


class NeuCodec(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):
    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", output_hidden_states=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        assert model_id in ["neuphonic/neucodec", "neuphonic/distill-neucodec"]
        # download the model weights file
        ckpt_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # initialize model
        model = cls(24_000, 480)

        # load weights
        state_dict = torch.load(ckpt_path)
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        return model

    def _prepare_audio(self, audio_or_path: Path | torch.Tensor):
        # load from file
        if isinstance(audio_or_path, (Path, str)):
            y, sr = torchaudio.load(audio_or_path)
            if sr != 16_000:
                y, sr = (T.Resample(sr, 16_000)(y), 16_000)
                y = y[None, :]  # [1, T] -> [B, 1, T]

        # ensure input tensor is of correct shape
        elif isinstance(audio_or_path, torch.Tensor):
            y = audio_or_path
            if len(y.shape) == 3:
                y = audio_or_path
            else:
                raise ValueError(
                    f"NeuCodec expects tensor audio input to be of shape [B, 1, T] -- received shape: {y.shape}"
                )

        # pad audio
        pad_for_wav = 320 - (y.shape[1] % 320)
        y = torch.nn.functional.pad(y, (0, pad_for_wav))

        return y

    def encode_code(self, audio_or_path: torch.Tensor) -> torch.Tensor:
        # prepare inputs
        y = self._prepare_audio(audio_or_path)
        semantic_features = self.feature_extractor(
            y, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.device)

        # acoustic encoding
        vq_emb = self.CodecEnc(y.to(self.device))
        vq_emb = vq_emb.transpose(1, 2)

        # semantic encoding
        semantic_output = (
            self.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]

        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = self.generator(concat_emb, vq=True)
        return vq_code

    def decode_code(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_post_emb = self.generator.quantizer.get_output_from_indices(
            vq_code.transpose(1, 2)
        )
        vq_post_emb = vq_post_emb.transpose(1, 2)
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]
        return recon_audio


class DistillNeuCodec(NeuCodec):
    def __init__(self, sample_rate: int, hop_length: int):
        nn.Module.__init__(self)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.semantic_model = HubertModel.from_pretrained(
            "ntu-spml/distilhubert", output_hidden_states=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ntu-spml/distilhubert"
        )
        self.SemanticEncoder_module = SemanticEncoder(768, 768, 1024)
        self.codec_encoder = DistillCodecEncoder()
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_prior = nn.Linear(
            768  # acoustic model
            + 768,  # semantic model
            2048,
        )
        self.fc_prior_sq = nn.Linear(512, 768)
        self.fc_post_a = nn.Linear(2048, 1024)

    def encode_code(self, audio_or_path: torch.Tensor | Path) -> torch.Tensor:
        # prepare inputs
        y = self._prepare_audio(audio_or_path)
        semantic_features = (
            self.feature_extractor(
                F.pad(y[0, :].cpu(), (160, 160)),
                sampling_rate=16_000,
                return_tensors="pt",
            )
            .input_values.to(self.device)
            .squeeze(0)
        )

        # acoustic encoding
        vq_emb = self.fc_prior_sq(self.codec_encoder(y))
        vq_emb = vq_emb.transpose(1, 2)

        # semantic encoding
        semantic_target = self.semantic_model(
            semantic_features
        ).last_hidden_state.transpose(1, 2)
        semantic_target = self.SemanticEncoder_module(semantic_target)

        if vq_emb.shape[-1] != semantic_target.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_target.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_target = semantic_target[:, :, :min_len]

        concat_emb = torch.cat([semantic_target, vq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = self.generator(concat_emb, vq=True)
        return vq_code
