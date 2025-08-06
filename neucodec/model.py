from pathlib import Path
from typing import Union, Dict, Type
import soundfile as sf
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from typing import Optional
from torchaudio import transforms as T
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

from .codec_encoder import CodecEncoder
from .codec_decoder_vocos import CodecDecoderVocos
from .module import SemanticEncoder


class NeuCodec(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0"
):
    def __init__(self, ckpt_path: str, sample_rate: int, hop_length: int):
        super().__init__()

        # load ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # load modules
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", output_hidden_states=True
        )
        self.semantic_model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)

        # load checkpoint
        self._load_ckpt(ckpt)

    @property
    def device(self):
        return next(self.parameters).device

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
        
        # Download the model weights file
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

        # Initialize model
        model = cls(ckpt_path, 24_000, 480)

        return model
        
    def _load_ckpt(self, ckpt):
        # assign keys to correct model components
        filtered_enc = {}
        filtered_gen = {}
        filtered_post = {}
        filtered_prior = {}
        filtered_semantic = {}
        for key, value in ckpt.items():
            if key.startswith("CodecEnc."):
                new_key = key[len("CodecEnc."):]
                filtered_enc[new_key] = value
            elif key.startswith("generator."):
                new_key = key[len("generator."):]
                filtered_gen[new_key] = value
            elif key.startswith("fc_post_a."):
                new_key = key[len("fc_post_a."):]
                filtered_post[new_key] = value
            elif key.startswith("SemanticEncoder_module."):
                new_key = key[len("SemanticEncoder_module."):]
                filtered_semantic[new_key] = value
            elif key.startswith("fc_prior."):
                new_key = key[len("fc_prior."):]
                filtered_prior[new_key] = value

        # load
        self.CodecEnc.load_state_dict(filtered_enc)
        self.CodecEnc.eval()
        self.generator.load_state_dict(filtered_gen, strict=False)
        self.generator.eval()
        self.fc_post_a.load_state_dict(filtered_post)
        self.fc_post_a.eval()
        self.fc_prior.load_state_dict(filtered_prior)
        self.SemanticEncoder_module.load_state_dict(filtered_semantic)
        self.SemanticEncoder_module.eval()

    @torch.inference_mode()
    def encode_code(
        self,
        input_waveform: torch.Tensor,
        semantic_features: torch.Tensor = None,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        pad_for_wav = 320 - (input_waveform.shape[1] % 320)
        input_waveform = torch.nn.functional.pad(input_waveform, (0, pad_for_wav))

        if semantic_features is None:
            semantic_features = self.feature_extractor(
                input_waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)  # [batch, frames, feat_dim]
        else:
            semantic_features = semantic_features[:, 0, :, :]

        semantic_output = self.semantic_model(semantic_features)
        semantic_hidden_16 = semantic_output.hidden_states[16]
        semantic_hidden_16 = semantic_hidden_16.transpose(
            1, 2
        )  # [batch, hidden_dim, frames]
        semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)
        if len(input_waveform.shape) == 2:
            wav = input_waveform.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
        else:
            wav = input_waveform.to(self.device)

        vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024]
        vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]
        concat_emb = torch.cat(
            [semantic_encoded, vq_emb], dim=1
        )  # [batch, 2048, frames]
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = self.generator(concat_emb, vq=True)
        return vq_code

    @torch.inference_mode()
    def decode_code(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_post_emb = self.generator.quantizer.get_output_from_indices(
            vq_code.transpose(1, 2)
        )
        vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(
            1, 2
        )  # [batch, 1024, frames]
        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[
            0
        ]  # [batch, time]
        return recon_audio

    @torch.inference_mode()
    def autoencode(self, fpath: str, output_fpath: Optional[str] = None):
        y, sr = torchaudio.load(fpath)
        if sr != 16_000:
            y = T.Resample(sr, 16_000)(y)
        vq_codes = self.encode_code(y)
        recon = self.decode_code(vq_codes)

        if output_fpath is None:
            name, fext = os.path.splitext(fpath)
            output_fpath = f"{name}_recon{fext}"

        sf.write(output_fpath, recon[0, 0, :].cpu(), self.sample_rate)

    @torch.inference_mode()
    def batch_encode(
        self, fpaths: list[str], return_tensor: bool = False
    ) -> tuple[list[torch.Tensor], list[int]] | tuple[torch.Tensor, list[int]]:
        # prepare batch
        wavs_batch, semantic_batch, token_durations = self._pad_batch(
            [self._preprocess_file(fpath) for fpath in fpaths]
        )
        vq_codes = self.encode_code(wavs_batch, semantic_batch)

        # return, unpad if we want to
        if return_tensor:
            return vq_codes, list(token_durations)

        unpadded_vq_codes = []
        for idx, token_dur in enumerate(token_durations):
            curr_codes = vq_codes[idx, :, :token_dur]
            unpadded_vq_codes.append(curr_codes)

        return unpadded_vq_codes, None

    @torch.inference_mode()
    def batch_decode(
        self,
        vq_codes: list[torch.Tensor] | torch.Tensor,
        token_durations: Optional[list[int]] = None,
    ):
        # pad tensor if need be
        if isinstance(vq_codes, list):
            vq_codes, token_durations = self._pad_codes(vq_codes)
        else:
            assert token_durations is not None

        # decode
        recons = self.decode_code(vq_codes)

        # unpad
        cut_recons = []
        for idx, token_dur in enumerate(token_durations):
            curr_recon = recons[idx, :, : int(token_dur * self.hop_length)]
            cut_recons.append(curr_recon)

        return cut_recons

    @torch.inference_mode()
    def batch_autoencode(
        self, fpaths: list[str], output_fpaths: Optional[list[str]] = None
    ) -> list[torch.Tensor]:
        vq_codes, token_durations = self.batch_encode(fpaths, return_tensor=True)
        cut_recons = self.batch_decode(vq_codes, token_durations)

        if output_fpaths:
            for recon, output_fpath in zip(cut_recons, output_fpaths):
                sf.write(output_fpath, recon.cpu().numpy()[0, :], self.sample_rate)

        return cut_recons

    def _preprocess_file(self, fpath: str):
        # load and resample
        y, sr = torchaudio.load(fpath)
        if sr != 16_000:
            y = T.Resample(sr, 16_000)(y)

        # compute duration for any cutting we might need to do, in terms of n_tokens
        token_duration = int((y.shape[-1] / 16_000) * 50)

        # get semantic model features: [harry] note i don't think this can be batched
        semantic_model_input = self.feature_extractor(
            y, sampling_rate=16_000, return_tensors="pt"
        ).input_features

        return y.to(self.device), semantic_model_input.to(self.device), token_duration

    def _pad_batch(self, batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
        # unpack batch
        wavs, semantic_features, token_durations = zip(*batch)
        max_length_semantic = max([f.shape[1] for f in semantic_features])
        max_length = max_length_semantic * 320

        # pad wavs
        wavs_padded = []
        for audio in wavs:
            padding = max_length - audio.shape[1]
            if padding > 0:
                padded_audio = F.pad(audio, (0, padding), mode="constant", value=0)
            else:
                padded_audio = audio[:, :max_length]
            wavs_padded.append(padded_audio)
        wavs_tensor = torch.stack(wavs_padded)

        # pad semantic features
        semantic_features_padded = []
        for feat in semantic_features:
            padding = max_length_semantic - feat.shape[1]
            padded_feat = F.pad(feat, (0, 0, 0, padding), mode="constant", value=0)
            semantic_features_padded.append(padded_feat)
        semantic_feature_tensor = torch.stack(semantic_features_padded)

        return wavs_tensor, semantic_feature_tensor, token_durations

    def _pad_codes(self, vq_codes: list[torch.Tensor]):
        max_len = max([i.shape[-1] for i in vq_codes])
        token_durations = []
        padded_codes = []
        for curr_codes in vq_codes:
            curr_len = curr_codes.shape[-1]
            token_durations.append(curr_len)
            padding = max_len - curr_len
            curr_codes = F.pad(curr_codes, (0, padding), mode="constant", value=0)
            padded_codes.append(curr_codes)
        return torch.stack(padded_codes), token_durations