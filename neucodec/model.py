from typing import Optional, Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from huggingface_hub import PyTorchModelHubMixin, ModelHubMixin, hf_hub_download
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
        if model_id == "neuphonic/neucodec": 
            ignore_keys = ["fc_post_s", "SemanticDecoder"]
        elif model_id == "neuphonic/distill-neucodec":
            ignore_keys = []

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

        # download meta.yaml to track number of downloads
        _ = hf_hub_download(
            repo_id=model_id,
            filename="meta.yaml",
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
        state_dict = torch.load(ckpt_path, map_location)
        contains_list = lambda s, l: any(i in s for i in l)
        state_dict = {
            k:v for k, v in state_dict.items() 
            if not contains_list(k, ignore_keys)
        }

        # TODO: we can move to strict loading once we clean up the checkpoints
        model.load_state_dict(state_dict, strict=False)

        return model
    
    def _prepare_audio(self, audio_or_path: torch.Tensor | Path | str):
        
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
        pad_for_wav = 320 - (y.shape[-1] % 320)
        y = torch.nn.functional.pad(y, (0, pad_for_wav))
        
        return y
        
    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        """
        Args:
            audio_or_path: torch.Tensor [B, 1, T] | Path | str, input audio

        Returns:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes
        """
         
        # prepare inputs
        y = self._prepare_audio(audio_or_path)  
        all_semantic_features = []
        for i in range(y.size(0)):
            semantic_features = (
                self.feature_extractor(
                    y[i, :].cpu(),
                    sampling_rate=16_000,
                    return_tensors="pt",
                )
                .input_features.to(self.device)
            )
            all_semantic_features.append(semantic_features)
        semantic_features = torch.vstack(all_semantic_features)
       
        # acoustic encoding
        acoustic_emb = self.CodecEnc(y.to(self.device))
        acoustic_emb = acoustic_emb.transpose(1, 2)

        # semantic encoding
        semantic_output = (
            self.semantic_model(semantic_features[:]).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        # concatenate embeddings
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]        
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # quantize
        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes

    def decode_code(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes

        Returns:
            recon: torch.Tensor [B, 1, T], reconstructed 24kHz audio
        """

        fsq_post_emb = self.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
        fsq_post_emb = fsq_post_emb.transpose(1, 2)
        fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2) 
        recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
        return recon
    

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
        self.fc_sq_prior = nn.Linear(512, 768)
        self.fc_post_a = nn.Linear(2048, 1024)
        
    def encode_code(self, audio_or_path:  torch.Tensor | Path | str) -> torch.Tensor:
        """
        Args:
            audio_or_path: torch.Tensor [B, 1, T] | Path | str, input audio

        Returns:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes
        """
         
        # prepare inputs
        y = self._prepare_audio(audio_or_path)
        
        all_semantic_features = []
        for i in range(y.size(0)):
            semantic_features = (
                self.feature_extractor(
                    F.pad(y[i, :].cpu(), (160, 160)),
                    sampling_rate=16_000,
                    return_tensors="pt",
                )
                .input_values.to(self.device)
                .squeeze(0)
            )
            all_semantic_features.append(semantic_features)
        semantic_features = torch.vstack(all_semantic_features)
    
        # acoustic encoding
        fsq_emb = self.fc_sq_prior(self.codec_encoder(y.to(self.device)))
        fsq_emb = fsq_emb.transpose(1, 2)

        # semantic encoding
        semantic_target = self.semantic_model(
            semantic_features
        ).last_hidden_state.transpose(1, 2)
        semantic_target = self.SemanticEncoder_module(semantic_target)

        if fsq_emb.shape[-1] != semantic_target.shape[-1]:
            min_len = min(fsq_emb.shape[-1], semantic_target.shape[-1])
            fsq_emb = fsq_emb[:, :, :min_len]
            semantic_target = semantic_target[:, :, :min_len]

        concat_emb = torch.cat([semantic_target, fsq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)
        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes


class NeuCodecOnnxDecoder(
    ModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):
    
    def __init__(self, onnx_path):
        
        # onnx import
        try: 
            import onnxruntime
        except ImportError as e:
            raise ImportError("Failed to import `onnxruntime`. Install with the following command: pip install onnxruntime") from e
        
        # load model
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=so
        )
        self.sample_rate = 24_000

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
        
        # download the model weights file
        onnx_path = hf_hub_download(
            repo_id=model_id,
            filename="model.onnx",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # download meta.yaml to track number of downloads
        _ = hf_hub_download(
            repo_id=model_id,
            filename="meta.yaml",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # initialize model
        model = cls(onnx_path) #cls(onnx_path)

        # only support CPU
        if map_location != "cpu":
            raise ValueError("The onnx decoder currently only supports CPU runtimes.")

        return model
    
    def encode_code(self, *args, **kwargs):
        raise NotImplementedError(
            "The onnx decoder has no functionality to encode codes, as it only contains the compiled decoder graph."
        )
    
    def decode_code(self, codes: np.ndarray) -> np.ndarray:
        """
        Args:
            fsq_codes: np.array [B, 1, F], 50hz FSQ codes

        Returns:
            recon: np.array [B, 1, T], reconstructed 24kHz audio
        """

        # validate inputs
        if not isinstance(codes, np.ndarray):
            raise ValueError("`Codes` should be an np.array.")
        if not len(codes.shape) == 3 or codes.shape[1] != 1:
            raise ValueError("`Codes` should be of shape [B, 1, F].")

        # run decoder
        recon = self.session.run(
            None, {"codes": codes}
        )[0].astype(np.float32)
        
        return recon
