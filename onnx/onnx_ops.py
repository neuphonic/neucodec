# Code altered from:
# https://github.com/biendltb/torch-istft-onnx
# https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Module
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ


# Calculate as: (seconds * sample_rate) / hop_size
# Common values (22.05KHz): 30 secs=~2600 mels, 60 secs=~5200 mels, 100 secs=~8620 mels
# This determines model size
MAX_FRAMES = 6000
EPSILON = 1e-8


class ISTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        normalized: bool = False,
        max_frames: int = MAX_FRAMES,
    ):
        """
        Implementation of inverse Short-Time Fourier Transform (ISTFT) in PyTorch
        Parameters
        ----------
        n_fft: Size of Fourier transform
        hop_length: The distance between neighboring sliding window frames.
        win_length: The size of window frame and STFT filter. (Default: ``n_fft``)
        window: The optional window function. Shape must be 1d and `<= n_fft`. (Default: ``torch.ones(win_length)``)
        normalized: Whether the STFT was normalized. (Default: ``False``)
        max_frames: max estimate of the number of frames in the signal for sum-square window calculation
        """
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.normalized = normalized
        self.forward_transform = None
        scale = self.n_fft / self.hop_length
        fourier_basis = torch.fft.fft(torch.eye(self.n_fft))

        cutoff = int((self.n_fft / 2 + 1))
        fourier_basis_sliced = torch.vstack(
            [torch.real(fourier_basis[:cutoff, :]), torch.imag(fourier_basis[:cutoff, :])]
        )
        inverse_basis = torch.linalg.pinv(scale * fourier_basis_sliced).transpose(0, 1).unsqueeze(1).float()
        fft_window = window
        if fft_window is None:
            fft_window = torch.ones(self.win_length)
        assert n_fft >= self.win_length
        fft_window = pad_center(fft_window, target_length=n_fft)
        # window the bases
        inverse_basis *= fft_window
        window_sum = window_sumsquare(
            fft_window,
            max_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
        )
        # self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float(), persistent=False)
        self.register_buffer("window_sum", window_sum, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Note:
        magnitude = torch.sqrt(real**2 + img**2)
        phase = torch.atan2(img, real)

        :param input: tensor with the last dim is real and imagine parts of the spectrogram
        :return:
        """
        assert input.shape[-1] == 2, "Last dimension must be 2 for magnitude and phase"
        device = input.device
        real, img = input[..., 0], input[..., 1]
        recombine_magnitude_phase = torch.cat([real, img], dim=1)
        inverse_basis_device = self.inverse_basis.to(device)
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            inverse_basis_device,
            stride=self.hop_length,
            padding=0,
        )
        win_dim = inverse_transform.size(-1)
        window_sum_valid = self.window_sum[:win_dim].to(device)
        # remove modulation effects
        inverse_transform = inverse_transform / (window_sum_valid + EPSILON)
        inverse_transform = inverse_transform.squeeze(dim=1)
        inverse_transform *= float(self.n_fft) / self.hop_length

        inverse_transform = inverse_transform[:, int(self.n_fft / 2) :]
        inverse_transform = inverse_transform[:, : -int(self.n_fft / 2)]

        if self.normalized:
            inverse_transform = inverse_transform * torch.sqrt(torch.tensor(self.n_fft))
        return inverse_transform


def window_sumsquare(
    window: Optional[torch.Tensor] = None,
    n_frames: int = MAX_FRAMES,
    hop_length: int = 512,
    win_length: int = 2048,
    n_fft: int = 2048,
) -> torch.Tensor:
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    Returns
    -------
    x : torch.Tensor, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
    win_sq = window
    if win_sq is None:
        win_sq = torch.ones(win_length)
    n = n_fft + hop_length * (n_frames - 1)
    x = torch.zeros((n,))

    win_sq = win_sq**2
    win_sq = pad_center(win_sq, target_length=n_fft, axis=-1)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def pad_center(data: torch.Tensor, target_length: int, axis: int = -1, pad_value: float = 0) -> torch.Tensor:
    """
    Center-pads a tensor along a specified axis to a target size.

    Args:
        data (torch.Tensor): The input tensor to pad.
        target_length (int): The target size along the specified axis.
        axis (int): The axis along which to pad the tensor.
        pad_value (float, optional): The value to use for padding. Defaults is 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    # Get the current size of the tensor along the specified axis
    current_len = data.shape[axis]

    # If the current size is already equal to the target size, return the original tensor
    if current_len == target_length:
        return data

    # Calculate the amount of padding needed on each side
    total_padding = target_length - current_len
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left

    # Create a padding tuple for torch.nn.functional.pad
    # torch.nn.functional.pad expects padding in reverse order of dimensions
    # and pairs for the beginning and end of each dimension
    if axis < 0:
        axis = data.dim() + axis
    pad_width = [0] * (2 * data.dim())
    pad_width[(data.dim() - 1 - axis) * 2] = pad_left
    pad_width[(data.dim() - 1 - axis) * 2 + 1] = pad_right

    # Apply padding
    padded_data = torch.nn.functional.pad(data, pad=pad_width, mode="constant", value=pad_value)

    return padded_data


class OnnxISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft, 
            window=torch.hann_window(n_fft), 
            normalized=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # # S = mag * torch.exp(phase * 1j)
        # # better directly produce the complex value
        # S = mag * (x + 1j * y)
        audio = self.istft(torch.stack([mag * x, mag * y], dim=-1))
        print(f"ISTFT output shape: {audio.shape}, pred shape: {x_pred.shape}")
        return audio.unsqueeze(1), x_pred


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class OnnxResidualFSQ(Module):
    """Adapted from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py"""
    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers,
        dim = None,
        is_channel_first = False,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        soft_clamp_input_value = None,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers
        # soft clamping the input value
        self.soft_clamp_input_value = soft_clamp_input_value
        # layers
        self.levels = levels
        self.layers = nn.ModuleList([])
        levels_tensor = torch.Tensor(levels)
        scales = []
        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)
            fsq = FSQ(
                levels = levels,
                dim = codebook_dim,
                **kwargs
            )
            self.layers.append(fsq)
        assert all([not fsq.has_projections for fsq in self.layers])
        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer('scales', torch.stack(scales), persistent = False)
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of
        self._codebooks = self.codebooks
    
    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch_size, seq_len, num_quant = indices.shape
        all_codes = []
        for q in range(num_quant):
            q_indices = indices[:, :, q] 
            codebook = self.codebooks[q] 
            q_codes = torch.embedding(codebook, q_indices.long())
            all_codes.append(q_codes)
    
        all_codes = torch.stack(all_codes, dim=0)
        scales = self.scales.unsqueeze(1).unsqueeze(1)
        all_codes = all_codes * scales
        
        return all_codes
       
    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices) 
        codes_summed = codes.sum(dim=0) 
        return self.project_out(codes_summed)
    
    def forward(self, x):
        # blank!
        return x