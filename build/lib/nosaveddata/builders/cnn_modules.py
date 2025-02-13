import torch
from torch.nn import functional as F
from torch import nn

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

import math
from functools import partial
from math import prod
from typing import Callable


from .weight_init import init_gpt



class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x



# Rethinking Atrous Convolution for Semantic Image Segmentation
class ASPP(nn.Module):
    def __init__(self, channels, atrous_rates=(2,4), ks=3):
        super(ASPP, self).__init__()

        self.convs = nn.ModuleList()
        for rate in atrous_rates:
          self.convs.append(nn.Sequential(nn.Conv2d(channels, channels, ks, dilation=rate, padding=(ks//2+rate-1)),
                                          nn.BatchNorm2d(channels)))

        self.conv_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, channels, 1),
                                       nn.BatchNorm2d(channels))

        expand_ratio = len(atrous_rates) + 1
        self.out_conv = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(channels*expand_ratio, channels, 1),
                                      nn.BatchNorm2d(channels),
                                      nn.ReLU(True))

    def forward(self,x):

        xs = []

        for conv in self.convs:
          _x = conv(x)
          xs.append(_x)
        
        _x = self.conv_pool(x)
        upsampled = F.interpolate(_x, size=x.shape[-2:], mode='bilinear', align_corners=False)
        xs.append(upsampled)

        x = torch.cat(xs,-3)
        
        x = self.out_conv(x)
        
        return x


class ASPP1D(nn.Module):
    def __init__(self, channels, atrous_rates=(2,4), ks=3):
        super(ASPP1D, self).__init__()

        self.convs = nn.ModuleList()
        for rate in atrous_rates:
          self.convs.append(nn.Sequential(nn.Conv1d(channels, channels, ks, dilation=rate, padding=(ks//2+rate-1)),
                                          LayerNorm(channels)
                                          )
                                          )

        self.conv_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                       nn.Conv1d(channels, channels, 1),
                                       LayerNorm(channels)
                                       )

        expand_ratio = len(atrous_rates) + 1
        self.out_conv = nn.Sequential(nn.ReLU(True),
                                      nn.Conv1d(channels*expand_ratio, channels, 1),
                                      LayerNorm(channels),
                                      nn.ReLU(True))
    def forward(self,x):

        xs = []

        for conv in self.convs:
          _x = conv(x)
          xs.append(_x)
        
        _x = self.conv_pool(x)
        upsampled = F.interpolate(_x, size=x.shape[-1:], mode='linear', align_corners=False)
        xs.append(upsampled)

        x = torch.cat(xs,-2)
        
        x = self.out_conv(x)
        
        return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv1D") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    # for tracer, math.ceil will make onnx graph become constant
    if isinstance(n_frames, torch.Tensor):
        ideal_length = (torch.ceil(n_frames).long() - 1) * stride + (
            kernel_size - padding_total
        )
    else:
        ideal_length = (math.ceil(n_frames) - 1) * stride + (
            kernel_size - padding_total
        )
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right
    before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

class FishConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super(FishConvNet, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size-self.stride if self.dilation==1 else (self.kernel_size + (self.dilation-1)*(self.kernel_size - 1) ) // 2
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0) 
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self


class FishTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super(FishTransConvNet, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self

# DropPath copied from timm library
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"



class ConvNeXt1D(nn.Module): # From Fish-Speech
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = FishConvNet(
            dim,
            dim,
            kernel_size=kernel_size,
            # padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
            dilation=dilation
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(init_gpt)

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x