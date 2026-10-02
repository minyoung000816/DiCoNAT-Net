import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# from basicsr.utils.registry import ARCH_REGISTRY
#
# from .arch_util import to_2tuple, trunc_normal_

from collections import OrderedDict

# for restormer
import numbers
from pdb import set_trace as stx

from einops import rearrange

# for idynamic

from layers import *
import torch.nn.functional as F
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D

# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x

    ##########################################################################


# Lightweight Gated Feature Fusion Module (LGFF)
class LGFF(nn.Module):
    def __init__(self, in_dim, out_dim, ffn_expansion_factor, bias):
        super(LGFF, self).__init__()
        self.project_in = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                                        nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.norm = LayerNorm(out_dim, LayerNorm_type='WithBias')
        self.ffn = GDFN(out_dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.project_in(x)
        x = x + self.ffn(self.norm(x))
        return x

    ##########################################################################


## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    ##########################################################################


## Divide and Multiply Feed-Forward Network (DMFN)
class DMFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DMFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias, dilation=1)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.project_out(x)
        return x


class TransBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel, dilation, ffn_expansion_factor, bias, sa=True):
        super(TransBlock, self).__init__()

        if sa == True:
            self.heads = num_heads
            self.kernel = kernel
            self.dilation = dilation
            self.na2d = NeighborhoodAttention2D(dim=dim, kernel_size=kernel,
                                                dilation=dilation, num_heads=num_heads)
            self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ffn = DMFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.sa = sa

    def forward(self, x):
        if self.sa == True:
            x_norm1 = self.norm1(x)
            x = x + self.attn(x_norm1) * self.chan_mod(x_norm1)
        x = x + self.ffn(self.norm2(x))

        return x

    def attn(self, x):
        # b, c, h, w -> b, h, w, c
        x = x.permute(0, 2, 3, 1)
        x = self.na2d(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def chan_mod(self, x):
        score = self.pool(x)
        score = self.conv(score.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        score = self.sigmoid(score)
        return score.expand_as(x)



class EBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8, norm=False, first=False):
        super(EBlock, self).__init__()
        if first:
            layers = [BasicConv(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=1)]
        else:
            layers = [BasicConv(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=2)]

        for _ in range(num_res):
            layers += [ResBlock(out_channel, out_channel, norm)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_blocks, num_heads, dilation, norm=False, last=False, feature_ensemble=False):
        super(DBlock, self).__init__()
        layers = []
        for _ in range(num_blocks//2):
            layers.append(TransBlock(channel, num_heads, 3, 1, 1, False))
            layers.append(TransBlock(channel, num_heads, 3, dilation, 1, False))

        if last:
            if feature_ensemble == False:
                layers.append(BasicConv(channel, 3, kernel_size=3, norm=norm, relu=False, stride=1))
        else:
            layers.append(
                BasicConv(channel, channel // 2, kernel_size=4, norm=norm, relu=True, stride=2, transpose=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FOrD_v1(nn.Module):
    def __init__(self, channel, rot_opt=False):
        super(FOrD_v1, self).__init__()

        self.decomp = BasicConv(channel, channel, kernel_size=1, relu=False, stride=1)

    def forward(self, x):
        x_decomp1 = self.decomp(x)
        x_decomp1_norm = F.normalize(x_decomp1, p=2, dim=1)
        x_decomp2 = x - torch.unsqueeze(torch.sum(x * x_decomp1_norm, dim=1), 1) * x_decomp1_norm

        if rot_opt:
            x_decomp2 = x_decomp2.transpose(2, 3).flip(2)

        return x_decomp1, x_decomp2


class DiCoNATStage(nn.Module):
    def __init__(self):
        super(DiCoNATStage, self).__init__()

        in_channel = 3
        base_channel = 32

        num_res_ENC = 6

        num_heads = [2, 4, 8]
        num_blocks = [4, 6, 8]

        self.Encoder1 = EBlock(in_channel, base_channel, num_res_ENC, first=True)
        self.Encoder2 = EBlock(base_channel, base_channel * 2, num_res_ENC, norm=False)
        self.Encoder3 = EBlock(base_channel * 2, base_channel * 4, num_res_ENC, norm=False)

        self.Convs1_1 = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.Convs1_2 = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        num_res_DEC = 6

        self.Decoder1_1 = DBlock(base_channel * 4, num_blocks[2], num_heads[2], dilation = 9,  norm=False)
        self.Decoder1_2 = DBlock(base_channel * 2,  num_blocks[1], num_heads[1], dilation = 18, norm=False)
        self.Decoder1_3 = DBlock(base_channel,  num_blocks[0], num_heads[0],last=True, dilation = 36, feature_ensemble=True)
        self.Decoder1_4 = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)


    def forward(self, x):
        output = list()

        x_rot = x.transpose(2, 3).flip(2)

        # Common encoder
        x_encomp1_1 = self.Encoder1(x)
        x_encomp1_2 = self.Encoder2(x_encomp1_1)
        x_encomp1_3 = self.Encoder3(x_encomp1_2)

        x_encomp2_1 = self.Encoder1(x_rot)
        x_encomp2_2 = self.Encoder2(x_encomp2_1)
        x_encomp2_3 = self.Encoder3(x_encomp2_2)

        x_encomp2_3 = x_encomp2_3.transpose(2, 3).flip(3)
        x_middle = x_encomp1_3 + x_encomp2_3

        # Resultant image reconstruction
        x_decomp1 = self.Decoder1_1(x_middle)
        x_decomp1 = self.Convs1_1(torch.cat([x_decomp1, x_encomp1_2], dim=1))
        x_decomp1 = self.Decoder1_2(x_decomp1)
        x_decomp1 = self.Convs1_2(torch.cat([x_decomp1, x_encomp1_1], dim=1))
        x_decomp1 = self.Decoder1_3(x_decomp1)
        x_decomp1 = self.Decoder1_4(x_decomp1)

        x_middle_rot = x_middle.transpose(2, 3).flip(2)

        x_decomp2 = self.Decoder1_1(x_middle_rot)
        x_decomp2 = self.Convs1_1(torch.cat([x_decomp2, x_encomp2_2], dim=1))
        x_decomp2 = self.Decoder1_2(x_decomp2)
        x_decomp2 = self.Convs1_2(torch.cat([x_decomp2, x_encomp2_1], dim=1))
        x_decomp2 = self.Decoder1_3(x_decomp2)
        x_decomp2 = self.Decoder1_4(x_decomp2)

        x_decomp2 = x_decomp2.transpose(2, 3).flip(3)

        x_final = x_decomp1 + x_decomp2 + x

        output.append(x_final)

        return output

class DiCoNATNet(nn.Module):
    def __init__(self):
        super(DiCoNATNet, self).__init__()

        self.diconat_net_1 = DiCoNATStage()
        self.diconat_net_2 = DiCoNATStage()

    def forward(self, x):
        out = self.diconat_net_1(x)
        out = self.diconat_net_2(out[-1])
        return out


def build_net():
    return DiCoNATNet()