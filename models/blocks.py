# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x*1.0
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x, ref=None):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True, bias=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        self.norm_type = norm_type
        self.in_channels = in_channels
        if norm_type in ['bn']:
            bias = False
        
        stride = 2 if scale == 'down' else 1
        self.scale = scale

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.)/2)))  # 边界填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.avgpool = nn.AvgPool2d(2, 2)  # 特征图缩小为一半
        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        if self.scale == 'down_avg':
            out = self.avgpool(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none'):
        super(ResidualBlock, self).__init__()

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        
        scale_config_dict = {'down': ['none', 'down'], 'up': ['up', 'none'], 'none': ['none', 'none']}
        scale_conf = scale_config_dict[scale]

        self.conv1 = ConvLayer(c_in, c_out, 3, scale_conf[0], norm_type=norm_type, relu_type=relu_type) 
        self.conv2 = ConvLayer(c_out, c_out, 3, scale_conf[1], norm_type=norm_type, relu_type='none')
  
    def forward(self, x):
        identity = self.shortcut_func(x)
        res = self.conv1(x)
        res = self.conv2(res)
        return identity + res
        

class Atten(nn.Module):
    def __init__(self, cin=512, middle=512, cout=512):
        super(Atten, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv2d(cin//2, middle//2, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(cin//2, cin//2, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(cin, middle, kernel_size=1, padding=0)

    def forward(self, x):
        x1, x2 = x.split([x.shape[1]//2, x.shape[1]//2], dim=1)
        x1 = self.conv_1x1(x1)
        sigmoid = self.sigmoid(self.conv_1x1(x1))
        x1 = self.conv_3x3(self.conv_3x3(x1)*sigmoid)

        x2 = self.conv_3x3(self.conv_1x1(x2))
        out = self.conv_1(torch.concat([x1, x2], dim=1))
        res = x + out
        return res


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B, C ,W ,H)
            returns :
                out : self attention value + input feature
                attention: B , N , N (N = Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B , C, (N)
        proj_key   = self.key_conv(x).view(m_batchsize, -1, width * height)  # B ,C , (W*H)
        energy     = torch.bmm(proj_query, proj_key)  # transpose check
        attention  = self.softmax(energy)  # B, (N), (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B , C , N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Gated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, rate=1, activation=nn.ELU()):
        super(Gated_Conv, self).__init__()

        padding = int(rate*(ksize-1)/2)  # 通过卷积将通道数变成输出两倍，其中一半用来做门控
        self.conv = nn.Conv2d(in_ch, 2*out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.activation = activation

    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1]/2), dim=1)  # 将特征图分成两半
        gate = torch.sigmoid(x1[0])   # 将值限制在0-1之间
        out = self.activation(x1[1]*gate)
        return out


class ACS_conv(nn.Module):
    def __init__(self, input_chan, output_chan, ksize, stride, padding, dilation):
        super(ACS_conv, self).__init__()
        self.input_c  = input_chan
        self.output_c = output_chan
        self.ksize    = ksize
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.conv   = nn.Conv2d(self.input_c, self.output_c, kernel_size=self.ksize, stride=self.stride,padding=self.padding, dilation=self.dilation)
        self.cratio = nn.Parameter(torch.ones(2))

    def forward(self, x):
        wight1 = torch.exp(self.cratio[0])/torch.sum(torch.exp(self.cratio))
        chan1  = x.shape[1]*wight1
        c1 = chan1 if chan1%2==0 else chan1-(chan1%2)
        c2 = x.shape[1]-c1
        c1 = int(c1.item())
        c2 = int(c2.item())

        if c1 == 0 or c2 == 0:
            out = self.conv(x)
        else:
            x1, x2 = x.split([c1, c2], dim=1)
            co1 = int(c1/(c1+c2)*self.output_c)
            x1 = nn.Conv2d(c1, co1, kernel_size=self.ksize, stride=self.stride,padding=self.padding, dilation=self.dilation)(x1)
            x2 = nn.Conv2d(c2, self.output_c-co1, kernel_size=self.ksize, stride=self.stride,padding=self.padding, dilation=self.dilation)(x2)
            out = torch.cat([x1, x2], dim=1)
        return out


class RRRB(nn.Module):  # modefy from FMEN
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio * n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio * n_feats, ratio * n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio * n_feats, n_feats, 1, 1, 0)

    def pad_tensor(self, t, pattern):
        pattern = pattern.view(1, -1, 1, 1)
        t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
        t[:, :, 0:1, :] = pattern
        t[:, :, -1:, :] = pattern
        t[:, :, :, 0:1] = pattern
        t[:, :, :, -1:] = pattern
        return t

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = self.pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x
        return out


class ERB(nn.Module):
    """ Enhanced residual block for building FEMN
    <https://arxiv.org/pdf/2204.08397.pdf>
    <https://github.com/NJU-Jet/FMEN>
    Diagram:
        --RRRB--LeakyReLU--RRRB--
    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """
    def __init__(self, n_feats, ratio=2):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.conv2 = RRRB(n_feats, ratio)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out
