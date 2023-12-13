import torch
import torch.nn as nn

from models.conv_type.base_conv import BaseNMConv
from models.conv_type.masker import HardConv2D, SoftConv2D


class DenseConv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros', **kwargs):
        super(DenseConv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, **kwargs)
        self.scale = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.scale(self.conv(x))


class SRSTEConv2dBN(BaseNMConv):
    def __init__(self, conv_type, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros', **kwargs):
        super(SRSTEConv2dBN, self).__init__()
        self.conv = conv_type(in_channels, out_channels, kernel_size, stride, padding=padding,
                              padding_mode=padding_mode, bias=bias, dilation=dilation, groups=groups, **kwargs)
        self.scale = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.training:
            self.mask.data = self.get_mask(self.conv.weight)
        x = self.scale(self.conv(x, self.mask))
        return x

    @torch.no_grad()
    def get_mask(self, weight):
        weight_temp = weight.detach().abs().permute(0, 2, 3, 1).reshape(-1, self.M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(self.M - self.N)]
        mask = torch.ones(weight_temp.shape, device=weight_temp.device)
        mask = mask.scatter_(dim=1, index=index, value=0.).reshape(weight.permute(0, 2, 3, 1).shape)
        mask = mask.permute(0, 3, 1, 2)
        return mask

    @torch.no_grad()
    def do_grad_decay_v1(self, decay):
        self.conv.weight.grad.add_(decay * (1 - self.mask) * self.conv.weight)


class SoftSRSTEConv2dBN(SRSTEConv2dBN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros', **kwargs):
        super(SoftSRSTEConv2dBN, self).__init__(SoftConv2D, in_channels, out_channels, kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                padding_mode=padding_mode, **kwargs)


class HardSRSTEConv2dBN(SRSTEConv2dBN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros', **kwargs):
        super(HardSRSTEConv2dBN, self).__init__(HardConv2D, in_channels, out_channels, kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                padding_mode=padding_mode, **kwargs)
