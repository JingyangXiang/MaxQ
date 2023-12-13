import torch
import torch.nn as nn

from models.conv_type.base_conv import BaseNMConv
from models.conv_type.masker import HardConv2D, SoftConv2D


# implemention for KPP Conv2d
class MaxQConv2dBN(BaseNMConv):
    def __init__(self, conv_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', **kwargs):
        super(MaxQConv2dBN, self).__init__()
        assert kernel_size in [1, 3]
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = conv_type(in_channels, out_channels, kernel_size, stride, padding=padding,
                              padding_mode=padding_mode, bias=bias, dilation=dilation, groups=groups, **kwargs)
        self.scale = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.training:
            self.mask.data = self.get_mask(self.conv.weight)
        out = self.scale(self.conv(x, self.mask))
        return out

    @torch.no_grad()
    def get_mask(self, weight):
        # N:M sparse masks
        weight_nm = weight.detach().abs().permute(0, 2, 3, 1).reshape(-1, self.M)
        index_local = torch.argsort(weight_nm, dim=1)[:, :int(self.M - self.N)]
        mask_local = torch.ones(weight_nm.shape, device=weight_nm.device)
        mask_local = mask_local.scatter_(dim=1, index=index_local, value=0.).reshape(weight.permute(0, 2, 3, 1).shape)
        mask_local = mask_local.permute(0, 3, 1, 2)

        # filter wise sparse masks
        weight_filter = weight.detach().abs().flatten(1)
        index = int((self.M - self.N) / self.M * weight_filter.shape[1])
        index_filter = torch.argsort(weight_filter, dim=1)[:, :index]
        mask_filter = torch.ones(weight_filter.shape, device=weight_filter.device)
        mask_filter = mask_filter.scatter_(dim=1, index=index_filter, value=0.)
        mask_filter = mask_filter.reshape(weight.shape)

        # kernel wise sparse mask
        weight_position = weight.detach().abs().reshape(self.in_channels * self.out_channels, -1).permute(1, 0)
        index = int((self.M - self.N) / self.M * weight_position.shape[1])
        index_position = torch.argsort(weight_position, dim=1)[:, :index]
        mask_position = torch.ones(weight_position.shape, device=weight_position.device)
        mask_position = mask_position.scatter_(dim=1, index=index_position, value=0.)
        mask_position = mask_position.permute(1, 0).reshape(weight.shape)

        mask = mask_local * (1 + mask_filter + mask_position)

        return mask

    @torch.no_grad()
    def do_grad_decay_v1(self, decay):
        self.conv.weight.grad.add_(
            decay * (1 - torch.clamp(self.mask, min=0, max=1)) * self.conv.weight)


class SoftMaxQConv2DBNV1(MaxQConv2dBN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', **kwargs):
        super(SoftMaxQConv2DBNV1, self).__init__(SoftConv2D, in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                 padding_mode=padding_mode, **kwargs)


class HardMaxQConv2DBNV1(MaxQConv2dBN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', **kwargs):
        super(HardMaxQConv2DBNV1, self).__init__(HardConv2D, in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                 padding_mode=padding_mode, **kwargs)
