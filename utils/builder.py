import torch.nn as nn

import models.conv_type


class Builder(object):
    def __init__(self, conv_bn_layer, nonlinearity):
        self.conv_bn_layer = conv_bn_layer
        self.nonlinearity = nonlinearity

    def convbn(self, kernel_size, in_planes, out_planes, stride=1):
        conv_bn_layer = self.conv_bn_layer

        if kernel_size == 3:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 1:
            conv = conv_bn_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 5:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        return conv

    def convbn3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.convbn(3, in_planes, out_planes, stride=stride)
        return c

    def convbn1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.convbn(1, in_planes, out_planes, stride=stride)
        return c

    def convbn7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.convbn(7, in_planes, out_planes, stride=stride)
        return c

    def convbn5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.convbn(5, in_planes, out_planes, stride=stride)
        return c

    def activation(self, **kwargs):
        if self.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{self.nonlinearity} is not an initialization option!")


def get_builder(args):
    print("==> Conv BN Type: {}".format(args.conv_bn_type))

    conv_bn_layer = getattr(models.conv_type, args.conv_bn_type)
    nonlinearity = args.nonlinearity

    builder = Builder(conv_bn_layer=conv_bn_layer, nonlinearity=nonlinearity)

    return builder
