"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.convbn1 = builder.convbn3x3(in_planes, planes, stride=stride)
        self.convbn2 = builder.convbn3x3(planes, planes, stride=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.convbn1x1(in_planes, self.expansion * planes, stride=stride),
            )

    def forward(self, x):
        out = F.relu(self.convbn1(x))
        out = self.convbn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.convbn1 = builder.convbn1x1(in_planes, planes)
        self.convbn2 = builder.convbn3x3(planes, planes, stride=stride)
        self.convbn3 = builder.convbn1x1(planes, self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.convbn1x1(in_planes, self.expansion * planes, stride=stride),
            )

    def forward(self, x):
        out = F.relu(self.convbn1(x))
        out = F.relu(self.convbn2(out))
        out = self.convbn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


def c_resnet18(pretrained=False, builder=None, num_classes=10):
    if pretrained:
        raise NotImplementedError
    return ResNet(builder, BasicBlock, [2, 2, 2, 2], num_classes)


def c_resnet34(pretrained=False, builder=None, num_classes=10):
    if pretrained:
        raise NotImplementedError
    return ResNet(builder, BasicBlock, [3, 4, 6, 3], num_classes)


def c_resnet50(pretrained=False, builder=None, num_classes=10):
    if pretrained:
        raise NotImplementedError
    return ResNet(builder, Bottleneck, [3, 4, 6, 3], num_classes)


def c_resnet101(pretrained=False, builder=None, num_classes=10):
    if pretrained:
        raise NotImplementedError
    return ResNet(builder, Bottleneck, [3, 4, 23, 3], num_classes)


def c_resnet152(pretrained=False, builder=None, num_classes=10):
    if pretrained:
        raise NotImplementedError
    return ResNet(builder, Bottleneck, [3, 8, 36, 3], num_classes)