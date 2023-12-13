# Code from https://github.com/simochen/model-tools.
import numpy as np

import torch
import torchvision
from torch.autograd import Variable
from models.mobilenet_v1_dense import mobilenet_v1


def print_model_param_nums(model=None, multiply_adds=True, *, prune_rate=0.):
    if model == None:
        model = torchvision.models.alexnet()
    total = 0.
    for name, params in model.named_parameters():
        if len(params.shape) == 4 and params.shape[-1] != 7 and params.shape[1] != 1:
            total += params.nelement() * (1 - prune_rate)
        else:
            total += params.nelement()
    return total / 1e6


def print_model_param_flops(model=None, input_res=224, multiply_adds=False, *, prune_rate=0.):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        # set bias_ops to 1 because merge(conv + bn)
        bias_ops = 1  # if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_para = self.weight.data.numel()
        num_zero = torch.sum(self.weight.data.eq(0)).item()
        # flops = flops * (1 - num_zero / num_para)
        if self.groups == self.in_channels or self.kernel_size[0] == 7:
            flops = flops
        else:
            flops = flops * (1 - prune_rate)
            # print(self, flops)
        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(1, 3, input_res, input_res), requires_grad=True)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_relu))

    return total_flops / 1e9


if __name__ == "__main__":
    str2model = {
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
    }
    NMs = [[1, 1], [1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]
    prune_rates = 1 - np.array([1 / 4, 1 / 8, 1 / 16, 2 / 4, 2 / 8, 2 / 16, 4 / 16])
    for key, value in str2model.items():
        for N, M in NMs:
            model = value()
            prune_rate = 1 - N / M
            total_flops = print_model_param_flops(model, prune_rate=prune_rate)
            print(45 * '=')
            print(key, f"{N}:{M}", '  + Number of FLOPs: %.2fG' % (total_flops))
            total_params = print_model_param_nums(model, prune_rate=prune_rate)
            print(key, f"{N}:{M}", '  + Number of params: %.2fM' % (total_params))

    key = 'mobilenetv1'
    value = mobilenet_v1
    for N, M in NMs:
        model = value()
        prune_rate = 1 - N / M
        total_flops = print_model_param_flops(model, prune_rate=prune_rate)
        print(45 * '=')
        print(key, f"{N}:{M}", '  + Number of FLOPs: %.2fM' % (total_flops * 1e3))
        total_params = print_model_param_nums(model, prune_rate=prune_rate)
        print(key, f"{N}:{M}", '  + Number of params: %.2fM' % (total_params))
