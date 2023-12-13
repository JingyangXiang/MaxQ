import argparse
import os
import sys
from functools import partial

sys.path.append('../')
import torch
import torchvision.models

import models
from utils.builder import get_builder
from utils.net_utils import init_N_M_and_mask


def rep_weight(weight, N, M):
    in_channels, out_channels = weight.shape[:2]
    tau = 1e-2
    # N:M sparse masks
    weight_nm = weight.detach().abs().permute(0, 2, 3, 1).reshape(-1, M)
    index = int(M - N)
    sorted_local, index_local = torch.sort(weight_nm, dim=1)
    mask_local = torch.ones(weight_nm.shape, device=weight_nm.device)
    mask_local = mask_local.scatter_(dim=1, index=index_local[:, :index], value=0.)
    # ================================================================================
    mask_local = mask_local.reshape(weight.permute(0, 2, 3, 1).shape)
    mask_local = mask_local.permute(0, 3, 1, 2)

    # filter wise sparse masks
    weight_filter = weight.detach().abs().flatten(1)
    index = int((M - N) / M * weight_filter.shape[1])
    sorted_filter, index_filter = torch.sort(weight_filter, dim=1)
    threshold_filter = sorted_filter[:, index - 1:index + 1].mean(dim=-1, keepdim=True)
    mask_filter = torch.ones(weight_filter.shape, device=weight_filter.device)
    mask_filter = mask_filter.scatter_(dim=1, index=index_filter[:, :index], value=0.)
    mask_filter = mask_filter * torch.sigmoid((weight_filter - threshold_filter) / tau)
    # print((sorted_filter - threshold_filter).max())
    mask_filter = mask_filter.reshape(weight.shape)

    # kernel wise sparse mask
    mask_position = 0.
    if weight.shape[-1] == 3:
        weight_position = weight.detach().abs().reshape(in_channels * out_channels, -1).permute(1, 0)
        index = int((M - N) / M * weight_position.shape[1])
        sorted_position, index_position = torch.sort(weight_position, dim=1)
        threshold_position = sorted_position[:, index - 1:index + 1].mean(dim=-1, keepdim=True)
        mask_position = torch.ones(weight_position.shape, device=weight_position.device)
        mask_position = mask_position.scatter_(dim=1, index=index_position[:, :index], value=0.)
        mask_position = mask_position * torch.sigmoid((weight_position - threshold_position) / tau)
        # print((sorted_position - threshold_position).max())
        mask_position = mask_position.permute(1, 0).reshape(weight.shape)
    mask = 1 + mask_filter + mask_position
    return mask


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    # MaxQ
    parser.add_argument('--N', type=int, default=2, help='N for N:M sparsity')
    parser.add_argument('--M', type=int, default=4, help='M for N:M sparsity')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1], help='mode for transfer weight')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--conv-bn-type', type=str, default='SoftMaxQConv2DBNV2Pro',
                        choices=models.conv_type.__all__, help='decay type for convtype')
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['relu', ], help='activation for model')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # 1. 加载预训练权重
    # 2. 加载到原本的模型里面
    # 3. 加载MMDet模型
    # 4. 将原本模型的权重映射到MMDet权重的字典
    # 5. MMDet模型加载权重, 校验是否正确转化
    # 6. 如果成功转化, 保存权重, 权重存储文件名称: {arch}-{N}-{M}.pth.tar

    args = parse_arguments()
    print("Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=False, num_classes=1000, builder=get_builder(args=args))
    model.apply(partial(init_N_M_and_mask, N=args.N, M=args.M))

    assert args.pretrained is not None and os.path.exists(args.pretrained)

    state_dict = torch.load(args.pretrained, 'cpu')['state_dict']

    missing_keys_torch, unexpected_keys_torch = model.load_state_dict(state_dict, strict=False)
    print(f"==" * 20)
    print(f"=> Model MaxQ missing Keys: {missing_keys_torch}")
    print(f"=> Model MaxQ unexpected Keys: {unexpected_keys_torch}")

    model_torchvision = torchvision.models.resnet50()
    state_dict_torchvision = {}

    for (key_maxq, value_maxq), (key_torchvision, value_torchvision) in zip(model.named_parameters(),
                                                                            model_torchvision.named_parameters()):
        state_dict_torchvision[key_torchvision] = value_maxq
        print(key_maxq, key_torchvision)

    for (key_maxq, value_maxq), (key_torchvision, _) in zip(filter(lambda x: "mask" not in x[0], model.named_buffers()),
                                                            model_torchvision.named_buffers()):
        state_dict_torchvision[key_torchvision] = value_maxq
        print(key_maxq, key_torchvision)

    if args.mode == 0:
        missing_keys, unexpected_keys = model_torchvision.load_state_dict(state_dict_torchvision, strict=False)
    elif args.mode == 1:
        process_state_dict_torchvision = {}
        for key in state_dict_torchvision:
            weight = state_dict_torchvision[key]
            if len(weight.shape) == 4 and weight.shape[1] != 3:
                mask = rep_weight(weight, args.N, args.M)
                weight = mask * weight
            process_state_dict_torchvision[key] = weight
        missing_keys, unexpected_keys = model_torchvision.load_state_dict(process_state_dict_torchvision, strict=False)
    else:
        raise NotImplementedError
    print(f"==" * 20)
    print(f"=> Model MMDet missing Keys: {missing_keys}")
    print(f"=> Model MMDet unexpected Keys: {unexpected_keys}")
    save_name = f'maxq-{args.arch}-{args.N}-{args.M}-v{args.mode}.pth.tar'
    save_path = os.path.join(os.path.dirname(args.pretrained), save_name)
    torch.save(model_torchvision.state_dict(), save_path)
    print(f"=> Model MMDet save path: {save_path}")
