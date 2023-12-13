# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import logging
import os
from functools import partial

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify

import models
from args import parse_arguments
from utils.builder import get_builder
from utils.net_utils import init_N_M_and_mask


def parse_arguments():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--set', default='ImageNet', type=str,
                        choices=['CIFAR100', "ImageNet"])
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--N', type=int, default=2, help='N for N:M sparsity')
    parser.add_argument('--M', type=int, default=4, help='M for N:M sparsity')
    parser.add_argument('--decay', type=float, default=0.0002, help='decay for SR-STE method')
    parser.add_argument('--decay-type', type=str, default='v1', choices=['v1', 'v2'], help='decay type for convtype')
    parser.add_argument('--nonlinearity', type=str, default='relu', choices=['relu', ], help='activation for model')
    parser.add_argument('--conv-bn-type', type=str, default='SoftSRSTEConv2dBN',
                        choices=models.conv_type.__all__, help='decay type for convtype')
    args = parser.parse_args()

    return args


def main():
    batch_sizes = [1, 2, 4, 8, 16]
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("[ONNXOPTIMIZER]")
    args = parse_arguments()
    if args.set in ['ImageNetSubset', 'ImageNet']:
        num_classes = 1000 if args.set == "ImageNet" else 200
    elif args.set == "CIFAR100":
        num_classes = 100
    else:
        raise NotImplementedError
    # create model
    logger.info("Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes, builder=get_builder(args=args))

    # init N:M for model
    model.apply(partial(init_N_M_and_mask, N=args.N, M=args.M))

    # check mask
    logger.info(40 * "==")
    checkmasks = []
    for name, module in model.named_modules():
        if hasattr(module, 'mask'):
            mask = torch.clamp(module.mask.permute(0, 2, 3, 1).reshape(-1, args.M), max=1.)
            checkmasks.append(torch.all(torch.eq(mask.sum(-1), args.N)))
    logger.info(f"=> check mask: {np.all(checkmasks)}")
    assert np.all(checkmasks)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logger.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint['state_dict']

            load_checkpoints = {}
            for key in checkpoint.keys():
                weight = checkpoint[key]
                load_checkpoints[key.replace("_orig_mod.", '')] = weight

            missing_keys, unexpected_keys = model.load_state_dict(load_checkpoints, strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.pretrained))
            logger.info("=> missing_keys'{}'".format(missing_keys))
            logger.info("=> unexpected_keys'{}'".format(unexpected_keys))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.pretrained))

        # init mask will be fix by registering mask for buffer
        model.eval()

        model_configurations = []
        export_all_closes = []

        for batch_size in batch_sizes:
            # dummy input
            dummy_input = torch.randn(batch_size, 3, 224, 224)

            # save path
            base_name = os.path.dirname(args.pretrained)

            # check N:M
            dir_name = os.path.basename(base_name)
            N, M = list(map(lambda x: int(x), dir_name.split("-")))
            assert args.N == N and args.M == M

            onnx_output_path = f'{base_name}/batch_size-{batch_size}-N-{args.N}-M-{args.M}-{args.arch}.onnx'

            input_names = ['input', ]
            output_names = ['output', ]

            # export onnx
            torch.onnx.export(model, dummy_input, onnx_output_path, verbose=False, do_constant_folding=True,
                              opset_version=12, input_names=input_names, output_names=output_names)

            # 加载导出的 ONNX 模型
            onnx_model = onnx.load(onnx_output_path)

            # 简化模型
            simplified_model, check = simplify(onnx_model)

            # 保存简化后的模型
            simplified_onnx_output_path = onnx_output_path.replace(".onnx", '-sim.onnx')
            onnx.save_model(simplified_model, simplified_onnx_output_path)

            # 校验导出的结果
            torch_output = model(dummy_input).detach().numpy()
            ort_session = ort.InferenceSession(onnx_output_path)
            onnx_output = ort_session.run(output_names, {input_names[0]: dummy_input.numpy()})
            assert np.allclose(torch_output, onnx_output, atol=1e-3)

            # 保存allclose的结果
            model_configurations.append(os.path.basename(simplified_onnx_output_path))
            export_all_closes.append(np.allclose(torch_output, onnx_output, atol=1e-3))

            # 移除之前没有简化的权重
            os.remove(onnx_output_path)

        length = max(map(lambda x: len(x), model_configurations)) + 2
        for model, export_all_close in zip(model_configurations, export_all_closes):
            logger.info(f"{model:{length}} all_close: {export_all_close}")


if __name__ == '__main__':
    main()
