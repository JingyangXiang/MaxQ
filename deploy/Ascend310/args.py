import argparse
import ast

import torch

from utils import time_file_str


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--set', default='ImageNetSubset', type=str,
                        choices=['CIFAR100', "ImageNet", "ImageNetSubset"])
    parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--lr-schedule', default='cos', choices=['step', 'cos'], type=str, help='lr scheduler')
    parser.add_argument('--lr-adjust', type=int, default=30, help='number of epochs that change learning rate')
    parser.add_argument('--warmup-length', type=int, default=0, help='number of epochs that warms up learning rate')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--use-pretrain', dest='use_pretrain', action='store_true',
                        help='use pre-trained model or not')
    parser.add_argument('--nesterov', dest='nesterov', type=ast.literal_eval,
                        help='nesterov for SGD')

    parser.add_argument('--no-bn-decay', dest='no_bn_decay', type=ast.literal_eval,
                        help='not apply weight decay for bn layer')

    parser.add_argument('--compile', dest='compile', type=ast.literal_eval,
                        help='compile for model')
    parser.add_argument('--no-dali', dest='no_dali', type=ast.literal_eval,
                        help='databackend such as pytorch as dali')

    # ONNX Inference
    parser.add_argument('--onnx-path', default='', type=str, metavar='PATH',
                        help='path to onnx checkpoint (default: none)')

    parser.add_argument('--N', type=int, default=2, help='N for N:M sparsity')
    parser.add_argument('--M', type=int, default=4, help='M for N:M sparsity')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.prefix = time_file_str()

    # check params
    if args.set.lower() == 'imagenet':
        assert args.nesterov is False
        assert args.no_bn_decay is True

    return args
