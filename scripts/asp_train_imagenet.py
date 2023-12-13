# ==================================================================================================================== #
# ResNet34 Pretrain
import os

N = 4
M = 4
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet34'
conv_bn_type = 'DenseConv2dBN'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
no_dali = False
workers = 16
epochs = 100

os.system(
    f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} --no-dali {no_dali} "
    f"--save_dir ./{set.lower()}/pretrained/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
    f"--N {N} --M {M} --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
    f"--workers {workers} --epochs {epochs}")
# ==================================================================================================================== #
# ResNet34
import os

NMs = [[2, 4], ]
decay_start = 0
decay_end = 0
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet34'
conv_bn_type = 'SoftRandomConv2dBN'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
no_dali = True
workers = 16
pretrained = './pretrained/resnet34_pretrained.pth'
epochs = 100

os.system(
    f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} --no-dali {no_dali} "
    f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
    f"--N {N} --M {M} --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
    f"--workers {workers} --decay-start {decay_start} --decay-end {decay_end} --pretrained {pretrained}")
