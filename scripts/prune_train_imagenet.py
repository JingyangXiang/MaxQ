import os

NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet18'
conv_bn_type = 'SoftSRSTEConv2dBN'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
workers = 16
pretrained = "./pretrained/resnet18_pretrained.pth"

for (N, M) in NMs:
    os.system(
        f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
        f"--save_dir ./{set.lower()}/N:M+pretrained/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
        f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
        f"--workers {workers} --pretrained {pretrained} ")
# ==================================================================================================================== #
# ResNet18
import os

NMs = [[2, 4], ]
decay_start = 0
decay_end = 30
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet18'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
workers = 16

for (N, M) in NMs:
    os.system(
        f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
        f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
        f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
        f"--workers {workers} --decay-start {decay_start} --decay-end {decay_end}")
# ==================================================================================================================== #
# ResNet34
import os

NMs = [[2, 4], ]
decay_start = 0
decay_end = 30
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet34'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
workers = 16

for (N, M) in NMs:
    os.system(
        f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
        f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
        f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
        f"--workers {workers} --decay-start {decay_start} --decay-end {decay_end}")
# ==================================================================================================================== #
# ResNet50
import os

NMs = [[2, 4], [1, 4], [2, 8], [1, 16]]
decay_start = 0
decay_end = 30
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'resnet50'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
workers = 16

for (N, M) in NMs:
    os.system(
        f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
        f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
        f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
        f"--workers {workers} --decay-start {decay_start} --decay-end {decay_end}")
# ==================================================================================================================== #
# MobileNetV1
import os

NMs = [[2, 4], [1, 4]]
decay_start = 0
decay_end = 30
data_path = './dataset/imagenet'
set = "ImageNet"
arch = 'mobilenet_v1'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0001
nesterov = False
no_bn_decay = True
warmup_length = 0
workers = 16

for (N, M) in NMs:
    os.system(
        f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
        f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} --warmup-length {warmup_length} "
        f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov} "
        f"--workers {workers} --decay-start {decay_start} --decay-end {decay_end}")
