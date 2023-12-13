import os

NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/imagenetsubset'
set = "ImageNetSubset"
arch = 'resnet18'
conv_bn_type = 'SoftSRSTEConv2dBN'
weight_decay = 0.0001
nesterov = True
no_bn_decay = True

for (N, M) in NMs:
    os.system(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
              f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} "
              f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} --nesterov {nesterov}")
