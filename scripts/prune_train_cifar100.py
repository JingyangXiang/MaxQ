import os

NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/cifar100'
set = "CIFAR100"
arch = 'c_resnet18'
conv_bn_type = 'SoftSRSTEConv2dBN'
weight_decay = 0.0005
nesterov = True
no_bn_decay = False

for (N, M) in NMs:
    os.system(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
              f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} "
              f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} "
              f"--nesterov {nesterov}")
# ==================================================================================================================== #
import os

NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/cifar100'
set = "CIFAR100"
arch = 'c_resnet18'
conv_bn_type = 'SoftMaxQConv2DBNV2'
weight_decay = 0.0005
nesterov = True
no_bn_decay = False

for (N, M) in NMs:
    for _ in range(10):
        os.system(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                  f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{conv_bn_type} "
                  f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} "
                  f"--nesterov {nesterov}")

# ==================================================================================================================== #
import os

decay_start_end = [[0, 30], [0, 60], [0, 90], [30, 60], [30, 90], [60, 90]]
NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/cifar100'
set = "CIFAR100"
arch = 'c_resnet18'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0005
nesterov = True
no_bn_decay = False

for (N, M) in NMs:
    for (decay_start, decay_end) in decay_start_end:
        for _ in range(10):
            os.system(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                      f"--save_dir ./{set.lower()}/N:M/nesterov={nesterov}/{arch}-{N}:{M}-{decay_start}-{decay_end}-{conv_bn_type} "
                      f"--N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} --weight-decay {weight_decay} "
                      f"--nesterov {nesterov} --decay-start {decay_start} --decay-end {decay_end}")

# ==================================================================================================================== #
import os

NMs = [[1, 4], [1, 8], [1, 16], [2, 4], [2, 8], [2, 16], [4, 8], [4, 16], [8, 16]]

data_path = './dataset/cifar100'
set = "CIFAR100"
arch = 'c_resnet18'
conv_bn_type = 'SoftMaxQConv2DBNV2Pro'
weight_decay = 0.0005
nesterov = True
no_bn_decay = False
decay_start = 0
decay_end = 30
prune_schedules = ['cos', 'exp']

for (N, M) in NMs:
    for prune_schedule in prune_schedules:
        for _ in range(10):
            save_dir = f"./{set.lower()}/{prune_schedule}/nesterov={nesterov}/{arch}-{N}:{M}-{decay_start}-{decay_end}-{conv_bn_type}"
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 10:
                break
            os.system(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                      f"--save_dir {save_dir} --N {N} --M {M} --decay 0.0002 --conv-bn-type {conv_bn_type} "
                      f"--weight-decay {weight_decay} --nesterov {nesterov} --decay-start {decay_start} "
                      f"--decay-end {decay_end} --prune-schedule {prune_schedule}")
