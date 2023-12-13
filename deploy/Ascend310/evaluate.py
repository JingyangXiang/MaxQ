"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
"""

import os

from args import parse_arguments
from utils import get_val_loader, onnx_forward, print_log


def main():
    args = parse_arguments()
    val_loader = get_val_loader(os.path.join(args.data, "val"))
    os.makedirs(args.save_dir, exist_ok=True)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')
    print_log("Creating model '{}'".format(args.arch), log)
    ori_top1, ori_top5 = onnx_forward(args.onnx_path, val_loader, print_log, log=log, print_freq=args.print_freq)
    print_log(f'[INFO] ONNX PATH: {args.onnx_path}', log)
    print_log('[INFO] ResNet18 before quantize top1:{:.2f}% top5:{:.2f}%'.format(ori_top1, ori_top5), log)


if __name__ == '__main__':
    main()
