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

import amct_onnx as amct
import onnxruntime as ort

from args import parse_arguments
from utils import prepare_image_input


def get_labels_from_txt(label_file):
    """Read all images' name and label from label_file"""
    images = []
    labels = []
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            images.append(line.split(' ')[0])
            labels.append(int(line.split(' ')[1]))
    return images, labels


def img_postprocess_quantization(probs, labels):
    """Do image post-process"""
    # calculate top1 and top5 accuracy
    top1_get = 0
    top5_get = 0
    prob_size = probs.shape[1]
    for index, label in enumerate(labels):
        top5_record = (probs[index, :].argsort())[prob_size - 5: prob_size]
        if label == top5_record[-1]:
            top1_get += 1
            top5_get += 1
        elif label in top5_record:
            top5_get += 1
    return float(top1_get) / len(labels), float(top5_get) / len(labels)


def onnx_forward_quantization(onnx_model, label_file, img_dir, batch_size=1, iterations=160):
    """forward"""
    ort_session = ort.InferenceSession(onnx_model, amct.AMCT_SO)

    images, labels = get_labels_from_txt(label_file)
    images = [os.path.join(img_dir, image) for image in images]
    top1_total = 0
    top5_total = 0

    input_names = ['input', ]
    output_names = ['output', ]
    iterations = min(iterations, len(labels) // batch_size)
    for i in range(iterations):
        input_batch = prepare_image_input(images[i * batch_size: (i + 1) * batch_size])
        output = ort_session.run(output_names, {input_names[0]: input_batch})
        top1, top5 = img_postprocess_quantization(output[0], labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5

    return top1_total / iterations, top5_total / iterations


def main():
    """main"""
    args = parse_arguments()
    img_dir = './images'
    label_file = os.path.join(img_dir, 'image_label.txt')
    path = os.path.abspath(args.onnx_path).strip(".onnx")

    OUTPUTS = os.path.join(path, 'outputs/calibration')
    TMP = os.path.join(OUTPUTS, 'tmp')

    model_file = args.onnx_path
    assert eval(os.path.basename(model_file).split('-')[1]) == args.batch_size

    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []

    amct.create_quant_config(config_file=config_json_file, model_file=model_file, skip_layers=skip_layers,
                             batch_num=1, activation_offset=True, config_defination=None)

    # Phase1: do conv+bn fusion, weights calibration and generate calibration model
    scale_offset_record_file = os.path.join(TMP, 'record.txt')
    modified_model = os.path.join(TMP, 'modified_model.onnx')
    amct.quantize_model(config_file=config_json_file, model_file=model_file,
                        modified_onnx_file=modified_model, record_file=scale_offset_record_file)

    before_top1, before_top5 = onnx_forward_quantization(modified_model, label_file=label_file, img_dir=img_dir,
                                                         batch_size=args.batch_size, iterations=32)

    # Phase2: save final model, one for onnx do fake quant test, one
    #         deploy model for ATC
    result_path = os.path.join(OUTPUTS, 'resnet50')
    amct.save_model(modified_model, scale_offset_record_file, result_path)

    # Phase3: run fake_quant model test
    print('[INFO] Do quantized model test:')
    quant_top1, quant_top5 = onnx_forward_quantization('%s_%s' % (result_path, 'fake_quant_model.onnx'),
                                                       label_file=label_file, img_dir=img_dir,
                                                       batch_size=args.batch_size, iterations=32)
    print('[INFO] ResNet18 before quantize  top1:{:>10} top5:{:>10}'.format(before_top1, before_top5))
    print('[INFO] ResNet18 after quantize   top1:{:>10} top5:{:>10}'.format(quant_top1, quant_top5))


if __name__ == '__main__':
    main()
