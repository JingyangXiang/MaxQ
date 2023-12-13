import os
import random
import time
from typing import Dict, List, Tuple

import amct_onnx as amct
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_image_label(directory: str):
    classes, class_to_idx = find_classes(directory)
    images = []
    labels = []
    for (class_name, label) in class_to_idx.items():
        class_path = os.path.join(directory, class_name)
        images_name = os.listdir(class_path)
        for image in images_name:
            images.append(os.path.join(class_path, image))
            labels.append(label)
    assert len(images) == len(labels) == 50000, f"ImageNet val dataset has 50000 images, but now only {len(images)}"
    return images, labels


def get_val_loader(valdir):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
    )
    return val_loader


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(print_string, log):
    print("{:} {:}".format(time_string(), print_string))
    log.write('{:} {:}\n'.format(time_string(), print_string))
    log.flush()


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def img_postprocess(probs, labels):
    """Do image post-process"""
    assert probs.shape[0] == 1
    label = labels[0]
    # calculate top1 and top5 accuracy
    top1_get = 0
    top5_get = 0
    top5_record = np.argsort(probs.reshape(-1))[-5:]
    if label == top5_record[-1]:
        top1_get += 1
        top5_get += 1
    elif label in top5_record:
        top5_get += 1
    return float(top1_get) / len(labels), float(top5_get) / len(labels)


def onnx_forward(onnx_model, val_loader, print_log, log, print_freq=100):
    """forward"""
    input_names = ['input', ]
    output_names = ['output', ]
    ort_session = ort.InferenceSession(onnx_model, amct.AMCT_SO)

    top1_total = 0
    top5_total = 0
    iterations = len(val_loader)

    for idx, (image, label) in enumerate(val_loader):
        image, label = image.numpy(), label.numpy()
        output = ort_session.run(output_names, {input_names[0]: image})[0]
        top1, top5 = img_postprocess(output, label)
        top1_total += top1
        top5_total += top5
        if (idx + 1) % print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Prec@1 {top1:.3f}%\t'
                      'Prec@5 {top5:.3f}%'.format(
                idx + 1, len(val_loader), top1=top1_total / (idx + 1) * 100, top5=top5_total / (idx + 1) * 100), log)
    return top1_total / iterations * 100, top5_total / iterations * 100


def prepare_image_input(
        images, height=256, width=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Read image files to blobs [batch_size, 3, 224, 224]"""
    input_tensor = np.zeros((len(images), 3, crop_size, crop_size), np.float32)

    imgs = np.zeros((len(images), 3, height, width), np.float32)
    for index, im_file in enumerate(images):
        im_data = cv2.imread(im_file)
        im_data = cv2.resize(im_data, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)

        imgs[index, :, :, :] = im_data.transpose(2, 0, 1).astype(np.float32)

    h_off = int((height - crop_size) / 2)
    w_off = int((width - crop_size) / 2)
    input_tensor = imgs[:, :, h_off: (h_off + crop_size), w_off: (w_off + crop_size)]
    # trans uint8 image data to float
    input_tensor /= 255
    # do channel-wise reduce mean value
    for channel in range(input_tensor.shape[1]):
        input_tensor[:, channel, :, :] -= mean[channel]
    # do channel-wise divide std
    for channel in range(input_tensor.shape[1]):
        input_tensor[:, channel, :, :] /= std[channel]

    return input_tensor

if __name__ == "__main__":
    # test get_image_label
    valdir = "./dataset/imagenet/val"
    images, labels = get_image_label(valdir)
    print(f"Val image num: {len(images)}")

    # test get_val_loader
    val_loader = get_val_loader(valdir)
    for idx, (image, label) in enumerate(val_loader):
        image = image.numpy()
        label = label.numpy()
        print(f"{idx:0>5}", image.shape, label.item())
        if idx > 10:
            break


