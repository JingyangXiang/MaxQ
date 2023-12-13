import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline

from utils.data_utils import DALIClassificationIterator


class DALIWrapper(object):
    def __init__(self, dalipipeline, memory_format):
        self.dalipipeline = dalipipeline
        self.memory_format = memory_format
        self.batch_size = dalipipeline.batch_size

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline, self.memory_format)

    def __len__(self):
        return len(self.dalipipeline)

    @staticmethod
    def gen_wrapper(dalipipeline, memory_format):
        for data in dalipipeline:
            if "data" in data[0].keys():
                input = data[0]["data"].contiguous(memory_format=memory_format)
                target = torch.reshape(data[0]["label"], [-1]).cuda().long()
                yield input, target
            else:
                raise NotImplementedError

        dalipipeline.reset()


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.RandomResizedCrop(
            device=dali_device,
            size=[crop, crop],
            interp_type=types.INTERP_LINEAR,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
            antialias=False
        )
        self.hue_rng = ops.random.Uniform(range=[-45, 45])
        self.cmnp = ops.CropMirrorNormalize(
            device=dali_device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images1 = self.res(images)
        output1 = self.cmnp(images1, mirror=self.coin())
        return [output1, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=False)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_LINEAR, antialias=False)
        self.cmnp = ops.CropMirrorNormalize(
            device=dali_device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, data_root, batch_size, num_threads, device_id, crop=224, val_size=256):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir=os.path.join(data_root, 'train'), crop=crop)
        pip_train.build()
        print(f'pip_train.epoch_size("Reader"): {pip_train.epoch_size("Reader")}')
        dali_iter_train = DALIClassificationIterator(
            pip_train,
            label=["data", "label"],
            size=pip_train.epoch_size("Reader")
        )
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                data_dir=os.path.join(data_root, 'val'), crop=crop, size=val_size)
        pip_val.build()
        print(f'pip_val.epoch_size("Reader"): {pip_val.epoch_size("Reader")}')
        dali_iter_val = DALIClassificationIterator(
            pip_val,
            label=["data", "label"],
            size=pip_val.epoch_size("Reader")
        )

        return dali_iter_val


class ImageNetDali:
    def __init__(self, args):
        super(ImageNetDali, self).__init__()
        memory_format = torch.contiguous_format
        data_root = args.data
        train_pipeline = get_imagenet_iter_dali(type='train', data_root=data_root, batch_size=args.batch_size,
                                                num_threads=args.workers, crop=224, device_id=0)
        self.train_loader = DALIWrapper(train_pipeline, memory_format=memory_format)
        val_pipeline = get_imagenet_iter_dali(type='val', data_root=data_root, batch_size=args.batch_size // 4,
                                              num_threads=args.workers, crop=224, device_id=0)
        self.val_loader = DALIWrapper(val_pipeline, memory_format=memory_format)
