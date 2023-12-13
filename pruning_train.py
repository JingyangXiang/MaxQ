# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import os
import pathlib
import random
import sys
import time
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models

import models
from args import parse_arguments
from data.cifar100 import CIFAR100
from data.imagenet import ImageNet
from data.imagenet_dali import ImageNetDali
from utils.builder import get_builder
from utils.engine import train_engine, validate
from utils.net_utils import AverageMeter, convert_secs2time, get_lr_schedule, init_N_M_and_mask, init_prune_schedule, \
    print_log, save_checkpoint, show_zero_num_and_update_mask, time_string, update_epoch


def main():
    args = parse_arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    best_acc1 = curr_acc1 = 0

    os.makedirs(args.save_dir, exist_ok=True)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("CUDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    print_log("GPU  device : {}".format(torch.cuda.get_device_name()), log)
    # init dataloader
    if args.set in ['ImageNetSubset', 'ImageNet']:
        data_loader = ImageNet(args)
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader
        num_classes = 1000 if args.set == "ImageNet" else 200
        label_smoothing = 0.1
    elif args.set == "ImageNetDali":
        data_loader = ImageNetDali(args)
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader
        num_classes = 1000
        label_smoothing = 0.1
    elif args.set == "CIFAR100":
        data_loader = CIFAR100(args)
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader
        num_classes = 100
        label_smoothing = 0.0
    else:
        raise NotImplementedError

    # create model
    print_log("Creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=args.use_pretrain, num_classes=num_classes,
                                       builder=get_builder(args=args))
    print_log("Model : {}".format(model), log)
    print_log("Parameters: {}".format(args), log)
    print_log("Compress Rate: {:.2f}%".format((args.M - args.N) / args.M * 100), log)
    print_log("Workers         : {}".format(args.workers), log)
    print_log("Learning-Rate   : {}".format(args.lr), log)
    print_log("Use Pre-Trained : {}".format(args.use_pretrain), log)
    print_log("lr adjust : {}".format(args.lr_adjust), log)
    print_log("lr schedule : {}".format(args.lr_schedule), log)
    print_log("label smoothing: {}".format(label_smoothing), log)
    # accelerate
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # init N:M for model
    model.apply(partial(init_N_M_and_mask, N=args.N, M=args.M))

    # init prune schedule
    model.apply(partial(init_prune_schedule, prune_schedule=args.prune_schedule))

    # init model
    model = model.cuda()

    # compile model
    assert not args.compile, "compile will cause something wrong"
    # model = torch.compile(model)

    # define loss function (criterion)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda()

    # define optimizer
    parameters = list(model.named_parameters())
    bn_params = [v for n, v in parameters if len(v.shape) == 1 and v.requires_grad]
    rest_params = [v for n, v in parameters if len(v.shape) != 1 and v.requires_grad]
    optimizer = torch.optim.SGD(
        [
            {
                "params": bn_params,
                "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
            },
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    # define lr schedule
    lr_schedule = get_lr_schedule(args)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print_log("=> loading checkpoint '{}'".format(args.pretrained), log)
            checkpoint = torch.load(args.pretrained)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint['state_dict']
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print_log("=> loaded checkpoint '{}'".format(args.pretrained), log)
            print_log("=> missing_keys'{}'".format(missing_keys), log)
            print_log("=> unexpected_keys'{}'".format(unexpected_keys), log)
            if 'random' in args.conv_bn_type.lower():
                # init prune schedule
                model.apply(partial(init_prune_schedule, prune_schedule=args.prune_schedule))
        else:
            print_log("=> no checkpoint found at '{}'".format(args.pretrained), log)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    if args.evaluate:
        validate(val_loader, model, criterion, log, print_log, args)
        return

    # init path
    filename = os.path.join(args.save_dir, 'checkpoint.{:}.{:}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{:}.{:}.pth.tar'.format(args.arch, args.prefix))

    # print_log(">>>>> accu after is: {:}".format(validate(val_loader, model, criterion, log, print_log, args)), log)

    # start train
    start_time = time.time()
    epoch_time = AverageMeter()

    # compile
    # model = torch.compile(model)

    # train network
    for epoch in range(args.start_epoch, args.epochs):
        lr = lr_schedule(optimizer, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(
            args.arch, epoch, args.epochs, time_string(), need_time), log)

        # init epochs
        model.apply(partial(update_epoch, cur_epoch=epoch, total_epoch=args.epochs,
                            decay_start=args.decay_start, decay_end=args.decay_end))

        # train for one epoch
        train_engine(train_loader, model, criterion, optimizer, epoch, log, print_log, args)

        # show zero
        model.apply(show_zero_num_and_update_mask)

        # evaluate on validation set before mask
        curr_acc1 = validate(val_loader, model, criterion, log, print_log, args)
        # torch.cuda.empty_cache()

        # remember best prec@1 and save checkpoint
        is_best = curr_acc1 > best_acc1
        best_acc1 = max(curr_acc1, best_acc1)
        print_log(f'=> Epoch: {epoch}, LR: {lr:.4f}, Acc: {curr_acc1:.2f}%, Best Acc: {best_acc1:.2f}%', log)
        if args.set != "CIFAR100":
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                             'state_dict': model.state_dict(), 'best_acc1': best_acc1,
                             'optimizer': optimizer.state_dict(), }, is_best, filename, bestname)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()

    # save_results to csv
    write_result_to_csv_scrach(
        N=args.N,
        M=args.M,
        arch=args.arch,
        set=args.set,
        best_acc1=best_acc1,
        curr_acc1=curr_acc1,
        device_target=torch.cuda.get_device_name(),
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        nesterov=args.nesterov,
        lr=args.lr,
        batch_size=args.batch_size,
        conv_bn_type=args.conv_bn_type,
        no_bn_decay=args.no_bn_decay,
        decay_start=args.decay_start,
        decay_end=args.decay_end,
        prune_schedule=args.prune_schedule
    )


def write_result_to_csv_scrach(**kwargs):
    results = pathlib.Path("results.csv")

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "N, "
            "M, "
            "Arch, "
            "Set, "
            "Current Val Top 1, "
            "Best Val Top 1, "
            "Device Target, "
            "Epochs, "
            "Weight Decay, "
            "Seed, "
            "Nesterov, "
            "LearningRate, "
            "BatchSize, "
            "ConvBNType, "
            "NoBNDecay, "
            "DecayStart, "
            "DecayEnd, "
            "PruneSchedule\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            ("{now}, "
             "{N}, "
             "{M}, "
             "{arch}, "
             "{set}, "
             "{curr_acc1:.02f}, "
             "{best_acc1:.02f}, "
             "{device_target}, "
             "{epochs}, "
             "{weight_decay}, "
             "{seed}, "
             "{nesterov}, "
             "{lr}, "
             "{batch_size}, "
             "{conv_bn_type}, "
             "{no_bn_decay}, "
             "{decay_start}, "
             "{decay_end}, "
             "{prune_schedule}\n"
             ).format(now=now, **kwargs)
        )


if __name__ == '__main__':
    main()
