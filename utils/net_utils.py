from __future__ import division

import random
import shutil
import time
from functools import partial

import numpy as np
import torch


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


def get_lr_schedule(args):
    if args.lr_schedule == 'step':
        lr_schedule = partial(step_lr, args=args)
    elif args.lr_schedule == 'cos':
        lr_schedule = partial(cos_lr, args=args)
    else:
        raise NotImplementedError
    return lr_schedule


def step_lr(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cos_lr(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < args.warmup_length:
        lr = args.lr * (epoch + 1) / args.warmup_length
    else:
        e = epoch - args.warmup_length
        es = args.epochs - args.warmup_length
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * e / es))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{:} {:}".format(time_string(), print_string))
    log.write('{:} {:}\n'.format(time_string(), print_string))
    log.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def forward_func(model, input, label, loss_func):
    output = model(input)
    loss = loss_func(output, label)
    return loss, output


def init_N_M_and_mask(module, N, M):
    if hasattr(module, "init_N_M_and_mask"):
        module.init_N_M_and_mask(N, M)


def init_prune_schedule(module, prune_schedule):
    if hasattr(module, "init_prune_schedule"):
        module.init_prune_schedule(prune_schedule)


def do_grad_decay_v1(module, decay):
    if hasattr(module, "do_grad_decay_v1"):
        module.do_grad_decay_v1(decay)


def do_grad_decay_v2(module, decay):
    if hasattr(module, "do_grad_decay_v2"):
        module.do_grad_decay_v2(decay)


def show_zero_num_and_update_mask(module):
    if hasattr(module, "show_zero_num_and_update_mask"):
        module.show_zero_num_and_update_mask()


def update_epoch(module, cur_epoch, total_epoch, decay_start, decay_end):
    if hasattr(module, "update_epoch"):
        module.update_epoch(cur_epoch, total_epoch, decay_start, decay_end)
