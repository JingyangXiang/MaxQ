import time
from functools import partial

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from utils.net_utils import AverageMeter, accuracy, forward_func, do_grad_decay_v1, do_grad_decay_v2

scaler = GradScaler()


def train_engine(train_loader, model, criterion, optimizer, epoch, log, print_log, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)

        # compute output
        with autocast():
            loss, output = forward_func(model, input_var, target_var, criterion)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(prec1.item(), output.size(0))
        top5.update(prec5.item(), output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # because we use amp mode so it needs to unscale grad before add weight decay
        scaler.unscale_(optimizer)
        if args.decay_type == 'v1':
            model.apply(partial(do_grad_decay_v1, decay=args.decay))
        elif args.decay_type == 'v2':
            model.apply(partial(do_grad_decay_v2, decay=args.decay))
        else:
            raise NotImplementedError

        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), log)


def validate(val_loader, model, criterion, log, print_log, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg
