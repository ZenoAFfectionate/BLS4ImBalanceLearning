import os
import sys
import time
import math
import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.cuda.amp import autocast as autocast
from torch.optim.lr_scheduler import _LRScheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class CosineAnnealingLRWarmup(_LRScheduler):
    ''' Cosine Annealing with Warm Up Learning Rate Scheduler'''

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=10, base_lr=0., warmup_lr=0.001):
        if T_max <= warmup_epochs:
            raise ValueError("T_max should be larger than warmup_epochs.")
        if warmup_epochs <= 1:
            raise ValueError("warmup_epochs should be larger than 1.")
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        for group in optimizer.param_groups: group['lr'] = base_lr
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch, verbose=True)

    def get_cos_lr(self):
        t = self.last_epoch - self.warmup_epochs
        T_remaining = self.T_max - self.warmup_epochs
        cos_factor = (1 + math.cos(math.pi * t / T_remaining)) / 2
        return [self.eta_min + (self.warmup_lr - self.eta_min) * cos_factor
                for _ in self.base_lrs]

    def get_warmup_lr(self):
        ratio = (self.last_epoch + 1) / self.warmup_epochs
        return [self.base_lr + (self.warmup_lr -self.base_lr) * ratio 
                for _ in self.base_lrs]

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_lr()
        else:
            return self.get_cos_lr()


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, extra_info=None):
    # return lam * criterion(pred, y_a, extra_info) + (1 - lam) * criterion(pred, y_b, extra_info)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, device, train_loader, scaler, criterion, optimizer, epoch, mixup=True):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1',  ':4.2f')
    top5 = AverageMeter('Acc@5',  ':4.2f')

    print('\nEpoch: %d' % epoch)
    model.train()
    for batch_idx, (images, target) in enumerate(train_loader):
        # forward the model and compute the loss
        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        batch_size = target.size(0)

        with autocast():
            if mixup:
                images, targets_a, targets_b, lam = mixup_data(images, target, alpha=1.0)
                output = model(images)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # backward the model and update the parameters
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # visualize the training process
        progress_info = (f'Loss: {losses.avg:.3f} | ' + 
                         f'Acc@1: {top1.avg:.2f}% | ' + 
                         f'Acc@5: {top5.avg:.2f}% ')
        progress_bar(batch_idx, len(train_loader), progress_info)

    return losses.avg


def valid(model, device, valid_loader, criterion):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1',  ':4.2f')
    top5 = AverageMeter('Acc@5',  ':4.2f')
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(valid_loader):
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)

            batch_size = target.size(0)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # visualize the validation process
            progress_info = (f'Loss: {losses.avg:.3f} | ' + 
                             f'Acc@1: {top1.avg:.2f}% | ' + 
                             f'Acc@5: {top5.avg:.2f}% ')
            progress_bar(batch_idx, len(valid_loader), progress_info)

    return losses.avg, top1.avg


def extract_features(model, train_loader, valid_loader, device):
    model.eval()  # set model to evaluation mode

    train_features, train_labels = [], []
    valid_features, valid_labels = [], []

    with torch.no_grad():
        # extract features from training data
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # move to device
            features = model(inputs)    # extract features
            train_features.append(features)
            train_labels.append(targets)

        # extract features from evaluation data
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(device)  # move to device
            features = model(inputs)    # extract features
            valid_features.append(features)
            valid_labels.append(targets)

    # merge all batches
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    valid_features = torch.cat(valid_features, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    # create new dataset based on the features and the labels
    train_feature_dataset = TensorDataset(train_features, train_labels)
    valid_feature_dataset = TensorDataset(valid_features, valid_labels)

    return train_feature_dataset, valid_feature_dataset


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if current == 0: begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg: L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
