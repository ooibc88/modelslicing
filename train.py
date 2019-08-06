import os
import argparse
import time
import shutil
from collections import OrderedDict
import importlib

import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from data_loader import data_loader
from utils.utilities import logger, AverageMeter, accuracy, timeSince
from models import upgrade_dynamic_layers, create_sr_scheduler

parser = argparse.ArgumentParser(description='CIFAR-10, CIFAR-100 and ImageNet-1k Model Slicing Training')
parser.add_argument('--exp_name', default='', type=str, help='optional exp name used to store log and checkpoint (default: none)')
parser.add_argument('--net_type', default='resnet', type=str, help='network type: vgg, resnet, and so on')
parser.add_argument('--groups', default=8, type=int, help='group num for Group Normalization (default: 0, no use)')
parser.add_argument('--depth', default=50, type=int, help='depth of the network')
parser.add_argument('--arg1', default=1.0, type=float, metavar='M', help='additional model arg, k for ResNet')

parser.add_argument('--sr_list', nargs='+', help='the slice rate list in descending order', required=True)
parser.add_argument('--sr_train_prob', nargs='+', help='the prob of picking subnet corresponding to sr_list')
parser.add_argument('--sr_scheduler_type', default='random', type=str, help='slice rate scheduler, support random, random[_min][_max], round_robin')
parser.add_argument('--sr_rand_num', default=1, type=int, metavar='N', help='the number of random sampled slice rate except min/max (default: 1)')
parser.add_argument('--no_gn', dest='use_gn', action='store_false', help='use GroupNorm or MultiBN (default: True)')

parser.add_argument('--epoch', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--cosine', dest='cosine', action='store_true', help='cosine LR scheduler')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_best', dest='resume_best', action='store_true', help='whether to resume the best_checkpoint (default: False)')
parser.add_argument('--checkpoint_dir', default='~/checkpoint/', type=str, metavar='PATH', help='path to checkpoint')

parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir', default='./data/', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--log_dir', default='./log/', type=str, metavar='PATH', help='path to log')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str, help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no_augment', dest='augment', action='store_false', help='whether to use standard augmentation for the datasets (default: True)')

parser.add_argument('--log_freq', default=10, type=int, metavar='N', help='log frequency')

parser.set_defaults(use_gn=True)
parser.set_defaults(cosine=False)
parser.set_defaults(resume_best=False)
parser.set_defaults(augment=True)

# initialize all global variables
args = parser.parse_args()
args.data_dir += args.dataset
args.sr_list = list(map(float, args.sr_list))
if args.sr_train_prob: args.sr_train_prob = list(map(float, args.sr_train_prob))
if not args.exp_name: args.exp_name = '{0}_{1}_{2}'.format(args.net_type, args.depth, args.dataset)
args.checkpoint_dir = '{0}{1}/'.format(os.path.expanduser(args.checkpoint_dir), args.exp_name)
args.log_path = '{0}{1}/log.txt'.format(args.log_dir, args.exp_name)
best_err1, best_err5 = 100., 100.

# create log dir
if not os.path.isdir('log/{}'.format(args.exp_name)):
    os.mkdir('log/{}'.format(args.exp_name))

# load dataset
train_loader, val_loader, args.class_num = data_loader(args)

def main():
    global args, best_err1, best_err5
    print_logger = logger(args.log_path, True, True)
    print_logger.info(vars(args))

    # create model and upgrade model to support model slicing
    model = create_model(args, print_logger)
    model = upgrade_dynamic_layers(model, args.groups, args.sr_list, args.use_gn)
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = create_lr_scheduler(args, optimizer, print_logger)

    if args.resume:
        checkpoint = load_checkpoint(print_logger)
        epoch, best_err1, best_err5, model_state, optimizer_state, scheduler_state = checkpoint.values()
        args.start_epoch = epoch+1
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(optimizer_state)
        print_logger.info("==> finish loading checkpoint '{}' (epoch {})".format(args.resume, epoch))

    cudnn.benchmark = True

    # start training
    sr_scheduler = create_sr_scheduler(args.sr_scheduler_type, args.sr_rand_num, args.sr_list, args.sr_train_prob)
    for epoch in range(args.start_epoch, args.epoch):
        scheduler.step(epoch)
        print_logger.info('Epoch: [{0}/{1}]\tLR: {LR:.6f}'.format(epoch, args.epoch, LR=scheduler.get_lr()[0]))

        # train one epoch
        run(epoch, model, train_loader, criterion, print_logger, sr_scheduler, optimizer)

        # evaluate on all the sr_idxs, from the smallest subnet to the largest
        for sr_idx in reversed(range(len(args.sr_list))):
            args.sr_idx = sr_idx
            model.module.update_sr_idx(sr_idx)
            err1, err5 = run(epoch, model, val_loader, criterion, print_logger)

        # record the best prec@1 for the largest subnet and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best: best_err5 = err5
        print_logger.info('Current best accuracy:\ttop1 = {top1:.4f} | top5 = {top5:.4f}'.format(top1=best_err1, top5=best_err5))
        save_checkpoint(OrderedDict([('epoch', epoch), ('best_err1', best_err1), ('best_err5', best_err5),
            ('model_state', model.state_dict()), ('optimizer_state', optimizer.state_dict()), ('scheduler_state', scheduler.state_dict())]),
            is_best, args.checkpoint_dir)

def create_model(args, print_logger):
    print_logger.info("==> creating model '{}'".format(args.net_type))
    models = importlib.import_module('models')
    if args.dataset.startswith('cifar'): model = getattr(models, 'cifar_{0}'.format(args.net_type))(args)
    elif args.dataset == 'imagenet': model = getattr(models, 'imagenet_{0}'.format(args.net_type))(args)
    print_logger.info('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model

def create_lr_scheduler(args, optimizer, print_logger):
    if args.cosine: scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.dataset.startswith('cifar'): scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    elif args.dataset == 'imagenet': scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.epoch/3.), int(args.epoch*2/3.)], gamma=0.1)
    else: raise Exception('unknown scheduler for dataset: {}'.format(args.dataset))
    return scheduler

def load_checkpoint(print_logger):
    print_logger.info("==> loading checkpoint '{}'".format(args.resume))

    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
    elif args.resume == 'checkpoint':
        if args.resume_best: checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir, 'best_checkpoint.ckpt'))
        else: checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir, 'checkpoint.ckpt'))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.resume))
    return checkpoint

def save_checkpoint(checkpoint, is_best, checkpoint_dir, checkpoint_name='checkpoint.ckpt'):
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    ckpt_name = "{0}{1}".format(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, ckpt_name)
    if is_best: shutil.copyfile(ckpt_name, "{0}{1}".format(checkpoint_dir, 'best_' + checkpoint_name))

def run(epoch, model, data_loader, criterion, print_logger, sr_scheduler=None, optimizer=None):
    global args
    is_train = True if optimizer!=None else False
    if is_train: model.train()
    else: model.eval()

    batch_time_avg = AverageMeter()
    loss_avg, top1_avg, top5_avg = AverageMeter(), AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, (input, target) in enumerate(data_loader):
        # print('start batch training', time.time())
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        # print('loaded data to cuda', time.time())
        if is_train:
            optimizer.zero_grad()

            for args.sr_idx in next(sr_scheduler):
                # update slice rate idx
                model.module.update_sr_idx(args.sr_idx) # DataParallel .module

                output = model(input)
                loss = criterion(output, target)
                loss.backward()

            optimizer.step()
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
        # print('finnish batch training', time.time())
        err1, err5 = accuracy(output, target, topk=(1,5))
        loss_avg.update(loss.item(), input.size()[0])
        top1_avg.update(err1, input.size()[0])
        top5_avg.update(err5, input.size()[0])

        batch_time_avg.update(time.time()-timestamp);timestamp = time.time()

        # print('start logging', time.time())
        if idx % args.log_freq == 0:
            print_logger.info('Epoch: [{0}/{1}][{2}/{3}][SR-{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\tTop 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epoch, idx, len(data_loader), args.sr_list[args.sr_idx], batch_time=batch_time_avg, loss=loss_avg, top1=top1_avg, top5=top5_avg))

    print_logger.info('* Epoch: [{0}/{1}]{2:>8s}  Total Time: {3}\tTop 1-err {top1.avg:.4f}  Top 5-err {top5.avg:.4f}\tTest Loss {loss.avg:.4f}'.format(
        epoch, args.epoch, ('[train]' if is_train else '[val]'), timeSince(s=batch_time_avg.sum), top1=top1_avg, top5=top5_avg, loss=loss_avg))
    return top1_avg.avg, top5_avg.avg


if __name__ == '__main__':
    main()