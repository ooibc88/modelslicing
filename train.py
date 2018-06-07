import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.distributions import normal

from models.resnet import ResNet
from models.preresnet import PreResNet
from models.DynamicPreResnet import DynamicPreResNet
from models.PyramidNet import PyramidNet
from models.DynamicPyramidNet import DynamicPyramidNet
from data_loader import data_loader
from tensorboard_logger import configure, log_value
from utilities import logger, AverageMeter, timeSince

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--exp_name', default='', type=str, help='optional exp name used to store log and checkpoint (default: none)')
parser.add_argument('--net_type', default='pyramidnet', type=str, help='networktype: resnet, resnext, densenet, pyamidnet, and so on')
parser.add_argument('--dynamic', dest='dynamic', action='store_true', help='whether to use dynamic training (default: False)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--alpha', default=300, type=int, help='number of new channel increases per depth (default: 300)')
parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
parser.add_argument('--lower_bound', default=0.4, type=float, help='lower bound keep rate drawn from distribution')

parser.add_argument('--epoch', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model on ImageNet-1k dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_best', dest='resume_best', action='store_true', help='whether to resume best_checkpoint (default: False)')
parser.add_argument('--checkpoint-dir', default='/home/shaofeng/ncrs-hdd1/checkpoint/', type=str, metavar='PATH', help='path to checkpoint')

parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='file:///home/shaofeng/share_test', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')

parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--data-dir', default='./data/', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str, help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation for CIFAR datasets (default: True)')

parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='to print the status at every iteration')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')

parser.set_defaults(dynamic=False)
parser.set_defaults(resume_best=False)
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(verbose=True)
# initialize all global variables
args = parser.parse_args()
args.data_dir += args.dataset

if not args.exp_name:
    args.exp_name = '{0}_{1}_{2}'.format(args.net_type, args.depth, args.dataset)
    if args.dynamic: args.exp_name = 'dynamic_{0}_lb_{1}'.format(args.exp_name, args.lower_bound)
    if args.net_type=='pyramidnet': args.exp_name+='_alpha_{0}'.format(args.alpha)
args.checkpoint_dir = '{0}{1}/'.format(args.checkpoint_dir, args.exp_name)

args.distributed = (args.world_size > 1)
best_err1, best_err5 = 100., 100.
if not os.path.isdir('log/{}'.format(args.exp_name)):
    os.mkdir('log/{}'.format(args.exp_name))

def main():
    global args, best_err1, best_err5
    print_logger = logger('log/{}/stdout.log'.format(args.exp_name), False, False)
    print_logger.info(vars(args))

    if args.tensorboard: configure("runs/%s" % (args.exp_name))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size,
                                init_method=args.dist_url, group_name=args.exp_name)
        train_loader, val_loader, class_num, train_sampler = data_loader(args)
    else:
        train_loader, val_loader, class_num = data_loader(args)
    
    if args.pretrained:
        print_logger.info("=> using pre-trained model '{}'".format(args.net_type))
        try:
            model = models.__dict__[str(args.net_type)](pretrained=True)
        except (KeyError, TypeError):
            print_logger.info('unknown model! \n torchvision provides the follwoing pretrained model:', model_names)
            return
    else:
        print_logger.info("=> creating model '{}'".format(args.net_type))
        if args.net_type == 'resnet':
            model = ResNet(args.dataset, args.depth, class_num, args.bottleneck) # for ResNet
        elif args.net_type == 'preresnet':
            if args.dynamic: model = DynamicPreResNet(args.dataset, args.depth, class_num, args.bottleneck) # for Dynamic Pre-activation ResNet
            else: model = PreResNet(args.dataset, args.depth, class_num, args.bottleneck) # for Pre-activation ResNet
        elif args.net_type == 'pyramidnet':
            if args.dynamic: model = DynamicPyramidNet(args.dataset, args.depth, args.alpha, class_num, args.bottleneck) # for Dynamic PyramidNet
            else: model = PyramidNet(args.dataset, args.depth, args.alpha, class_num, args.bottleneck) # for PyramidNet
        else:
            raise Exception ('unknown network architecture: {}'.format(args.net_type))

    print_logger.info('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.distributed: model = torch.nn.parallel.DistributedDataParallel(model).cuda()
    else: model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.epoch * 0.5), int(args.epoch * 0.75)], gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        print_logger.info("=> loading checkpoint '{}'".format(args.resume))

        if os.path.isfile(args.resume): checkpoint = torch.load(args.resume)
        elif args.resume == 'checkpoint': checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir, 'checkpoint.ckpt'))
        else: print_logger.info("=> no checkpoint found at '{}'".format(args.resume)); return

        args.start_epoch = checkpoint['epoch'] + 1  # start epoch += 1
        model.load_state_dict(checkpoint['state_dict'])
        best_err1 = checkpoint['best_err1']
        best_err5 = checkpoint['best_err5']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print_logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # used if args.dynamic:
    distributions = normal.Normal(1., 1. / 3.)
    for epoch in range(args.start_epoch, args.epoch):
        if args.distributed: train_sampler.set_epoch(epoch)
        scheduler.step(epoch)
        print_logger.info('Epoch: [{0}/{1}]\tLR: {LR:.6f}'.format(epoch, args.epoch, LR=scheduler.get_lr()[0]))
            
        # train for one epoch
        run(epoch, model, train_loader, criterion, print_logger, optimizer=optimizer, dist=distributions)
        
        # evaluate on validation set
        err1, err5 = run(epoch, model, val_loader, criterion, print_logger)
        
        # record best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best: best_err5 = err5
        print_logger.info('Current best accuracy:\ttop1 = {top1:.4f} | top5 = {top5:.4f}'.format(top1=best_err1, top5=best_err5))
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, file_dir=args.checkpoint_dir)
    print_logger.info('Best accuracy:\ttop1 = {top1:.4f} | top5 = {top5:.4f}'.format(top1=best_err1, top5=best_err5))

def save_checkpoint(checkpoint, is_best, file_dir, file_name='checkpoint.ckpt'):
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    ckpt_name = "{0}{1}".format(file_dir, file_name)
    torch.save(checkpoint, ckpt_name)
    if is_best: shutil.copyfile(ckpt_name, "{0}{1}".format(file_dir, 'best_'+file_name))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size).item())

    return res

def draw_keep_rate(dist, progress=1., lower_bound=args.lower_bound):
    keep_rate = dist.sample().item()
    return max(lower_bound, min(1., keep_rate))

def run(epoch, model, data_loader, criterion, print_logger, optimizer=None, keep_rate=1., dist=None):
    is_train = True if optimizer!=None else False
    if is_train: model.train()
    else: model.eval()

    batch_time_avg = AverageMeter()
    loss_avg, top1_avg, top5_avg = AverageMeter(), AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, (input, target) in enumerate(data_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if is_train:
            if args.dynamic:
                keep_rate = draw_keep_rate(dist, progress=float(idx)/len(data_loader))
                output = model(input, keep_rate)
            else:
                output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                if args.dynamic: output = model(input, keep_rate)
                else: output = model(input)
                loss = criterion(output, target)

        err1, err5 = accuracy(output, target, topk=(1,5))
        loss_avg.update(loss.item(), input.size()[0])
        top1_avg.update(err1, input.size()[0])
        top5_avg.update(err5, input.size()[0])

        batch_time_avg.update(time.time()-timestamp)
        timestamp = time.time()

        if args.verbose == True:
            print_logger.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  '{4}Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epoch, idx, len(data_loader), ('keep_rate {0:.4f}\t'.format(keep_rate) if args.dynamic else ''),
                batch_time=batch_time_avg, loss=loss_avg, top1=top1_avg, top5=top5_avg))

    print_logger.info('* Epoch: [{0}/{1}]{2:>6s}  Total Time: {3}\tTop 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\tTest Loss {loss.avg:.3f}'.format(
        epoch, args.epoch, ('train' if optimizer is not None else 'val'), timeSince(s=batch_time_avg.sum), top1=top1_avg, top5=top5_avg, loss=loss_avg))
    if args.tensorboard:
        log_value('train_loss', loss_avg.avg, epoch)
        log_value('train_error', top1_avg.avg, epoch)
    return top1_avg.avg, top5_avg.avg

if __name__ == '__main__':
    main()