from train import *

def main():
    global args, best_err1, best_err5
    print_logger = logger('log/{}/validate.log'.format(args.exp_name), False, False)
    print_logger.info(vars(args))

    # loading date
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size,
                                init_method=args.dist_url, group_name=args.exp_name)
        train_loader, val_loader, class_num, train_sampler = data_loader(args)
    else:
        train_loader, val_loader, class_num = data_loader(args)

    # create model
    print_logger.info("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = ResNet(args.dataset, args.depth, class_num, args.bottleneck)  # for ResNet
    elif args.net_type == 'preresnet':
        if args.dynamic: model = DynamicPreResNet(args.dataset, args.depth, class_num, args.bottleneck)  # for Dynamic Pre-activation ResNet
        else: model = PreResNet(args.dataset, args.depth, class_num, args.bottleneck)  # for Pre-activation ResNet
    elif args.net_type == 'pyramidnet':
        if args.dynamic: model = DynamicPyramidNet(args.dataset, args.depth, args.alpha, class_num, args.bottleneck)  # for Dynamic PyramidNet
        else: model = PyramidNet(args.dataset, args.depth, args.alpha, class_num, args.bottleneck)  # for PyramidNet
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    print_logger.info('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.distributed: model = torch.nn.parallel.DistributedDataParallel(model).cuda()
    else: model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # loading model from checkpoint
    print_logger.info("=> loading checkpoint '{}'".format(args.resume))
    if os.path.isfile(args.resume): checkpoint = torch.load(args.resume)
    elif args.resume == 'checkpoint': checkpoint = torch.load('{0}{1}'.format(args.checkpoint_dir, 'checkpoint.ckpt'))
    else: print_logger.info("=> no checkpoint found at '{}'".format(args.resume)); return

    model.load_state_dict(checkpoint['state_dict'])
    best_err1 = checkpoint['best_err1']
    best_err5 = checkpoint['best_err5']
    print_logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    args.dynamic = False
    for keep_rate in torch.range(1., 0.1, -0.05):
        err1_train, err5_train = run(checkpoint['epoch'], model, train_loader, criterion, print_logger, keep_rate=keep_rate)
        err1_val, err5_val = run(checkpoint['epoch'], model, val_loader, criterion, print_logger, keep_rate=keep_rate)
        print_logger.info('> Train Set Top 1-err {top1_train.avg:.3f}  Top 5-err {top5_train.avg:.3f}\n'
                          'Validation Set Top 1-err {top1_val.avg:.3f}  Top 5-err {top5_val.avg:.3f}'.format(
            top1_train=err1_train, top5_train=err5_train, top1_val=err1_val, top5_val=err5_val))


if __name__ == '__main__':
    main()