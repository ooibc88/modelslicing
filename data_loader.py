import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data_loader(args):
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir, train=True, download=False, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            class_num = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir, train=True, download=False, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            class_num = 10
        else:
            raise Exception ('unknown dataset: {}'.format(args.dataset))
        return train_loader, val_loader, class_num

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        class_num = 1000
        return train_loader, val_loader, class_num, train_sampler

    else:
        raise Exception ('unknown dataset: {}'.format(args.dataset))