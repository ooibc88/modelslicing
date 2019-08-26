from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        warmup_epoch: target learning rate is linearly reached at the warmup_epoch
        scheduler: scheduler used after warmup_epoch (eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, warmup_epoch, multiplier=1.0, scheduler=None):
        assert multiplier > 1., 'multiplier should be greater than 1.'
        self.multiplier = multiplier
        self.warmup_epoch = warmup_epoch
        self.scheduler = scheduler
        self.finish_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            if self.scheduler:
                if not self.finish_warmup:
                    self.scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finish_warmup = True
                return self.scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr*((self.multiplier-1.)*self.last_epoch/self.warmup_epoch+1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finish_warmup and self.scheduler:
            if epoch is None:
                self.scheduler.step(None)
            else:
                self.scheduler.step(epoch - self.warmup_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

if __name__ == '__main__':
    import torch
    v = torch.zeros(10, requires_grad=True)
    optim = torch.optim.SGD([v], lr=0.01)

    scheduler = CosineAnnealingLR(optim, 95)
    scheduler = GradualWarmupScheduler(optim, multiplier=10, warmup_epoch=5, scheduler=scheduler)

    for epoch in range(0, 100):
        scheduler.step(epoch)
        print(epoch, optim.param_groups[0]['lr'])

