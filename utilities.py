import time
import math
import logging
import sys

# setup logger
def logger(log_dir, need_time=True, need_stdout=False):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y-%I:%M:%S')
    if need_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        log.addHandler(ch)
    if need_time:
        fh.setFormatter(formatter)
        if need_stdout:
            ch.setFormatter(formatter)
    log.addHandler(fh)
    return log

def timeSince(since=None, s=None):
    if s is None:
        s = int(time.time() - since)
    m = math.floor(s / 60)
    s %= 60
    h = math.floor(m / 60)
    m %= 60
    return '%dh %dm %ds' %(h, m, s)

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

