import torch
from torch import nn
from functools import reduce
import operator
import time

MULTI_ADD = 1
# global variables
total_flops = 0
total_params = 0
total_time = 0
module_cnt = 0
verbose = False
# output format control
name_space = 95
param_space = 18
flop_space = 18
time_space = 18
# forward times
forward_num = 10

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose: print('Elapsed time: %f ms.' % self.time)

def calc_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

def profile_forward(layer, input):
    with Timer() as t:
        for _ in range(forward_num):
            layer.old_forward(input)
            if torch.cuda.is_available(): torch.cuda.synchronize()
    return int(t.time * 1e9 / forward_num)

def profile_layer(layer, x):
    global total_flops, total_params, total_time, module_cnt, verbose, MULTI_ADD
    delta_ops, delta_params, delta_time = 0, 0, 0.

    # Conv2d
    if isinstance(layer, nn.Conv2d):
        out_h = int((x.size(2) + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size(3) + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * MULTI_ADD
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)
        module_cnt += 1

    # Linear
    elif isinstance(layer, nn.Linear):
        weight_ops = layer.weight.numel() * MULTI_ADD
        bias_ops = layer.bias.numel()
        delta_ops = (weight_ops + bias_ops)
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)
        module_cnt += 1

    # ReLU can be omited
    elif isinstance(layer, nn.ReLU):
        delta_ops = reduce(operator.mul, x.size()[1:])
        delta_time = profile_forward(layer, x)
        module_cnt += 1

    # Pool2d
    elif type(layer) in [nn.AvgPool2d, nn.MaxPool2d]:
        in_w = x.size(2)
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size(1) * out_w * out_h * kernel_ops
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)
        module_cnt += 1

    # AdaptiveAvgPool2d
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        delta_ops = x.size(1) * x.size(2) * x.size(3)
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)

    # BatchNorm2d, GroupNorm
    elif type(layer) in [nn.BatchNorm2d, nn.GroupNorm]:
        delta_ops = x.size(1) * x.size(2) * x.size(3)
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)

    # ops ignore flops
    elif type(layer) in [nn.Dropout2d, nn.Dropout]:
        delta_params = calc_layer_param(layer)
        delta_time = profile_forward(layer, x)

    # ignore layer type
    elif type(layer) in [nn.Sequential]: # nn.BatchNorm2d, nn.GroupNorm, nn.ReLU
        return
    else:
        raise TypeError('unknown layer type: %s' % str(layer))

    total_flops += delta_ops
    total_params += delta_params
    total_time += delta_time
    if verbose:
        print(str(layer).ljust(name_space, ' ') +
            '{:,}'.format(delta_params).rjust(param_space, ' ') +
            '{:,}'.format(delta_ops).rjust(flop_space, ' ') +
            '{:,}'.format(delta_time).rjust(time_space, ' '))
    return


def profiling(model, H, W, C=3, B=1, debug=False):
    global total_flops, total_params, total_time, module_cnt, verbose
    total_flops, total_params, module_cnt, verbose = 0, 0, 0, debug
    data = torch.zeros(B, C, H, W)

    def is_leaf(model):
        ''' measure all leaf nodes '''
        return len(list(model.children())) == 0

    def modify_forward(model):
        for child in model.children():
            if is_leaf(child):
                def new_forward(m):
                    def lambda_forward(x):
                        profile_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    def line_breaker():
        print(''.center(name_space + param_space +
                        flop_space + time_space, '-'))

    print('Item'.ljust(name_space, ' ') +
        'params'.rjust(param_space, ' ') +
        'flops'.rjust(flop_space, ' ') +
        'nanosecs'.rjust(time_space, ' '))
    if verbose: line_breaker()
    modify_forward(model)
    model.forward(data)
    restore_forward(model)
    if verbose:
        line_breaker()
        print('Total'.ljust(name_space, ' ') +
            '{:,}'.format(total_params).rjust(param_space, ' ') +
            '{:,}'.format(total_flops).rjust(flop_space, ' ') +
            '{:,}'.format(total_time).rjust(time_space, ' '))

    return total_params, total_flops, total_time
