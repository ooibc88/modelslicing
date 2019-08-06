import numpy as np
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['DynamicConv2d', 'DynamicGN', 'DynamicLinear', 'DynamicBN', 'DYNAMIC_LAYERS',
           'update_sr_idx', 'bind_update_sr_idx', 'upgrade_dynamic_layers',
           'create_sr_scheduler']

class DynamicConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, sr_in_list=(1.,), sr_out_list=None):
        self.sr_idx, self.sr_in_list = 0, sorted(set(sr_in_list), reverse=True)
        if sr_out_list is not None: self.sr_out_list = sorted(set(sr_out_list), reverse=True)
        else: self.sr_out_list = self.sr_in_list
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride, padding, dilation, groups=groups, bias=bias)

    def forward(self, input):
        in_channels = round(self.in_channels*self.sr_in_list[self.sr_idx])
        out_channels = round(self.out_channels*self.sr_out_list[self.sr_idx])
        weight, bias = self.weight[:out_channels, :in_channels, :, :], None
        if self.bias is not None: bias = self.bias[:out_channels]
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation,
                round(self.groups*self.sr_in_list[self.sr_idx]) if self.groups>1 else 1)

class DynamicGN(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, sr_in_list=(1.,)):
        self.sr_idx, self.sr_in_list = 0, sorted(set(sr_in_list), reverse=True)
        super(DynamicGN, self).__init__(num_groups, num_channels, eps)

    def forward(self, input):
        num_channels = int(self.num_channels*self.sr_in_list[self.sr_idx])
        weight, bias = self.weight[:num_channels], self.bias[:num_channels]
        return F.group_norm(input, round(num_channels*self.num_groups/float(self.num_channels)),
                            weight, bias, self.eps)

class DynamicBN(nn.Module):
    def __init__(self, num_features, affine=True, track_running_stats=True, sr_in_list=(1.,)):
        super(DynamicBN, self).__init__()
        self.sr_idx, self.sr_in_list = 0, sorted(set(sr_in_list), reverse=True)
        self.bn_list = nn.Sequential(*[nn.BatchNorm2d(int(num_features * sr),
            affine=affine, track_running_stats=track_running_stats) for sr in self.sr_in_list])

    def forward(self, input):
        return self.bn_list[self.sr_idx](input)

class DynamicLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sr_in_list=(1.0,), sr_out_list=None):
        self.sr_idx, self.sr_in_list = 0, sorted(set(sr_in_list), reverse=True)
        if sr_out_list is not None: self.sr_out_list = sorted(set(sr_out_list), reverse=True)
        else: self.sr_out_list = self.sr_in_list
        super(DynamicLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        in_features = round(self.in_features*self.sr_in_list[self.sr_idx])
        out_features = round(self.out_features*self.sr_out_list[self.sr_idx])
        weight, bias = self.weight[:out_features, :in_features], None
        if self.bias is not None: bias = self.bias[:out_features]
        return F.linear(input, weight, bias)

DYNAMIC_LAYERS = (DynamicConv2d, DynamicLinear, DynamicGN, DynamicBN)

def update_sr_idx(model, idx):
    model.apply(lambda module: (setattr(module, 'sr_idx', idx)
        if hasattr(module, 'sr_idx') else None))

def bind_update_sr_idx(model):
    model.update_sr_idx = update_sr_idx.__get__(model)

def upgrade_dynamic_layers(model, num_groups=8, sr_in_list=(1.,), use_gn=True):
    sr_in_list = sorted(set(sr_in_list), reverse=True)

    def update(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, DynamicConv2d(module.in_channels, module.out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.groups, module.bias is not None, sr_in_list))
            elif isinstance(module, nn.Linear):
                setattr(model, name, DynamicLinear(module.in_features, module.out_features,
                                                    module.bias is not None, sr_in_list))
            elif isinstance(module, nn.BatchNorm2d):
                if use_gn: setattr(model, name, DynamicGN(num_groups, module.num_features, module.eps, sr_in_list))
                else: setattr(model, name, DynamicBN(module.num_features,
                                                     module.affine, module.track_running_stats, sr_in_list))

            update(module)

    # replace all conv/bn/linear layers with dynamic counterparts
    update(model)
    # bind all dynamic layers with function update_sr_idx
    bind_update_sr_idx(model)

    # get all modules, and update the 1st module's sr_in_list and last module's sr_out_list to all 1s
    modules = list(filter(lambda module: isinstance(module, DYNAMIC_LAYERS), model.modules()))
    modules[0].sr_in_list = [1. for _ in range(len(sr_in_list))]
    modules[-1].sr_out_list = [1. for _ in range(len(sr_in_list))]

    # return self to support chain operations (optional return)
    return model

def create_sr_scheduler(scheduler_type, sr_rand_num, sr_list, sr_prob=None):
    '''
    :param scheduler_type:  round_robin, random+optionally specified min/max slice rate
    :param sr_rand_num:     # of random sampled slice rate
    :param sr_list:         slice rate list
    :param sr_prob:         probabilities associated with each slice rate for random sampling
    :return:                a list of slice rate for the current training batch
    '''
    idx_num = len(sr_list)
    min_max_idxs, candidate_idxs = [], list(range(idx_num))
    if sr_prob: sr_prob=np.array(sr_prob)/sum(sr_prob)

    if scheduler_type.find('max') >= 0:
        candidate_idxs.remove(0)
        min_max_idxs.append(0)
    if scheduler_type.find('min') >= 0:
        candidate_idxs.remove(idx_num-1)
        min_max_idxs.append(idx_num-1)

    while True:
        if scheduler_type.startswith('random'):
            rand_idxs = np.random.choice(candidate_idxs, size=sr_rand_num, p=sr_prob, replace=False)
            yield sorted(rand_idxs.tolist()+min_max_idxs)
        elif scheduler_type == 'round_robin':
            yield candidate_idxs
        else:
            raise Exception('unknown scheduler type: {}'.format(scheduler_type))
