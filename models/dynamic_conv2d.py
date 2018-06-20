import torch
from torch.nn import Conv2d
from torch.nn import BatchNorm2d, GroupNorm
from torch.nn import functional as F
from torch.nn import ModuleList

class DynamicConv2d(Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=bias)

    def forward(self, input, keep_rate=1.):
        assert 0.<keep_rate<=1.
        in_channels = round(keep_rate*self.in_channels)
        # out_channels = round(keep_rate*self.out_channels)

        if in_channels < self.in_channels:
            input_mask = torch.zeros((1, self.in_channels, 1, 1), device=input.device)
            input_mask[:, :in_channels] = float(self.in_channels)/float(in_channels)
            input = input * input_mask.expand_as(input)

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # if out_channels < self.out_channels:
        #     out[:, out_channels:] = 0.
        #     out *= float(self.out_channels)/out_channels
        return out

# original working version
# class DynamicConvBN2d(Conv2d):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=False):
#         self.in_channels, self.out_channels = in_channels, out_channels
#         super(DynamicConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, groups, bias=bias)
#         self.bn = BatchNorm2d(out_channels, track_running_stats=False) # running estimate not working, set to False
#
#     def forward(self, input, keep_rate=1.):
#         assert 0.<keep_rate<=1.
#         out_channels = round(keep_rate*self.out_channels)
#
#         out = F.conv2d(input, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#         out = self.bn(out)
#
#         if out_channels < self.out_channels:
#             out[:, out_channels:] = 0.
#             out *= float(self.out_channels)/out_channels
#         return out

# multiple bn working version
class DynamicConvBN2d(Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias=bias)

        self.bn_map = {round(keep_rate.item(), 1): BatchNorm2d(out_channels) for keep_rate in torch.range(0.4, 1.01, 0.1)}
        self.bn_list = ModuleList(self.bn_map.values())

    def forward(self, input, keep_rate=1.):
        assert 0. < keep_rate <= 1.

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        self.bn_key = self.round_to_value(keep_rate, self.bn_map.keys())
        out = self.bn_map[self.bn_key](out)

        out_channels = round(self.bn_key * self.out_channels) # change to rounded keep_rate = bn_key
        if out_channels < self.out_channels:
            out[:, out_channels:] = 0.
            out *= float(self.out_channels) / out_channels
        return out

    def round_to_value(self, val, vals):
        ''' return nearest value to val (float) in vals (list of float) '''
        return min(vals, key=lambda v: abs(v - val))

class DynamicConvGN2d(Conv2d):

    def __init__(self, num_groups, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConvGN2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias=bias)

        if num_groups > out_channels: num_groups = out_channels
        self.gn = GroupNorm(num_groups, out_channels)
        self.kr_list = list((1./num_groups)*cnt for cnt in range(1, num_groups+1))
        # print(self.kr_list, num_groups, in_channels, out_channels)

    def forward(self, input, keep_rate=1.):
        assert 0. < keep_rate <= 1.

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        out = self.gn(out)

        out_channels = round(self.round_to_value(keep_rate, self.kr_list)*self.out_channels)
        if out_channels < self.out_channels:
            out[:, out_channels:] = 0.
            # out *= float(self.out_channels) / out_channels
        return out

    def round_to_value(self, val, vals):
        ''' return nearest value to val (float) in vals (list of float) '''
        return min(vals, key=lambda v: abs(v - val))


