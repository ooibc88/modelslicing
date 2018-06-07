import torch
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
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

class DynamicBatchNorm2d(BatchNorm2d):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(DynamicBatchNorm2d, self).__init__(num_features, eps, momentum, affine,
                 track_running_stats)

        # if self.track_running_stats:
        #     self.keep_rate = 1.0
        #     self.register_buffer('cur_running_mean', torch.tensor(self.running_mean))
        #     self.register_buffer('cur_running_var', torch.tensor(self.running_var))
        #
        #     def hook(module, grad_input, grad_output):# -> Tensor or None
        #         num_channels = round(module.keep_rate*module.num_features)
        #         # module.running_mean[num_channels:] = module.cur_running_mean[num_channels:]
        #         # module.running_var[num_channels:] = module.cur_running_var[num_channels:]
        #         print('in hook')
        #         print(module.running_mean)
        #         print(module.cur_running_mean)
        #         module.cur_running_mean[:2] = 0.
        #     self.register_backward_hook(hook)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, keep_rate=1.):
        assert 0. < keep_rate <= 1.
        self._check_input_dim(input)

        num_channels = round(keep_rate*self.num_features)
        if num_channels < self.num_features:
            # self.keep_rate = keep_rate
            # self.cur_running_mean = torch.tensor(self.running_mean)
            # self.cur_running_var = torch.tensor(self.running_var)

            forward_input = input[:, :num_channels].contiguous()
            out =  F.batch_norm(
                forward_input, self.running_mean[:num_channels], self.running_var[:num_channels],
                self.weight[:num_channels], self.bias[:num_channels],
                self.training or not self.track_running_stats, self.momentum, self.eps)

            padding = torch.zeros((input.size(0), self.num_features-num_channels, input.size(2), input.size(3)), device=out.device)
            out = torch.cat((out, padding), 1)
            return out

        out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

        # if num_channels < self.num_features:
        #     out[:, num_channels:] = 0.

        return out


# conv_bn = DynamicConvBN2d(4, 5, 3)
# input = torch.randn((4, 4, 5, 5))
# # input[:, 2:].fill_(0.)
# input.requires_grad = True
#
#
# print('id', id(input))
# print(input)
# output = conv_bn(input, 0.5)
# print(output)
#
# output = torch.sum(output)**2
# output.backward()
#
# print(input.grad)
# print(conv_bn.bn_key)
# print(conv_bn.bn_map[conv_bn.bn_key])
# print(conv_bn.bn_map[conv_bn.bn_key].weight)
# print(conv_bn.bn_map[conv_bn.bn_key].weight.grad)
