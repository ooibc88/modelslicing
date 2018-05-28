
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import functional as F

class DynamicConv2d(Conv2d):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=bias)
        # self.register_buffer('input_mask', torch.zeros((1, in_channels, 1, 1)))

    def forward(self, input, keep_rate=1.):
        assert 0.<keep_rate<=1.
        # in_channels = round(keep_rate*self.in_channels)
        out_channels = round(keep_rate*self.out_channels)

        # if in_channels < self.in_channels:
            # self.input_mask.fill_(0.)[:, :in_channels] = float(self.in_channels)/float(in_channels)
            # input_mask = torch.zeros((1, self.in_channels, 1, 1), device=input.device)
            # input_mask[:, :in_channels] = float(self.in_channels)/float(in_channels)
            # input = input * input_mask.expand_as(input)

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if out_channels < self.out_channels:
            out[:, out_channels:] = 0.
            out *= float(self.out_channels)/out_channels
        return out

class DynamicConvBN2d(Conv2d):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=bias)
        self.bn = BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, input, keep_rate=1.):
        assert 0.<keep_rate<=1.
        # in_channels = round(keep_rate*self.in_channels)
        out_channels = round(keep_rate*self.out_channels)

        # if in_channels < self.in_channels:
            # self.input_mask.fill_(0.)[:, :in_channels] = float(self.in_channels)/float(in_channels)
            # input_mask = torch.zeros((1, self.in_channels, 1, 1), device=input.device)
            # input_mask[:, :in_channels] = float(self.in_channels)/float(in_channels)
            # input = input * input_mask.expand_as(input)

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        out = self.bn(out)

        if out_channels < self.out_channels:
            out[:, out_channels:] = 0.
            out *= float(self.out_channels)/out_channels
        return out

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

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, keep_rate=1.):
        assert 0. < keep_rate <= 1.
        self._check_input_dim(input)
        # carve out input, running_mearn/var,
        in_channels = round(keep_rate*self.num_features)
        if in_channels < self.num_features:
            return F.batch_norm(
                input[:, :in_channels], self.running_mean[:in_channels], self.running_var[:in_channels],
                self.weight[:in_channels], self.bias[:in_channels],
                self.training or not self.track_running_stats, self.momentum, self.eps)

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

def test():
    import torch
    torch.manual_seed(0)
    conv2d = DynamicConv2d(4, 5, 3)
    print(conv2d.bias); exit()
    x = torch.randn((1, 4, 5, 5))
    print(x)
    # print(conv2d(x))
    output = conv2d(x, 0.4)
    print(output)
    print(output.size())
    loss = torch.sum(output)**2
    loss.backward()
    print(conv2d.weight.grad)


# test()
