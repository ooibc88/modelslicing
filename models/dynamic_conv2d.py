import torch
from torch.nn import Conv2d
from torch.nn import functional as F
import time

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
                 padding=0, dilation=1, groups=1, bias=True):
        self.in_channels, self.out_channels = in_channels, out_channels
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=bias)

    def forward(self, input, keep_rate=1., restore_out=True):
        assert 0.<keep_rate<=1.
        self.keep_rate = keep_rate
        in_channels, out_channels = round(keep_rate*self.in_channels), round(keep_rate*self.out_channels)
        if out_channels < self.out_channels:
            if in_channels < input.size()[1]:
                input = input[:, :in_channels]

            current_weight = self.weight[:out_channels, :in_channels]
            current_bias = self.bias[:out_channels]

            output = F.conv2d(input, current_weight, current_bias, self.stride,
                            self.padding, self.dilation, self.groups)

            if restore_out:
                zero_padding = torch.zeros((output.size()[0], self.out_channels-out_channels, output.size()[2], output.size()[3]), device=output.device)
                return torch.cat((output, zero_padding), dim=1)
            else: return output
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

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