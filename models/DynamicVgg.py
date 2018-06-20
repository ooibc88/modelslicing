'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dynamic_conv2d import DynamicConvGN2d

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class Block(nn.Module):
    def __init__(self, inplanes, outplanes, num_groups, downsample=None):
        super(Block, self).__init__()
        self.conv = DynamicConvGN2d(num_groups, inplanes, outplanes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        # wrap input/output due to nn.Sequential
        x, keep_rate = input
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.relu(self.conv(x, keep_rate))
        return (x, keep_rate)


class DynamicVGG(nn.Module):
    def __init__(self, vgg_name, num_groups):
        super(DynamicVGG, self).__init__()
        assert (isinstance(num_groups, int) and num_groups>0)
        self.features = self._make_layers(cfg[vgg_name], num_groups)
        self.classifier = nn.Linear(512, 10)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x, keep_rate):
        out = self.features((x, keep_rate))[0]
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, num_groups):
        downsample = None
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                # downsample = nn.Sequential(
                #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                #     nn.BatchNorm2d(in_channels),
                # )
            else:
                layers += [Block(in_channels, x, num_groups, downsample)]
                in_channels = x
                downsample = None
        return nn.Sequential(*layers)


def test():
    print(__name__)
    net = DynamicVGG('VGG19', 16)
    x = torch.randn(2,3,32,32)
    y = net(x, 1.0)
    print(y.size())
    print(net)

# test()

