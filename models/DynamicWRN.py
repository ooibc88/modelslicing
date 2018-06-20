'''
adapted from:   https://github.com/meliketoy/wide-resnet.pytorch
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from models.dynamic_conv2d import DynamicConvGN2d

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, num_groups, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = DynamicConvGN2d(num_groups, in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = DynamicConvGN2d(num_groups, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, input):
        # wrap input/output due to nn.Sequential
        x, keep_rate = input
        out = F.relu(self.bn1(x))
        out = self.conv1(out, keep_rate)
        out = F.relu(self.dropout(out))
        out = self.conv2(out, keep_rate)

        out += self.shortcut(x)
        return (out, keep_rate)

class DynamicWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_groups, dropout_rate=0., num_classes=10):
        super(DynamicWideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, num_groups, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, num_groups, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, num_groups, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, num_groups, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, num_groups, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, keep_rate):
        out = self.conv1(x)
        out = self.layer1((out, keep_rate))[0]
        out = self.layer2((out, keep_rate))[0]
        out = self.layer3((out, keep_rate))[0]
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=DynamicWideResNet(28, 10, 8, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)), 1)

    print(y.size())
    print(net)
    print(sum([p.data.nelement() for p in net.parameters()]))