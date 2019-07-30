import torch.nn as nn

cfg = {
    'VGG11': [64, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'VGG13': [64, 64, 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'VGG16': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
    'VGG19': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512],
}

class CifarVGG(nn.Module):
    def __init__(self, depth, num_classes=10, widen_factor=1.0):
        super(CifarVGG, self).__init__()
        self.widen_factor = widen_factor
        self.features = self._make_layers(cfg['VGG{0}'.format(depth)])
        self.classifier = nn.Linear(int(512*widen_factor), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(x*self.widen_factor)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=8, stride=1)]
        return nn.Sequential(*layers)

def cifar_vgg(args):
    return CifarVGG(args.depth, args.class_num, args.arg1)
