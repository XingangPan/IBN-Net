import math
import warnings

import torch
import torch.nn as nn

from .modules import IBN


__all__ = ['resnext50_ibn_a', 'resnext101_ibn_a', 'resnext152_ibn_a']


model_urls = {
    'resnext101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth',
}


class Bottleneck_IBN(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, ibn=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck_IBN, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(D*C)
        else:
            self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt_IBN(nn.Module):

    def __init__(self,
                 baseWidth,
                 cardinality,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        super(ResNeXt_IBN, self).__init__()
        block = Bottleneck_IBN

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.conv1.weight.data.normal_(0, math.sqrt(2. / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth,
                            self.cardinality, stride, downsample, ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth,
                                self.cardinality, 1, None, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    """
    Construct ResNeXt-50-IBN-a.
    """
    model = ResNeXt_IBN(baseWidth, cardinality, [3, 4, 6, 3], ('a', 'a', 'a', None))
    if pretrained:
        warnings.warn("Pretrained model not available for ResNeXt-50-IBN-a!")
    return model


def resnext101_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    """
    Construct ResNeXt-101-IBN-a.
    """
    model = ResNeXt_IBN(baseWidth, cardinality, [3, 4, 23, 3], ('a', 'a', 'a', None))
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnext101_ibn_a']))
    return model


def resnext152_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    """
    Construct ResNeXt-152-IBN-a.
    """
    model = ResNeXt_IBN(baseWidth, cardinality, [3, 8, 36, 3], ('a', 'a', 'a', None))
    if pretrained:
        warnings.warn("Pretrained model not available for ResNeXt-152-IBN-a!")
    return model
