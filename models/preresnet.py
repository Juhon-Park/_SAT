import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['PreResNet', 'preresnet20', 'preresnet32', 'preresnet44', 'preresnet56',
           'preresnet110', 'preresnet1202', 'wideresnet40']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv0 = conv3x3(inplanes, planes, stride)
        self.act0 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(planes, planes)
        self.act1 = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = self.act0(out)
        out = self.conv0(out)

        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(PreResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn1 = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(64*block.expansion, 64*block.expansion, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*block.expansion)
        self.fc2 = nn.Conv2d(64*block.expansion, num_classes, kernel_size=1)
        # self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64*block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers = []
        self.layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.layers.append(block(self.inplanes, planes))

        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(self.relu(self.bn1(x)))
        x = self.fc2(self.relu(self.bn2(x)))
        #x = self.sigmoid(x)
        #x = self.avgpool(x)
        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)

        return x


def preresnet20(num_classes = 1000):
    """Constructs a PreResNet-20 model.
    """
    model = PreResNet(BasicBlock, [3, 3, 3], num_classes = num_classes)
    return model


def preresnet32(**kwargs):
    """Constructs a PreResNet-32 model.
    """
    model = PreResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def preresnet44(**kwargs):
    """Constructs a PreResNet-44 model.
    """
    model = PreResNet(Bottleneck, [7, 7, 7], **kwargs)
    return model


def preresnet56(**kwargs):
    """Constructs a PreResNet-56 model.
    """
    model = PreResNet(Bottleneck, [9, 9, 9], **kwargs)
    return model


def preresnet110(**kwargs):
    """Constructs a PreResNet-110 model.
    """
    model = PreResNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def preresnet1202(**kwargs):
    """Constructs a PreResNet-1202 model.
    """
    model = PreResNet(Bottleneck, [200, 200, 200], **kwargs)
    return model



class WideBasicBlockQ(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    #expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, dropRate = 0.0):
        super(WideBasicBlockQ, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.act0 = nn.ReLU(inplace = True)
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p = 0.3)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.act1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                    padding=0, bias=False) or None



    def forward(self, x):
        if not self.equalInOut:
            x = self.act0(self.bn0(x))
        else:
            out = self.act0(self.bn0(x))
        out = self.act1(self.bn1(self.conv0(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv1(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)



class WideResNet(nn.Module):
    def __init__(self, block, depth, num_classes=1000, widen_factor = 1):
        super(WideResNet, self).__init__()


        
        block.widen_factor = widen_factor

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth-4)%6 ==0) 
        n = int((depth-4)/6)

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, nChannels[0], nChannels[1], n, stride =1)
        self.layer2 = self._make_layer(block, nChannels[1], nChannels[2], n, stride=2)
        self.layer3 = self._make_layer(block, nChannels[2], nChannels[3], n, stride=2)
        
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.act = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride=1):

        self.layers = []
        for i in range(nb_layers):
            self.layers.append(block(i==0 and in_planes or out_planes, out_planes,i ==0 and stride or 1, dropRate = 0.3))
        return nn.Sequential(*self.layers)



    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def wideresnet40(depth=40, num_classes=10, widen_factor=2):

    model = WideResNet(WideBasicBlockQ, depth = depth , num_classes = num_classes, widen_factor = widen_factor)
    return model