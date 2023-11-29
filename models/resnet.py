'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.normalize import standardization


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_activation=False):
        x = standardization(x)

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        x1 = out
        out = self.layer4(out)
        x2 = out
        out = F.avg_pool2d(out, 4)
        x3 = out
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if return_activation:
            return out, x3
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class NaiveCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NaiveCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d()
        self.fc1 = nn.Sequential(nn.Linear(64 * 7 * 7, 128),
                                 nn.ReLU(inplace=True))
        self.dropout3 = nn.Dropout()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_activation=False, return_snnl_activation=False):
        x1 = self.conv1(x)
        out = self.pool1(x1)
        out = self.dropout1(out)

        x2 = self.conv2(out)
        out = self.pool2(x2)
        out = self.dropout2(out)

        x3 = self.fc1(out.view(out.size(0), -1))
        x3 = self.dropout3(x3)
        out = self.fc2(x3)

        if return_activation:
            return out, x3
        if return_snnl_activation:
            return [x1, x2, x3, out]
        return out


class MoreNaiveCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MoreNaiveCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d()
        self.fc1 = nn.Sequential(nn.Linear(32 * 7 * 7, 64),
                                 nn.ReLU(inplace=True))
        self.dropout3 = nn.Dropout()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, return_activation=False):
        x1 = self.conv1(x)
        out = self.pool1(x1)
        out = self.dropout1(out)

        x2 = self.conv2(out)
        out = self.pool2(x2)
        out = self.dropout2(out)

        x3 = self.fc1(out.view(out.size(0), -1))
        act = self.dropout3(x3)
        out = self.fc2(x3)

        if return_activation:
            return out, act
        return out


def ResNet10(num_classes):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes)


def ResNet14(num_classes):
    return ResNet(BasicBlock, [1, 2, 2, 1], num_classes)


def ResNet16(num_classes):
    return ResNet(BasicBlock, [1, 2, 2, 2], num_classes)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def ResNet18x(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34x(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def test():
    net = ResNet34x()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    from torchsummary import summary
    summary(net, (3, 32, 32), device='cpu')

    net = NaiveCNN()
    y = net(torch.randn(1, 1, 28, 28))

    from torchsummary import summary
    summary(net, (1, 28, 28), device='cpu')

    net = MoreNaiveCNN()
    y = net(torch.randn(1, 1, 28, 28))

    from torchsummary import summary
    summary(net, (1, 28, 28), device='cpu')

