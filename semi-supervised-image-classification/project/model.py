# model.py
import torch.nn as nn
import torch.nn.functional as F

# 基础残差块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # shortcut 连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# WideResNet 主体结构
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        # 三个 block，每个 block 由 n 个 BasicBlock 组成
        self.layer1 = self._make_layer(channels[0], channels[1], n, stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], n, stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], n, stride=2)

        self.linear = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)
