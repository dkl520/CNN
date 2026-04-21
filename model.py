"""
model.py — 自定义 ResNet（残差网络）
=====================================
核心思想：引入 Skip Connection（跳跃连接）
  output = F(x) + x
解决了深层网络的梯度消失问题，让网络可以很深。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    标准残差块（BasicBlock）

    结构示意：
                  ┌──────────────┐
    x ────────────┤   Shortcut   ├────────────────────┐
                  └──────────────┘                     │
                  │                                    ▼
                  ▼                                  [+] → ReLU → output
    Conv3x3 → BN → ReLU → Conv3x3 → BN ────────────┘

    当 stride>1 或 通道数变化时，Shortcut 用 1×1 Conv 做维度对齐。
    """

    expansion = 1  # BasicBlock 不扩展通道数

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 跳跃连接（当尺寸或通道不匹配时用 1×1 Conv 对齐）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)          # 跳跃路径

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)         # ← 残差相加！核心所在
        return out


class BottleneckBlock(nn.Module):
    """
    瓶颈残差块（用于更深的 ResNet-50/101/152）
    1×1 降维 → 3×3 卷积 → 1×1 升维，大幅减少参数量
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid = out_channels

        self.conv1 = nn.Conv2d(in_channels,  mid,                  kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid,           mid,                  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid,           out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)


class ResNet(nn.Module):
    """
    通用 ResNet，支持 ResNet-18 / 34（BasicBlock）
    针对 CIFAR-10（32×32）优化，第一层不做大幅降采样。

    网络结构（CIFAR版）：
      Conv 3×3 → BN → ReLU
      Layer1: [64,  64]  × n_blocks
      Layer2: [64,  128] × n_blocks, stride=2
      Layer3: [128, 256] × n_blocks, stride=2
      Layer4: [256, 512] × n_blocks, stride=2
      AdaptiveAvgPool → FC(512 → num_classes)
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # CIFAR-10：输入32×32，用3×3 Conv（不用7×7+MaxPool，否则太小）
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)   # 全局平均池化 → 1×1
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化（Kaiming）
        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """堆叠多个残差块，组成一个 Stage"""
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ── 工厂函数 ──────────────────────────────────────────
def ResNet18(num_classes=10):
    """ResNet-18: 每个Stage 2个BasicBlock"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    """ResNet-34: 每个Stage更多Block"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    model = ResNet18()
    dummy = torch.randn(2, 3, 32, 32)   # batch=2, RGB, 32×32
    out   = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {out.shape}")     # 应该是 [2, 10]
    total = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {total:,}")       # ~11M
