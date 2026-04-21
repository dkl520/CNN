"""
model.py — 自定义 ResNet（残差网络）
=====================================
核心思想：引入 Skip Connection（跳跃连接/捷径）
  公式：H(x) = F(x) + x
解决了深层网络中的“退化问题”（随着层数增加，准确率反而下降），
通过让网络学习“残差” F(x) = H(x) - x，使得深层模型至少不会比浅层模型差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    标准残差块（BasicBlock）- 用于 ResNet-18 和 ResNet-34

    结构逻辑：
    1. 主路径：两次 3x3 卷积，每次卷积后接 Batch Norm (BN) 稳定分布。
    2. 跳跃路径（Shortcut）：直接将输入 x 传到后面。
    3. 融合：将主路径输出与跳跃路径输出相加，再经过 ReLU 激活。
    """

    expansion = 1  # BasicBlock 的输出通道数与输入相同（不扩展）

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一层卷积：负责可能的降采样（当 stride=2 时）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 归一化，加速收敛并减少过拟合

        # 第二层卷积：保持通道数和尺寸不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径（Shortcut）部分
        self.shortcut = nn.Sequential()
        # 如果步长不为 1（尺寸变了）或者输入输出通道数不一致，
        # 则需要用 1x1 卷积在 Shortcut 路径上调整 x 的维度，以便和主路径相加。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # 1. 记录跳跃路径的 identity
        identity = self.shortcut(x)

        # 2. 计算主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 3. 将残差 F(x) 与 原始输入 x (经过维度对齐后) 相加
        out += identity
        out = F.relu(out) # 最后的激活
        return out


class BottleneckBlock(nn.Module):
    """
    瓶颈残差块（Bottleneck）- 用于更深的 ResNet-50/101/152

    采用 1x1 -> 3x3 -> 1x1 的结构：
    - 第一个 1x1 卷积用于“压减”通道数（降维），减少计算量。
    - 中间的 3x3 卷积在低维空间进行特征提取。
    - 最后一个 1x1 卷积用于“恢复”并扩展通道数（升维）。
    """

    expansion = 4 # Bottleneck 输出通道数通常是中间层的 4 倍

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels # 中间层的宽度

        # 1x1 降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 特征提取
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 升维
        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 捷径维度对齐
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
        out += identity
        return F.relu(out)


class ResNet(nn.Module):
    """
    通用 ResNet 骨干架构
    针对 CIFAR-10 (32x32) 做的优化：
    - 原版 ResNet (ImageNet) 第一层是 7x7 卷积 + MaxPool，会将图片缩小 4 倍。
    - CIFAR 图片太小，所以这里第一层改用 3x3 卷积且不使用 MaxPool，以保留更多空间信息。
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 1. Stem 部分：初始特征提取
        # 输入: [Batch, 3, 32, 32] -> 输出: [Batch, 64, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 2. 四个阶段的残差块堆叠 (Stages)
        # 每个 stage 会根据 stride=2 进行 2 倍下采样，并增加通道数
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1) # 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 16x16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x4

        # 3. 输出部分
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将 4x4 变为 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes) # 分类全连接层

        # 4. 权重初始化：使用 Kaiming (He) 初始化，这对 ReLU 激活的网络至关重要
        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建一个 Stage：包含一个可能做下采样的 block 和多个保持维度的 block。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        """参数初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming 正态分布
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BN 层：权重设为 1，偏置设为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层：高斯分布
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平为 [Batch, Features]
        x = self.fc(x)
        return x


# ── 工厂函数：通过配置参数生成特定模型 ──────────────────────────

def ResNet18(num_classes=10):
    """
    ResNet-18 结构配置：
    4个阶段，每个阶段各有 2 个 BasicBlock。总深度 = 1 + 2*4*2 + 1 = 18 层（算上权重层）。
    """
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    """ResNet-34 结构配置：[3, 4, 6, 3] 个 BasicBlock"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    # 简单的冒烟测试
    model = ResNet18()
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}") # [2, 3, 32, 32]
    print(f"输出尺寸: {output.shape}")      # [2, 10]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}") # 约 11.17M