import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Layer 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 通道加倍: 16->32
        self.bn1 = nn.BatchNorm2d(32)  # <--- 新增: 批归一化

        # Layer 2: Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 通道加倍: 32->64
        self.bn2 = nn.BatchNorm2d(64)  # <--- 新增: 批归一化

        self.pool = nn.MaxPool2d(2, 2)

        # Dropout: 防止过拟合
        self.dropout = nn.Dropout(0.5)  # <--- 新增: 丢弃 50% 的神经元

        # 计算全连接层的输入维度
        # 28x28 -> pool(14x14) -> pool(7x7)
        # 64通道 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 典型的 Conv-BN-ReLU-Pool 结构
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # FC 层也加 Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层之间 drop
        x = self.fc2(x)

        return x