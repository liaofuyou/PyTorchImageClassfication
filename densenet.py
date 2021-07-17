import torch
from torch import nn
from torch.nn import Module


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    @staticmethod
    def conv_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=(3, 3), padding=1))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


class DenseNet(Module):

    def __init__(self):
        super().__init__()

        # 第一块
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))

        # `num_channels`为当前的通道数
        num_channels, growth_rate = 64, 32

        # 四个密集快
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):

            # 稠密块
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate

            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net = nn.Sequential(b1, *blks,
                                 nn.BatchNorm2d(num_channels),
                                 nn.ReLU(),
                                 nn.AdaptiveMaxPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(num_channels, 10))

    @staticmethod
    def transition_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=(1, 1)),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.net(x)
