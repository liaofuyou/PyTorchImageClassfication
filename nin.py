from torch import nn
from torch.nn import Module


class NIN(Module):

    def __init__(self):
        super().__init__()
        self.nin_net = nn.Sequential(
            # 一
            self.nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            # 二
            self.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            # 三
            self.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
            # 四；标签类别数是10
            self.nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            nn.Flatten())

    @staticmethod
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
            nn.ReLU())

    def forward(self, x):
        return self.nin_net(x)
