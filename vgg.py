from torch import nn
from torch.nn import Module


class VGG(Module):

    def __init__(self):
        super().__init__()
        # 五个块
        # (num_convs, out_channels)
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        # 全连接层部分
        fully_connections = [
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 10)
        ]

        self.vgg = nn.Sequential(*conv_blks,
                                 nn.Flatten(),
                                 *fully_connections)

    @staticmethod
    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    @staticmethod
    def get_small_conv_arch(conv_arch: list):
        ratio = 4
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        return small_conv_arch

    def forward(self, x):
        return self.vgg(x)
