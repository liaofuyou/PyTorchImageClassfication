from torch import nn
from torch.nn import Module
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1)

        # 匹配维度
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(Module):

    def __init__(self):
        super().__init__()
        # 一、普通
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 二、残差
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        # 三、残差
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        # 四、残差
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        # 五、残差
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        # 结果
        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

    @staticmethod
    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        return self.net(x)
