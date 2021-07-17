from torch import nn
from torch.nn import Sequential, Module


class LeNet(Module):

    def __init__(self):
        super().__init__()
        self.lenet = Sequential(nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2),
                                nn.Sigmoid(),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(6, 16, kernel_size=(5, 5)),
                                nn.Sigmoid(),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(),
                                nn.Linear(120, 84),
                                nn.Sigmoid(),
                                nn.Linear(84, 10))

    def forward(self, x):
        return self.lenet(x)
