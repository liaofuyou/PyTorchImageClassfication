from torch import nn
from torch.nn import Module


class AlexNet(Module):

    def __init__(self):
        super().__init__()
        self.alexnet = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2), nn.Flatten(),

            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),

            nn.Linear(4096, 10))

    def forward(self, x):
        return self.alexnet(x)
