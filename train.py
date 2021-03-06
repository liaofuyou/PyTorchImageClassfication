import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from alexnet import AlexNet
from densenet import DenseNet
from googlenet import GoogleNet
from lenet import LeNet
from mnist_datamodule import MNISTDataModule
from nin import NIN
from resnet import ResNet
from vgg import VGG


class ImageClassifier(pl.LightningModule):

    def __init__(
            self,
            net_name="LeNet",
            hidden_dim: int = 128,
            learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()

        if net_name == "AlexNet":
            self.net = AlexNet()
        elif net_name == "VGG":
            self.net = VGG()
        elif net_name == "NIN":
            self.net = NIN()
        elif net_name == "GoogleNet":
            self.net = GoogleNet()
        elif net_name == "ResNet":
            self.net = ResNet()
        elif net_name == "DenseNet":
            self.net = DenseNet()
        else:
            self.net = LeNet()

        self.metric = torchmetrics.Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Accuracy
        print(f"Accuracy on batch {batch_idx}: {self.metric(y_hat, y)}")
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        # Accuracy
        print(f"Accuracy on all data: {self.metric.compute()}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    # init model
    model = ImageClassifier("lenet")

    # init data
    dm = MNISTDataModule()

    # train
    trainer = pl.Trainer(max_epochs=10, val_check_interval=0.25)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
