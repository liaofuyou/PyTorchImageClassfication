import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from alexnet import AlexNet
from lenet import LeNet
from mnist_datamodule import MNISTDataModule
from nin import NIN
from vgg import VGG


class ImageClassifier(pl.LightningModule):

    def __init__(
            self,
            net_name="lenet",
            hidden_dim: int = 128,
            learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()

        if net_name == "lenet":
            self.net = LeNet()
        elif net_name == "alexnet":
            self.net = AlexNet()
        elif net_name == "vgg":
            self.net = VGG()
        elif net_name == "nin":
            self.net = NIN()
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
    trainer = pl.Trainer(val_check_interval=0.25)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
