import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F

from mnist_datamodule import MNISTDataModule


class LitClassifier(pl.LightningModule):

    def __init__(
            self,
            hidden_dim: int = 128,
            learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
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

        self.metric = torchmetrics.Accuracy()

    def forward(self, x):
        return self.lenet(x)

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
    model = LitClassifier()

    # init data
    dm = MNISTDataModule()

    # train
    trainer = pl.Trainer(val_check_interval=0.25)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
