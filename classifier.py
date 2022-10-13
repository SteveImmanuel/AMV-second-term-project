import os
import torch
import pytorch_lightning as pl
from constant import *


class Classifier(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)


class LitClassifier(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.model = Classifier(input_size, output_size)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def _calculate_acc(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        return torch.sum(y_hat == y).item() / len(y)

    def training_step(self, batch: torch.tensor, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('train/loss', loss)
        self.log('train/acc', acc)
        return loss

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('validation/loss', loss)
        self.log('validation/acc', acc)
        return loss

    def test_step(self, batch: torch.tensor, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler':
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=LR_DECAY_FACTOR,
                    patience=LR_DECAY_PATIENCE,
                ),
                'monitor':
                'validation/loss',
                'frequency':
                1,
            }
        }
