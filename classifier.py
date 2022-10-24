import os
from typing import Tuple
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
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)


class ClassifierV2(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, conv_channel: int = 4):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, conv_channel, 1, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear((input_size // 2) * conv_channel, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, output_size),
        )
        self.transformer_enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(conv_channel, 1, conv_channel * 4, batch_first=True),
            8,
        )

    def forward(self, x: torch.tensor):
        batch, n_features = x.shape

        x = x.view(batch, 2, n_features // 2)
        conv_out = self.conv(x).permute(0, 2, 1)

        out = self.transformer_enc(conv_out)  # (batch, n_features // 2, conv_channel)
        out = out.flatten(1)  # (batch, n_features // 2 * conv_channel)

        final_out = self.head(out)
        return final_out


class LitClassifier(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int, model_factory) -> None:
        super().__init__()
        self.model = model_factory(input_size, output_size)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    def _calculate_acc(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        return torch.sum(y_hat == y).item() / len(y)

    def training_step(self, batch: torch.tensor, batch_idx: int):
        self.model.train()
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('train/loss', loss)
        self.log('train/acc', acc)
        return loss

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('validation/loss', loss)
        self.log('validation/acc', acc)
        return loss

    def test_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        return self.model(x), y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler':
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=LR_DECAY_FACTOR,
                    patience=LR_DECAY_PATIENCE,
                    min_lr=LR_MIN,
                ),
                'monitor':
                'validation/loss',
                'frequency':
                1,
            },
        }


if __name__ == '__main__':
    model = ClassifierV2(6 * 2, 28).cuda()
    rand_tensor = torch.rand(5, 6 * 2).cuda()
    model(rand_tensor)