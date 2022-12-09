import os
import torch
import pytorch_lightning as pl
from constant import *
from typing import Union, List
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class LitClassifier(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.kl_div = torch.nn.KLDivLoss(log_target=False, reduction='none')
        self.class_weight = torch.tensor(
            [
                0.5,
                0.5,
                1.10672511,
                1,
                1.48783994,
                3.1587219,
                2.33665907,
                3.02619148,
                1.98894292,
                4,
            ],
            device='cuda',
        )

    def loss_func(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        raw_loss = self.kl_div(y_hat, y)  # (batch_size, n_class)
        loss = torch.einsum('ij,j->i', raw_loss, self.class_weight)
        loss = torch.mean(loss)
        return loss

    def _calculate_acc(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        y_pseudo = torch.argmax(y, dim=1)
        return torch.sum(y_hat == y_pseudo).item() / len(y_pseudo)

    def _calculate_acc_raw(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        y_pseudo = torch.argmax(y, dim=1)
        return torch.sum(y_hat == y_pseudo).item(), len(y_pseudo)

    def training_step(self, batch: torch.tensor, batch_idx: int):
        self.model.train()
        x, y = batch
        y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('train/batch_loss', loss)
        self.log('train/batch_acc', acc)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('validation/batch_loss', loss)
        self.log('validation/batch_acc', acc)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        total_correct = 0
        total_data = 0
        total_loss = 0
        for output in outputs:
            b_correct, b_data = self._calculate_acc_raw(output['pred'], output['target'])
            total_correct += b_correct
            total_data += b_data
            total_loss += output['loss']

        self.log('train/acc', total_correct / total_data, prog_bar=True)
        self.log('train/loss', total_loss / len(outputs), prog_bar=True)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        total_correct = 0
        total_data = 0
        total_loss = 0
        for output in outputs:
            b_correct, b_data = self._calculate_acc_raw(output['pred'], output['target'])
            total_correct += b_correct
            total_data += b_data
            total_loss += output['loss']

        self.log('validation/acc', total_correct / total_data, prog_bar=True)
        self.log('validation/loss', total_loss / len(outputs), prog_bar=True)
        return super().validation_epoch_end(outputs)

    def test_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('test/batch_loss', loss)
        self.log('test/batch_acc', acc)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
        x, y = batch
        return self.model(*x), y

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


class LitRegressor(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss_func = torch.nn.MSELoss()

    def training_step(self, batch: torch.tensor, batch_idx: int):
        self.model.train()
        x, y = batch
        y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        self.log('train/batch_loss', loss)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        self.log('validation/batch_loss', loss)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        total_loss = 0
        for output in outputs:
            total_loss += output['loss']

        self.log('train/loss', total_loss / len(outputs), prog_bar=True)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        total_loss = 0
        for output in outputs:
            total_loss += output['loss']

        self.log('validation/loss', total_loss / len(outputs), prog_bar=True)
        return super().validation_epoch_end(outputs)

    def test_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        y_hat = self.model(*x)
        loss = self.loss_func(y_hat, y)
        self.log('test/batch_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
        x, y = batch
        return x, self.model(*x), y

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
