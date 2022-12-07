import os
import torch
import pytorch_lightning as pl
from constant import *
from typing import Union, List
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class LitClassifier(pl.LightningModule):
    def __init__(self, model_factory, n_class: int, input_size: int = None) -> None:
        super().__init__()
        if input_size:
            self.model = model_factory(input_size, n_class)
        else:
            self.model = model_factory(n_class, False)
        self.loss_func = torch.nn.KLDivLoss(log_target=False)

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
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('train/batch_loss', loss)
        self.log('train/batch_acc', acc)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)
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
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('test/batch_loss', loss)
        self.log('test/batch_acc', acc)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
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


class LitRegressor(pl.LightningModule):
    def __init__(self, model_factory, n_class: int) -> None:
        super().__init__()
        self.model = model_factory(n_class)
        self.loss_func = torch.nn.MSELoss()
        # self.tensorboard = self.logger.experiment

    def training_step(self, batch: torch.tensor, batch_idx: int):
        self.model.train()
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log('train/batch_loss', loss)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)
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
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log('test/batch_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
        x, y = batch
        return x, self.model(x), y

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


class LitJoint(pl.LightningModule):
    def __init__(
        self,
        model_factory,
        n_class: int,
        total_keypoints: int,
        freeze_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.model = model_factory(n_class, total_keypoints, freeze_cnn)
        self.loss_func = torch.nn.KLDivLoss(log_target=False)
        # self.loss_func = torch.nn.CrossEntropyLoss()

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
        x1, x2, y = batch
        y_hat = self.model(x1, x2)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('train/batch_loss', loss)
        self.log('train/batch_acc', acc)
        return {'loss': loss, 'pred': y_hat, 'target': y}

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        self.model.eval()
        x1, x2, y = batch
        with torch.no_grad():
            y_hat = self.model(x1, x2)
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
        x1, x2, y = batch
        y_hat = self.model(x1, x2)
        loss = self.loss_func(y_hat, y)
        acc = self._calculate_acc(y_hat, y)
        self.log('test/batch_loss', loss)
        self.log('test/batch_acc', acc)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
        x1, x2, y = batch
        return self.model(x1, x2), y

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
    pass