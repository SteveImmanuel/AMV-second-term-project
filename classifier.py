import os
import torch
import pytorch_lightning as pl


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

    def training_step(self, batch:torch.tensor, batch_idx:int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
