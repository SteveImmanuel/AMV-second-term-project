import logging
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from models.lightning import LitClassifier
from models.classifier import KeypointClassifier
from dataset.joint_dataset import KeypointExtractedDataset
from constant import *

logging.basicConfig(level=logging.INFO)

train_dataset = KeypointExtractedDataset('data/fer2013plus/FER2013Train')
val_dataset = KeypointExtractedDataset('data/fer2013plus/FER2013Valid')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

lambda_cnn = 1
classifier = LitClassifier(
    KeypointClassifier,
    train_dataset.total_class,
    1434,
)

logger = TensorBoardLogger(save_dir=os.getcwd(), name=f'runs/keypoint')
early_stop_cb = EarlyStopping(monitor='validation/loss', patience=10, verbose=True, mode='min')
checkpoint_cb = ModelCheckpoint(monitor='validation/loss', save_top_k=1, mode='min')
trainer = pl.Trainer(
    logger=logger,
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=5,
    accelerator='gpu',
    devices=1,
    callbacks=[early_stop_cb, checkpoint_cb],
)
trainer.fit(classifier, train_loader, val_loader)