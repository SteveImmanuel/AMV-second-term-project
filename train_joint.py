import logging
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from models.lightning import LitJoint
from models.classifier import JointClassifierV1, JointClassifierV2
from dataset.image_dataset import FER2013PlusDataset
from constant import *

logging.basicConfig(level=logging.INFO)

train_dataset = FER2013PlusDataset('./data/fer2013plus/FER2013Train', split='train')
val_dataset = FER2013PlusDataset('./data/fer2013plus/FER2013Valid', split='validation')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

lambda_cnn = 1
classifier = LitJoint(
    JointClassifierV1,
    train_dataset.total_class,
    15,
    'runs/regressor/version_6/checkpoints/epoch=19-step=2000.ckpt',
    False,
    lambda_cnn,
)

# logger = TensorBoardLogger(save_dir=os.getcwd(), name=f'runs/jointv2')
logger = TensorBoardLogger(save_dir=os.getcwd(), name=f'runs/jointv1/lambda_cnn={lambda_cnn}')
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