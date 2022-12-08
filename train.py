import logging
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from models.lightning import *
from models.classifier import *
from dataset import *
from models.keypoint_extractor import *
from constant import *
from args import parse_arg

logging.basicConfig(level=logging.INFO)
args = parse_arg()

if args.model_type == 'cnn':
    train_dataset = ImageDataset('./data/fer2013plus/FER2013Train')
    val_dataset = ImageDataset('./data/fer2013plus/FER2013Valid')
    model = CNNClassifier(train_dataset.total_class)
    lighting_module = LitClassifier(model)
    log_dir = 'runs/cnn'
elif args.model_type == 'keypoint':
    train_dataset = ImageDataset('data/fer2013plus/FER2013Train')
    val_dataset = ImageDataset('data/fer2013plus/FER2013Valid')
    model = KeypointClassifier(train_dataset.total_class, 15, args.keypoint_model_path)
    lighting_module = LitClassifier(model)
    log_dir = 'runs/keypoint'
elif args.model_type == 'keypoint-mediapipe':
    train_dataset = JointDataset('data/fer2013plus/FER2013Train')
    val_dataset = JointDataset('data/fer2013plus/FER2013Valid')
    model = KeypointClassifierMediapipe(train_dataset.total_class, 478)
    lighting_module = LitClassifier(model)
    log_dir = 'runs/keypoint'
elif args.model_type == 'joint':
    train_dataset = JointDataset('data/fer2013plus/FER2013Train')
    val_dataset = JointDataset('data/fer2013plus/FER2013Valid')
    model = JointClassifier(train_dataset.total_class, 15, args.keypoint_model_path, False)
    lighting_module = LitClassifier(model)
    log_dir = 'runs/joint'
elif args.model_type == 'joint-mediapipe':
    train_dataset = JointDataset('data/fer2013plus/FER2013Train')
    val_dataset = JointDataset('data/fer2013plus/FER2013Valid')
    model = JointClassifierMediapipe(train_dataset.total_class, 478, False)
    lighting_module = LitClassifier(model)
    log_dir = 'runs/joint'
elif args.model_type == 'regressor':
    train_dataset = KeypointDataset('./data/keypoints/training.csv', split='train')
    val_dataset = KeypointDataset('./data/keypoints/training.csv', split='validation')
    model = CustomKeypointExtractor(train_dataset.total_coordinates)
    lighting_module = LitRegressor(model)
    log_dir = 'runs/regressor'

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
early_stop_cb = EarlyStopping(monitor='validation/loss', patience=10, verbose=True, mode='min')
checkpoint_cb = ModelCheckpoint(monitor='validation/loss', save_top_k=1, mode='min')
logger = TensorBoardLogger(save_dir=os.getcwd(), name=log_dir)
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
trainer.fit(lighting_module, train_loader, val_loader)