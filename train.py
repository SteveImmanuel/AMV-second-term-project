import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from classifier import LitClassifier
from dataset.extracted_dataset import KeypointExtractedDataset
from detectron_utils import *
from constant import *

# default logger used by trainer

logging.basicConfig(level=logging.INFO)

setup_dataset('semaphore_keypoint_train', 'data/train/annotation.json', 'data/train')
setup_dataset('semaphore_keypoint_val', 'data/val/annotation.json', 'data/val')
setup_dataset('semaphore_keypoint_test', 'data/test/annotation.json', 'data/test')

train_dataset = KeypointExtractedDataset('semaphore_keypoint_train')
val_dataset = KeypointExtractedDataset('semaphore_keypoint_val')
test_dataset = KeypointExtractedDataset('semaphore_keypoint_test')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classifier = LitClassifier(17 * 3, 28)

logger = TensorBoardLogger(save_dir=os.getcwd(), name='runs')
trainer = pl.Trainer(
    logger=logger,
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=2,
    accelerator='gpu',
    devices=1,
)
trainer.fit(classifier, train_loader, val_loader)
trainer.test(classifier, test_loader)