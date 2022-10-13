import logging
import pytorch_lightning as pl
from dataset.extracted_dataset import KeypointExtractedDataset
from detectron_utils import *
logging.basicConfig(level=logging.INFO)


setup_dataset('semaphore_keypoint_train', 'data/train/annotation.json', 'data/train')
setup_dataset('semaphore_keypoint_val', 'data/val/annotation.json', 'data/val')
train_dataset = KeypointExtractedDataset('semaphore_keypoint_train')
val_dataset = KeypointExtractedDataset('semaphore_keypoint_val')