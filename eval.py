import logging
import pytorch_lightning as pl
from dataset.extracted_dataset import KeypointExtractedDataset
from detectron_utils import *
logging.basicConfig(level=logging.INFO)

setup_dataset('semaphore_keypoint_test', 'data/test/annotation.json', 'data/test')
