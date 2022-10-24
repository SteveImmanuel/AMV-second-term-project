import logging
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchmetrics import Accuracy, F1Score, Recall, Precision, ConfusionMatrix
from classifier import LitClassifier, ClassifierV2, Classifier
from dataset.extracted_dataset import KeypointExtractedDatasetV2
from detectron_utils import *
from constant import *

logging.basicConfig(level=logging.INFO)

setup_dataset('semaphore_keypoint_test', 'data/test/annotation.json', 'data/test')

test_dataset = KeypointExtractedDatasetV2('semaphore_keypoint_test', 'data/test/df2.csv')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
classifier = LitClassifier(6 * 2, 28, Classifier)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=1,
    accelerator='gpu',
    devices=1,
)

classifier = LitClassifier.load_from_checkpoint(
    'runs/improved/checkpoints/best.ckpt',
    input_size=6 * 2,
    output_size=28,
    model_factory=ClassifierV2,
)
classifier.eval()
(pred, gt), = trainer.predict(classifier, test_loader)
pred = torch.argmax(pred, dim=1)

cm = ConfusionMatrix(28)(pred, gt)
labels = list(test_dataset.label_idx_mapping.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm_improved.png')
recall = Recall(num_classes=28, average='macro')(pred, gt)
precision = Precision(num_classes=28, average='macro')(pred, gt)
f1_score = F1Score(num_classes=28, average='macro')(pred, gt)
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1_score}')