import logging
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchmetrics import F1Score, Recall, Precision, ConfusionMatrix
from classifier import LitClassifier, Classifier
from dataset.extracted_dataset import KeypointExtractedDataset
from detectron_utils import *
from constant import *

logging.basicConfig(level=logging.INFO)

setup_dataset('semaphore_keypoint_test', 'data/test/annotation.json', 'data/test')

test_dataset = KeypointExtractedDataset('semaphore_keypoint_test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
classifier = LitClassifier(17 * 3, 28, Classifier)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=1,
    accelerator='gpu',
    devices=1,
)

classifier = LitClassifier.load_from_checkpoint(
    'runs/baseline/checkpoints/best.ckpt',
    input_size=17 * 3,
    output_size=28,
    model_factory=Classifier,
)
classifier.eval()
(pred, gt), = trainer.predict(classifier, test_loader)
pred = torch.argmax(pred, dim=1)

cm = ConfusionMatrix(28)(pred, gt)
labels = list(test_dataset.label_idx_mapping.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm_baseline.png')
recall = Recall(28, average='macro')(pred, gt)
precision = Precision(28, average='macro')(pred, gt)
f1_score = F1Score(28, average='macro')(pred, gt)
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1_score}')