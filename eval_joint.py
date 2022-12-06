import logging
import torch
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchmetrics import F1Score, Recall, Precision, ConfusionMatrix
from models.lightning import LitJoint
from models.classifier import JointClassifierV1, JointClassifierV2
from dataset.image_dataset import FER2013PlusDataset
from constant import *

logging.basicConfig(level=logging.INFO)

test_dataset = FER2013PlusDataset('./data/fer2013plus/FER2013Test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=1,
    accelerator='gpu',
    devices=1,
)

classifier = LitJoint.load_from_checkpoint(
    'runs/jointv2/lambda_cnn=1.0/version_8/checkpoints/epoch=10-step=2464.ckpt',
    model_factory=JointClassifierV2,
    n_class=test_dataset.total_class,
    total_keypoints=15,
    keypoint_extractor_ckpt='runs/regressor/version_6/checkpoints/epoch=19-step=2000.ckpt',
    freeze_extractor=False,
    lambda_cnn=1.0,
)
classifier.eval()
# a = trainer.predict(classifier, test_loader)
# print(len(a), len(a[0]), a[0][0].shape, a[0][1].shape)
# pred, gt = trainer.predict(classifier, test_loader)
preds = []
gts = []
for pred, gt in trainer.predict(classifier, test_loader):
    pred = torch.argmax(pred, dim=1)
    gt = torch.argmax(gt, dim=1)
    preds.append(pred)
    gts.append(gt)

preds = torch.concat(preds, dim=0)
gts = torch.concat(gts, dim=0)
# print(len(preds))
# pred_temp = Counter(preds.numpy())
# gt_temp = Counter(gts.numpy())
# pred_temp = {test_dataset.idx_to_cat[idx]: value for idx, value in pred_temp.items()}
# gt_temp = {test_dataset.idx_to_cat[idx]: value for idx, value in gt_temp.items()}
# print(pred_temp, gt_temp)

cm = ConfusionMatrix(test_dataset.total_class)(pred, gt)
labels = list(test_dataset.idx_to_cat.values())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm_baseline.png')
recall = Recall(test_dataset.total_class, average='macro')(pred, gt)
precision = Precision(test_dataset.total_class, average='macro')(pred, gt)
f1_score = F1Score(test_dataset.total_class, average='macro')(pred, gt)
acc = torch.sum(gts == preds).item() / len(gts)
print(f'Accuracy: {acc}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1_score}')