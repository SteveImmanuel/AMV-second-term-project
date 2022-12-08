import logging
import torch
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torchmetrics import F1Score, Recall, Precision, ConfusionMatrix
from models.lightning import *
from models.classifier import *
from dataset import *
from models.keypoint_extractor import *
from constant import *
from constant import *
from args import parse_arg

logging.basicConfig(level=logging.INFO)
args = parse_arg()

if args.model_type == 'cnn':
    cm_name = 'cm_cnn.png'
    test_dataset = ImageDataset('./data/fer2013plus/FER2013Test')
    model = CNNClassifier(test_dataset.total_class)
elif args.model_type == 'keypoint':
    cm_name = 'cm_keypoint.png'
    test_dataset = ImageDataset('./data/fer2013plus/FER2013Test')
    model = KeypointClassifier(test_dataset.total_class, 15, args.keypoint_model_path)
elif args.model_type == 'keypoint-mediapipe':
    cm_name = 'cm_keypoint_mediapipe.png'
    test_dataset = JointDataset('./data/fer2013plus/FER2013Test')
    model = KeypointClassifierMediapipe(test_dataset.total_class, 478)
elif args.model_type == 'joint':
    cm_name = 'cm_joint.png'
    test_dataset = JointDataset('./data/fer2013plus/FER2013Test')
    model = JointClassifier(test_dataset.total_class, 15, args.keypoint_model_path, False)
elif args.model_type == 'joint-mediapipe':
    cm_name = 'cm_joint_mediapipe.png'
    test_dataset = JointDataset('./data/fer2013plus/FER2013Test')
    model = JointClassifierMediapipe(test_dataset.total_class, 478, False)
else:
    raise KeyError('Model type not supported for evaluation')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    default_root_dir=os.getcwd(),
    log_every_n_steps=1,
    accelerator='gpu',
    devices=1,
)
lighting_module = LitClassifier.load_from_checkpoint(args.model_path, model=model)
lighting_module.eval()

preds = []
gts = []
for pred, gt in trainer.predict(lighting_module, test_loader):
    pred = torch.argmax(pred, dim=1)
    gt = torch.argmax(gt, dim=1)
    preds.append(pred)
    gts.append(gt)

preds = torch.concat(preds, dim=0)
gts = torch.concat(gts, dim=0)

cm = ConfusionMatrix(test_dataset.total_class)(preds, gts)
labels = list(test_dataset.idx_to_cat.values())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.savefig(cm_name)

recall_m = Recall(test_dataset.total_class, average='macro')(preds, gts)
precision_m = Precision(test_dataset.total_class, average='macro')(preds, gts)
f1_score_m = F1Score(test_dataset.total_class, average='macro')(preds, gts)
recall_w = Recall(test_dataset.total_class, average='weighted')(preds, gts)
precision_w = Precision(test_dataset.total_class, average='weighted')(preds, gts)
f1_score_w = F1Score(test_dataset.total_class, average='weighted')(preds, gts)
acc = torch.sum(gts == preds).item() / len(gts)
print(f'Accuracy: {acc}')
print(f'Recall (Macro): {recall_m}')
print(f'Precision (Macro): {precision_m}')
print(f'F1 Score (Macro): {f1_score_m}')
print(f'Recall (Weighted): {recall_w}')
print(f'Precision (Weighted): {precision_w}')
print(f'F1 Score (Weighted): {f1_score_w}')