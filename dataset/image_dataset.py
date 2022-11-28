import torch
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet50_Weights


class FER2013PlusDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, preprocess: transforms.Compose = None) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        if preprocess is None:
            self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
        else:
            self.preprocess = preprocess

        self._prepare_data()

    def _prepare_data(self):
        self.idx_to_cat = {
            0: 'neutral',
            1: 'happiness',
            2: 'surprise',
            3: 'sadness',
            4: 'anger',
            5: 'disgust',
            6: 'fear',
            7: 'contempt',
            8: 'unknown',
            9: 'not a face',
        }
        self.data = pd.read_csv(os.path.join(self.dataset_dir, 'label.csv'), header=None)

    @property
    def total_class(self):
        return len(self.idx_to_cat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_path = os.path.join(self.dataset_dir, item[0])

        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        label = torch.FloatTensor(item[2:].to_numpy(dtype=np.float32))
        label = label / len(label)
        return (img, ), label


if __name__ == '__main__':
    fer_train = FER2013PlusDataset('./data/fer2013plus/FER2013Train')
    print(fer_train[0][1])