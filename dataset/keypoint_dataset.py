import torch
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
from typing import Tuple


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_path: str,
            split: str,
            img_res: Tuple[int, int] = (48, 48),
            val_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.img_res = img_res
        self.split = split
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(48),
                transforms.GaussianBlur(kernel_size=(5, 9)),
                transforms.RandomPosterize(bits=5, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomEqualize(p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(48),
                transforms.ToTensor(),
            ])

        self._prepare_data()
        self.total_train = int((1 - val_ratio) * len(self.data))

    def _prepare_data(self):
        self.data = pd.read_csv(self.dataset_path)
        self.data = self.data.fillna(method='ffill')

    def __len__(self):
        if self.split == 'test':
            return len(self.data)
        else:
            if self.split == 'train':
                return self.total_train
            else:
                return len(self.data) - self.total_train

    def _preprocess_img(self, img):
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(96, 96)

        img = Image.fromarray(img)
        return self.transform(img)

    def __getitem__(self, idx):
        if self.split == 'validation':
            idx += self.total_train

        item = self.data.iloc[idx]
        img = item['Image'].split(' ')
        img = self._preprocess_img(img)

        if self.split != 'test':
            label = torch.FloatTensor(item[0:30].to_numpy(dtype=np.float32)) / 96
            return (img, ), label
        else:
            return (img, ), torch.empty(0)

    @property
    def total_coordinates(self):
        return 30


if __name__ == '__main__':
    dataset = KeypointDataset('./data/keypoints/test.csv', split='test')
    # import matplotlib.pyplot as plt
    # plt.imshow(dataset[0][0].permute(1, 2, 0), cmap='gray')
    # plt.show()

    # print(dataset[0][0], dataset[0][1])

    from models.keypoint_extractor import CustomKeypointExtractor
    model = CustomKeypointExtractor(30)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    iterator = iter(dataloader)
    x, y = next(iterator)
    x, y = next(iterator)
    x, y = next(iterator)
    print(x)
    print(model(*x))

    # model.fc = torch.nn.Linear(2048, 30)
