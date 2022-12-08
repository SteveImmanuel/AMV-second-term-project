import torch
import pandas as pd
import numpy as np
import logging
import os
from typing import Optional
from tqdm import tqdm
from models.keypoint_extractor import *


class JointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        preprocess: transforms.Compose = None,
        cache_path: Optional[str] = None,
        model_extractor: Optional[BaseKeypointExtractor] = None,
    ):
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)
        self._create_idx_mapping()

        if preprocess is None:
            self.preprocess = transforms.Compose([transforms.ToTensor()])
        else:
            self.preprocess = preprocess

        default_cache_name = os.path.join(dataset_path, f'{os.path.split(dataset_path)[-1]}.csv')
        self.cache_path = cache_path if cache_path else default_cache_name
        if os.path.exists(self.cache_path):
            self.logger.info(f'Loading extracted keypoints from {self.cache_path}')
            self.dataframe = pd.read_csv(self.cache_path)
        else:
            self.logger.info(f'No cache found for {dataset_path}, extracting keypoints')
            if not model_extractor:
                self.model_extractor = MediapipeKeypointExtractor(.2)
            else:
                self.model_extractor = model_extractor
            self._extract_keypoints()

    def _create_idx_mapping(self):
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

    @property
    def total_class(self):
        return len(self.idx_to_cat)

    def _extract_keypoints(self):
        source_data = pd.read_csv(os.path.join(self.dataset_path, 'label.csv'), header=None)

        total = 0
        total_missing = 0
        n_features = self.model_extractor.get_total_keypoints()
        columns = [[f'x{i}', f'y{i}', f'z{i}'] for i in range(n_features // 3)]
        columns = [item for sublist in columns for item in sublist]
        columns = ['image', *columns, 'label']
        data = {key: [] for key in columns}

        for i in tqdm(range(len(source_data))):
            item = source_data.iloc[i]
            img_path = os.path.join(self.dataset_path, item[0])
            label = ','.join(map(str, item[2:].to_numpy(dtype=np.float32)))

            img = cv2.imread(img_path)
            keypoints = self.model_extractor.extract_keypoints(img)
            
            data['image'].append(item[0])
            data['label'].append(label)
            if keypoints:
                x, y, z = keypoints
                x = np.array(x)
                y = np.array(y)
                z = np.array(z)

                x = (x - np.min(x)) / (np.max(x) - np.min(x))
                y = (y - np.min(y)) / (np.max(y) - np.min(y))
                z = (z - np.min(z)) / (np.max(z) - np.min(z))

                x = x.tolist()
                y = y.tolist()
                z = z.tolist()

                for i in range(0, n_features // 3):
                    data[f'x{i}'].append(x[i])
                    data[f'y{i}'].append(y[i])
                    data[f'z{i}'].append(z[i])
            else:
                self.logger.warning(f'Keypoints not found for image: {img_path}, defaulting to zero')
                for i in range(0, n_features // 3):
                    data[f'x{i}'].append(0)
                    data[f'y{i}'].append(0)
                    data[f'z{i}'].append(0)
                total_missing += 1
            total += 1
        self.dataframe = pd.DataFrame(data=data)
        self.logger.info(f'Saving extracted keypoints to {self.cache_path}')
        self.dataframe.to_csv(self.cache_path, index=False)
        if total_missing > 0:
            self.logger.warning(f'Keypoints extracted for {total} images, {total_missing} missing')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        item = list(self.dataframe.iloc[idx])
        img_path = os.path.join(self.dataset_path, item[0])
        img = Image.open(img_path)
        img = self.preprocess(img)
        features = torch.tensor(item[1:-1], dtype=torch.float32)
        label = list(map(float, item[-1].split(',')))
        label = torch.tensor(label, dtype=torch.float32) / 10.0
        return (img, features), label


if __name__ == '__main__':
    fer2013keypoints = JointDataset('data/fer2013plus/FER2013Valid')
    fer2013keypoints = JointDataset('data/fer2013plus/FER2013Train')
    fer2013keypoints = JointDataset('data/fer2013plus/FER2013Test')
    # print(fer2013keypoints[0][1].shape)