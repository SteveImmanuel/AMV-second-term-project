import torch
import pandas as pd
import logging
import os
from typing import Optional
from tqdm import tqdm
from models.keypoint_extractor import *


class KeypointExtractedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        cache_path: Optional[str] = None,
        model_extractor: Optional[BaseKeypointExtractor] = None,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.logger = logging.getLogger(__name__)

        self.cache_path = cache_path if cache_path else f'{dataset_path}/{split}.csv'
        if os.path.exists(self.cache_path):
            self.logger.info(f'Loading extracted keypoints from {self.cache_path}')
            self.dataframe = pd.read_csv(self.cache_path)
        else:
            self.logger.info(f'No cache found for {dataset_path}:{split}, extracting keypoints')
            self._create_label_idx_mapping()
            if not model_extractor:
                self.model_extractor = MediapipeKeypointExtractor()
            else:
                self.model_extractor = model_extractor
            self._extract_keypoints()

    def _create_label_idx_mapping(self):
        labels = os.listdir(f'{self.dataset_path}/{self.split}')
        self.label_idx_mapping = {label: idx for idx, label in enumerate(labels)}

    def _extract_keypoints(self):
        total = 0
        total_missing = 0
        n_features = self.model_extractor.get_total_keypoints()
        columns = [*[f'x{i}' for i in range(n_features)], 'label']
        data = {key: [] for key in columns}

        base_path = os.path.join(self.dataset_path, self.split)

        for label in os.listdir(base_path):
            self.logger.info(f'Extracting keypoints for label: {label}')
            for img_name in tqdm(os.listdir(os.path.join(base_path, label))):
                img_path = os.path.join(base_path, label, img_name)
                img = cv2.imread(img_path)
                keypoints = self.model_extractor.extract_keypoints(img)
                data['label'].append(self.label_idx_mapping[label])
                if keypoints:
                    for i, keypoint in enumerate(keypoints):
                        data[f'x{i}'].append(keypoint)
                else:
                    for i in range(n_features):
                        data[f'x{i}'].append(0.0)
                    self.logger.warn(f'Keypoints not found for image: {img_path}, defaulting to 0')
                    total_missing += 1
                total += 1

        # for i in tqdm(range(len(dataset))):
        #     label = os.path.splitext(os.path.split(dataset[i]['file_name'])[-1])[0]
        #     keypoints = quick_predict_img(keypoint_model, dataset[i]['file_name'])

        #     assert len(keypoints) >= 1, f'No keypoints found for {dataset[i]["file_name"]}'
        #     if len(keypoints) > 1:
        #         self.logger.warn(f'Multiple keypoints found for {dataset[i]["file_name"]}, using first one')
        #     keypoints = keypoints[0].flatten()

        #     data['label'].append(label)
        #     for j, key in enumerate(columns[:-1]):
        #         data[key].append(keypoints[j].cpu().item())

        self.dataframe = pd.DataFrame(data=data)
        self.logger.info(f'Saving extracted keypoints to {self.cache_path}')
        self.dataframe.to_csv(self.cache_path, index=False)
        self.logger.info(f'Keypoints extracted for {total} images, {total_missing} missing')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        item = list(self.dataframe.iloc[idx])
        return torch.FloatTensor(item[:-1]), torch.LongTensor([self.label_idx_mapping[item[-1]]]).squeeze()


if __name__ == '__main__':
    fer2013keypoints = KeypointExtractedDataset('data/fer2013', 'train')