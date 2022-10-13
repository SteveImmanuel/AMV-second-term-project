import torch
import pandas as pd
import logging
import os
from detectron_utils import *
from typing import Optional
from tqdm import tqdm

class KeypointExtractedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name:str, cache_path:Optional[str]=None, model_extractor_config:Optional[str]=None):
        self.dataset_name = dataset_name
        self.split = dataset_name.split('_')[-1]
        self.logger = logging.getLogger(__name__)

        self.cache_path = cache_path if cache_path else f'./data/{self.split}/dataframe.csv'
        if os.path.exists(self.cache_path):
            self.logger.info(f'Loading extracted keypoints from {self.cache_path}')
            self.dataframe = pd.read_csv(self.cache_path)
        else:
            self.logger.info(f'No cache found for {self.dataset_name}, extracting keypoints')
            self.model_extractor_config = model_extractor_config
            self._extract_keypoints()

        self._create_label_idx_mapping()
    
    def _create_label_idx_mapping(self):
        labels = self.dataframe['label'].unique()
        self.label_idx_mapping = {label:idx for idx, label in enumerate(labels)}

    def _extract_keypoints(self):
        dataset = DatasetCatalog.get(self.dataset_name)
        if self.model_extractor_config:
            keypoint_cfg = get_model_config(self.model_extractor_config)
        else:
            keypoint_cfg = get_model_config()
        keypoint_model = DefaultPredictor(keypoint_cfg)

        columns = [[f'x{i}', f'y{i}', f'c{i}'] for i in range(17)]
        columns = [*[item for sublist in columns for item in sublist], 'label']
        data = {key: [] for key in columns}
        for i in tqdm(range(len(dataset))):
            label = os.path.splitext(os.path.split(dataset[i]['file_name'])[-1])[0]
            keypoints = quick_predict_img(keypoint_model, dataset[i]['file_name'])

            assert len(keypoints) >= 1, f'No keypoints found for {dataset[i]["file_name"]}'
            if len(keypoints) > 1:
                self.logger.warn(f'Multiple keypoints found for {dataset[i]["file_name"]}, using first one')
            keypoints = keypoints[0].flatten()
            
            data['label'].append(label)
            for j, key in enumerate(columns[:-1]):
                data[key].append(keypoints[j].cpu().item())

        self.dataframe = pd.DataFrame(data=data)
        self.logger.info(f'Saving extracted keypoints to {self.cache_path}')
        self.dataframe.to_csv(self.cache_path, index=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx:int):
        item = list(self.dataframe.iloc[idx])
        return torch.FloatTensor(item[:-1]), torch.LongTensor([self.label_idx_mapping[item[-1]]]).squeeze()