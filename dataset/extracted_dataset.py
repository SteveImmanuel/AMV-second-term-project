import torch
from detectron_utils import *

class KeypointExtractedDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path:str, images_path:str):
        self.annotation_path = annotation_path
        self.images_path = images_path

    def _extract_keypoints(self, model:DefaultPredictor, img_path:str):
        setup_dataset('semaphore_keypoint', self.annotation_path, self.images_path)
        keypoint_cfg = get_model_config('./configs/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml')
        keypoint_model = DefaultPredictor(keypoint_cfg)
        
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx:int):
        raise NotImplementedError()