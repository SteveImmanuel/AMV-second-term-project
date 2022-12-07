import torch
import cv2
import mediapipe as mp
import utils.detectron as DetectronUtils
import utils.mediapipe as MediapipeUtils
from detectron2.engine.defaults import DefaultPredictor
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
from PIL import Image
from collections import OrderedDict


class BaseKeypointExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def extract_keypoints(self, image):
        raise NotImplementedError()

    def get_total_keypoints(self):
        raise NotImplementedError()


class DetectronKeypointExtractor(BaseKeypointExtractor):
    def __init__(self, model_config='./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml') -> None:
        super().__init__()
        self.model = DefaultPredictor(DetectronUtils.get_model_config(model_config))

    def extract_keypoints(self, image):
        return self.model(image)['instances'].pred_keypoints


class MediapipeKeypointExtractor(BaseKeypointExtractor):
    def __init__(self, min_confidence=0.5) -> None:
        super().__init__()
        self.model = MediapipeUtils.get_model(min_confidence)

    def extract_keypoints(self, image):
        results = self.model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints_x = []
        keypoints_y = []
        keypoints_z = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            for landmark in face_landmarks:
                keypoints_x.append(landmark.x)
                keypoints_y.append(landmark.y)
                keypoints_z.append(landmark.z)
            return keypoints_x, keypoints_y, keypoints_z
        return False

    def get_total_keypoints(self):
        return 1434


class CustomKeypointExtractor(BaseKeypointExtractor):
    def __init__(self, total_coordinates) -> None:
        super().__init__()
        self.total_coordinates = total_coordinates

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Linear(2048, total_coordinates)
        self.transform = transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
        ])

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'model.' in k:
                name = k[6:]
                new_state_dict[name] = v
        self.load_state_dict(new_state_dict)

    def extract_keypoints(self, image):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        image = self.transform(image)
        return self.forward(image.unsqueeze(0)).squeeze(0).detach().numpy()

    def get_total_keypoints(self):
        raise 30

    def forward(self, img):
        batch, channel, height, width = img.shape
        img = img.expand([batch, 3, height, width])
        return self.model(img)


if __name__ == '__main__':
    media_pipe = MediapipeKeypointExtractor()
    # detectron = DetectronKeypointExtractor()
    img = cv2.imread('upscaled16x.jpg')
    res = media_pipe.extract_keypoints(img)
    print(len(res))