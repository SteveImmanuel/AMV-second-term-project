import torch
import cv2
import mediapipe as mp
import utils.detectron as DetectronUtils
import utils.mediapipe as MediapipeUtils
from detectron2.engine.defaults import DefaultPredictor


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
    def __init__(self) -> None:
        super().__init__()
        self.model = MediapipeUtils.get_model()

    def extract_keypoints(self, image):
        results = self.model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            for landmark in face_landmarks:
                keypoints += [landmark.x, landmark.y, landmark.z]
            return keypoints
        return False

    def get_total_keypoints(self):
        return 1434


if __name__ == '__main__':
    media_pipe = MediapipeKeypointExtractor()
    # detectron = DetectronKeypointExtractor()
    img = cv2.imread('upscaled16x.jpg')
    res = media_pipe.extract_keypoints(img)
    print(len(res))