import cv2
import mediapipe as mp
import numpy as np


def get_model(min_confidence=0.5):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
    )


def get_detect_model(min_confidence=0.5):
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    )


def get_bounding_box(model, img):
    result = model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.detections:
        return None
    bbox = result.detections[0].location_data.relative_bounding_box
    return bbox.xmin, bbox.ymin, bbox.width, bbox.height


def crop_face(img, bbox):
    x, y, w, h = bbox
    return np.copy(img[int(y * img.shape[0]):int((y + h) * img.shape[0]),
                       int(x * img.shape[1]):int((x + w) * img.shape[1])])
