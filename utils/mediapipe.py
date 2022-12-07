import cv2
import mediapipe as mp


def get_model(min_confidence=0.5):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
    )