import cv2
import mediapipe as mp


def get_model():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        refine_landmarks=True,
    )