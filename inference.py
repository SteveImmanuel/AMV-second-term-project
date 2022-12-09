import cv2
import torch
import numpy as np
import mediapipe as mp
from args import parse_inference_arg
from utils.mediapipe import get_detect_model, get_model, get_bounding_box, crop_face
from models.keypoint_extractor import MediapipeKeypointExtractor
from models.classifier import JointClassifierMediapipe
from models.lightning import LitClassifier


def get_keypoint_features(face):
    landmark = keypoint_extractor.get_landmark(face)
    if landmark:
        x, y, z = keypoint_extractor.get_normalized_keypoints(landmark)
        keypoints = torch.cat(
            (
                torch.tensor(x, device='cuda', dtype=torch.float),
                torch.tensor(y, device='cuda', dtype=torch.float),
                torch.tensor(z, device='cuda', dtype=torch.float),
            ),
            dim=0,
        )
        keypoints = keypoints.reshape(3, -1).T.flatten()
        return landmark, keypoints
    return None, torch.zeros(478 * 3, dtype=torch.float, device='cuda')


def draw_landmarks(mp_drawing, mp_drawing_styles, image, mp_face_mesh, landmark):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmark,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmark,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmark,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
    )


classes = {
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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

model_path = parse_inference_arg().model_path
bbox_model = get_detect_model()
classifier = JointClassifierMediapipe(len(classes), 478)
lighting_module = LitClassifier.load_from_checkpoint(model_path, model=classifier)
model = lighting_module.model
model.cuda()
model.eval()
keypoint_extractor = MediapipeKeypointExtractor()
is_recording = False

cv2.namedWindow('Facial Expression Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Facial Expression Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FREERATIO)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue
    window = np.zeros(2304000, dtype=np.uint8).reshape(480, 640 + 480 + 480, 3) + 255
    bbox = get_bounding_box(bbox_model, img)
    if bbox:
        colored_face = crop_face(img, bbox)
        face = cv2.cvtColor(colored_face, cv2.COLOR_BGR2GRAY)
        landmark, keypoint_features = get_keypoint_features(face)

        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0)
        face_tensor = torch.tensor(face, device='cuda', dtype=torch.float) / 255.0
        with torch.no_grad():
            prediction = torch.exp(model(face_tensor.unsqueeze(0), keypoint_features.unsqueeze(0)))
        prediction = prediction.squeeze().cpu().numpy()
        res = list(zip(classes.values(), prediction))
        draw_landmarks(mp_drawing, mp_drawing_styles, colored_face, mp_face_mesh, landmark)
        colored_face = cv2.resize(colored_face, (480, 480))
        window[:, :640, :] = img
        window[:, 640:1120, :] = colored_face

        for i in range(len(res)):
            text_pos = (1120 + 10, 480 // len(res) * i + (480 // (len(res) * 2)))
            start_point = (text_pos[0] + 90, text_pos[1] - 10)
            end_point = (text_pos[0] + 90 + int(300 * res[i][1]), text_pos[1])
            num_pos = (end_point[0] + 10, text_pos[1])
            cv2.putText(window, res[i][0], text_pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA, False)
            cv2.putText(window, f'{res[i][1]:.4f}', num_pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA,
                        False)
            cv2.rectangle(window, start_point, end_point, (0, 0, 0), -1)
        cv2.imshow('Facial Expression Detection', window)
        if is_recording:
            writer.write(window)

    keypress = cv2.waitKey(5) & 0xFF
    if keypress == 27:
        break
    elif keypress == ord('r') and not is_recording:
        is_recording = True
        writer = cv2.VideoWriter('rec.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=23, frameSize=(1600, 480))
        print('Recording started')
    elif keypress == ord('r') and is_recording:
        is_recording = False
        writer.release()
        print('Recording stopped, file saved to rec.mp4')

cap.release()