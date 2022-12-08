import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--model-path', help='Load pretrained model')
    parser.add_argument('--keypoint-model-path', help='Load pretrained model')
    parser.add_argument(
        '--model-type',
        help='Model type configuration',
        choices=['cnn', 'keypoint', 'keypoint-mediapipe', 'regressor', 'joint', 'joint-mediapipe'],
        type=str,
        required=True,
    )
    result = parser.parse_args()
    return result