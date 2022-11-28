import cv2
import os
import torch
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


def setup_dataset(name: str, annotation_path: str, images_path: str):
    register_coco_instances(name, {}, annotation_path, images_path)


def get_model_config(
    config_file: str = './configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
    threshold: float = .7,
):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    model_weight = '/'.join(os.path.normpath(config_file).split(os.sep)[-2:])
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_weight)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return cfg


def quick_predict_img(model: DefaultPredictor, img_path: str) -> torch.tensor:
    img = cv2.imread(img_path)
    return model(img)['instances'].pred_keypoints


if __name__ == '__main__':
    setup_dataset('semaphore_keypoint_train', 'data/train/annotation.json', 'data/train')
    setup_dataset('semaphore_keypoint_val', 'data/val/annotation.json', 'data/val')
    setup_dataset('semaphore_keypoint_test', 'data/test/annotation.json', 'data/test')

    dataset_dicts = DatasetCatalog.get('semaphore_keypoint_train')
    dataset_metadata = MetadataCatalog.get("semaphore_keypoint_train")
    d = dataset_dicts[5]
    # cfg = get_model_config()
    # model = DefaultPredictor(cfg)
    # pred = quick_predict_img(model, d["file_name"])
    # print(pred.shape)

    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    cv2.namedWindow('temp', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('temp', out.get_image()[:, :, ::-1])
    cv2.resizeWindow('temp', 800, 800)

    cfg = get_model_config()
    model = DefaultPredictor(cfg)
    pred = model(img)

    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
    out = visualizer.draw_instance_predictions(pred['instances'].to('cpu'))
    print(pred['instances'].pred_keypoints)
    print(d['width'], d['height'])
    print(torch.tensor(d['annotations'][0]['keypoints']).reshape(1, 17, 3))

    a = torch.tensor(d['annotations'][0]['keypoints']).reshape(1, 17, 3)
    b = pred['instances'].pred_keypoints.cpu()
    print(a - b)
    cv2.namedWindow('temp2', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('temp2', out.get_image()[:, :, ::-1])
    cv2.resizeWindow('temp2', 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
