import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

def setup_dataset(name:str, annotation_path:str, images_path:str):
    register_coco_instances(name, {}, annotation_path, images_path)

def get_model_config(config_file:str='./configs/keypoint_rcnn_R_50_FPN_3x.yaml', threshold:float=.7):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    return cfg

def quick_predict_img(model:DefaultPredictor, img_path:str):
    img = cv2.imread(img_path)
    return model(img)['instances'].pred_keypoints
    

if __name__ == '__main__':
    setup_dataset('simple_keypoint', 'dataset/steve/result/result.json', 'dataset/steve/ds0/img')
    cfg = get_model_config('./configs/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml')
    model = DefaultPredictor(cfg)
    pred = quick_predict_img(model, 'dataset/steve/ds0/img/A.jpg')
    print(pred)