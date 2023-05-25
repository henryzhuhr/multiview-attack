from mmdet.apis import init_detector, inference_detector

config_file = 'libs/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py'
checkpoint_file = 'pretrained/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0') 
result=inference_detector(model, 'images/test.png')
print(result)
print(result.pred_instances.bboxes)