from mmdet.apis import init_detector, inference_detector

config_file = 'libs/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py'
checkpoint_file = 'pretrained/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, 'test/Town10HD-test/Town10HD-point_0103-distance_000-direction_1-DAS.png')

pred_results=[]
for [x1, y1, x2, y2], conf, cls in zip(
        (result.pred_instances.bboxes).cpu().numpy(),
        (result.pred_instances.scores).cpu().numpy(),
        (result.pred_instances.labels).cpu().numpy(),
    ):
    if conf>0.5:
        pred_results.append([int(x1), int(y1), int(x2), int(y2), conf, cls])

# print(pred_results)



import torch
import torchvision

model = torchvision.models.detection.FasterRCNN(num_classes=91)


target={
    "boxes": torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]]),
}
model.forward(torch.rand(1, 3, 300, 400), torch.rand(1, 4))
