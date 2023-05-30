

from frcnn.model.faster_rcnn.resnet import resnet


if __name__ == "__main__":
    fasterRCNN = resnet(80, 50, pretrained=True, class_agnostic=False)
    fasterRCNN.create_architecture()