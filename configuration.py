import os

class Configuration:
    def __init__(self) -> None:
        self.iou_threshold = 0.9
        self.weights = os.path.join("model", "24_10_23_yolov8x_no_aug_iou_0.7.pt")
    