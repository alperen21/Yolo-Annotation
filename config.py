import os

class Config:
    def __init__(self) -> None:
        self.iou_threshold = 0.5
        self.weights = os.path.join("weights", "yolov5s.pt")
    