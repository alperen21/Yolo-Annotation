import os

class Configuration:
    def __init__(self) -> None:
        self.iou_threshold = 0.9
        self.weights = os.path.join("model", "24_10_23_yolov8x_no_aug_iou_0.7.pt")
        self.cell_types = ["RBC", "WBC", "PLT"]
        self.confidence_threshold = 0.01
    
    def set_prediction_names(self, names) -> None:
        self.name_to_idx_dict = dict()
        for idx, name in names.items():
            self.name_to_idx_dict[name] = idx
    
    def name_to_idx(self, name) -> int:
        return self.name_to_idx_dict[name]
    
    def is_prediction_included_celltypes(self, prediction) -> bool:
        return prediction in [self.name_to_idx(cell_type) for cell_type in self.cell_types]


    