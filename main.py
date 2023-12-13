# Load the model
# Change the confidence threshold
# For each image:
    # Detect again
    # For each detection:
        #   Get the box coordinates
        #   Calculate the IoU with the txt file
        #   If IoU < threshold for all boxes in the file:
        #    Write the detection to the new txt file

from ultralytics import YOLO
from configuration import Configuration
from iou import calculate_max_iou
from input import get_images, get_image_name
import os
import torch

def write_detection(img_url: str, detection: torch.tensor, class_) -> None:
    tensor_2d_float = detection.float()
    detection = tensor_2d_float.tolist()

    img_name = get_image_name(img_url)
    txt_url = os.path.join("boxes", img_name + ".txt")
    with open(txt_url, "r") as f:
        lines = f.readlines()
        lines.append(f"{class_} " + " ".join([str(x) for x in detection]) + "\n")
    for idx, line in enumerate(lines):
        line = line.strip()
        line += "\n"
        lines[idx] = line
    with open(txt_url, "w") as f:
        f.writelines("".join(lines))
        

def main():
    print("starting...")
    config = Configuration()
    model = YOLO(config.weights)
    model.conf = 0.01  # Confidence threshold
    print("model loaded...")
    for image in get_images():
        print("image: ", image.img_url)
        detections = model(image.img_url, save=True)
        config.set_prediction_names(detections[0].names)
        for detection in detections:

            boxes = detection.boxes.xyxyn
            classes = detection.boxes.cls

            for class_, box in zip(classes, boxes):
                iou = calculate_max_iou(box, image.bounding_boxes)

                if not config.is_prediction_included_celltypes(class_):
                    print("predicted cell type is not included in the configurations")
                    continue
                else:
                    print("predicted cell type is included in the configurations")

                if iou < config.iou_threshold:
                    print("writing the detection...")
                    write_detection(image.img_url, box, class_)

if __name__ == "__main__":
    main()