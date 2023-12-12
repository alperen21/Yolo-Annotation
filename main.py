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
from config import Config
from iou import calculate_iou
from input import get_images, get_image_name
import os

def write_detection(img_url: str, detection: 'list[float]') -> None:
    img_name = get_image_name(img_url)
    txt_url = os.path.join("boxes", img_name + ".txt")
    with open(txt_url, "a") as f:
        f.write("0 " + " ".join([str(x) for x in detection]) + "\n")

def main():
    config = Config()
    model = YOLO(config.model_config, config.weights)
    model.conf = 0.01  # Confidence threshold
    for image in get_images():
        detections = model(image.img_url)
        for detection in detections.xyxy[0]:
            box = detection[0:4]
            iou = calculate_iou(box, image.bounding_boxes)
            if iou < config.iou_threshold:
                write_detection(image.img_url, detection)

if __name__ == "__main__":
    main()