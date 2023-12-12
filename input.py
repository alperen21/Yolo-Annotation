# Load the model
# Change the confidence threshold
# For each image:
    # Detect again
    # For each detection:
        #   Get the box coordinates
        #   Calculate the IoU with the txt file
        #   If IoU < threshold for all boxes in the file:
        #    Write the detection to the new txt file


import cv2 
import os
from config import config
from ultralytics import YOLO
import pathlib

class Image:
    def __init__(self, img_url: str, bounding_boxes: 'list[float]') -> None:
        self.img_url = img_url
        self.bounding_boxes = bounding_boxes

def get_image_name(img_url: str) -> str:
    base_name = os.path.basename(img_url)  # Get the filename with extension
    img_name = os.path.splitext(base_name)[0]  # Split the extension and get the filename without extension
    return img_name

def get_bounding_boxes(img_url: str) -> 'list[float]':
    img_name = get_image_name(img_url)
    txt_url = os.path.join("boxes", img_name + ".txt")
    with open(txt_url, "r") as f:
        lines = f.readlines()
        bounding_boxes = [float(line.split()[1]) for line in lines]
        return bounding_boxes

def get_images() -> 'list[Image]':
    for root, _, files in os.walk(os.path.join("input", "images")):
        for file in files:
            if file.endswith(".jpg"):
                img_url = os.path.join(root, file)
                bounding_boxes = get_bounding_boxes(img_url)
                yield Image(img_url, bounding_boxes)

