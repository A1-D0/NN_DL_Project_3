# yolo_module.py
# -------------------------
# YOLOv3 Inference Module
# -------------------------

import os
import cv2
import pandas as pd
from ultralytics import YOLO


def load_yolov3_model():
    """
    Load pretrained YOLOv3 model.
    """
    return YOLO('yolov3.pt')


def run_yolov3_inference(model, img_path, conf=0.25, iou=0.45, show=True):
    """
    Run YOLOv3 inference on a single image.
    Args:
        model: YOLOv3 model instance
        img_path: Path to the image
        conf: Confidence threshold
        iou: IoU threshold for NMS
        show: Whether to display the image with detections
    Returns:
        YOLO result object
    """
    results = model(img_path, conf=conf, iou=iou)
    if show:
        results[0].show()
    return results[0]


def run_yolov3_batch(model, image_dir, output_dir=None, conf=0.25, iou=0.45):
    """
    Run YOLOv3 inference on all images in a directory.
    Args:
        model: YOLOv3 model instance
        image_dir: Directory of input images
        output_dir: Optional directory to save images with predictions
        conf: Confidence threshold
        iou: IoU threshold for NMS
    Returns:
        List of result objects
    """
    results_list = []
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_dir, file_name)
            results = model(img_path, conf=conf, iou=iou)
            results_list.append((file_name, results[0]))

            if output_dir:
                save_path = os.path.join(output_dir, file_name)
                results[0].save(filename=save_path)

    return results_list


def save_yolo_predictions(results_list, output_csv):
    """
    Save YOLOv3 predictions to CSV.
    Args:
        results_list: List of (filename, result) tuples
        output_csv: Path to output CSV file
    """
    data = []
    for fname, result in results_list:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            data.append([fname, cls_id, conf, x1, y1, x2, y2])

    df = pd.DataFrame(data, columns=["filename", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    df.to_csv(output_csv, index=False)


def run_yolov3_on_custom_image(model, img_path, conf=0.25, iou=0.45, show=True):
    """
    Run YOLOv3 inference on a custom image.
    Args:
        model: YOLOv3 model instance
        img_path: Path to the image
        conf: Confidence threshold
        iou: IoU threshold
        show: Whether to display image with results
    Returns:
        Result object
    """
    return run_yolov3_inference(model, img_path, conf, iou, show)
