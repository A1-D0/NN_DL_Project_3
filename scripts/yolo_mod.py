# yolo_module.py
# -------------------------
# YOLOv3 Inference Module
# -------------------------

from ultralytics import YOLO


def load_yolov3_model():
    """
    Load pretrained YOLOv3 model.
    """
    return YOLO('yolov3.pt')


def run_yolov3_inference(model, img_path):
    """
    Run YOLOv3 inference on a single image.
    Displays the image with detections.
    Returns the result object.
    """
    results = model(img_path)
    results[0].show()
    return results[0]
