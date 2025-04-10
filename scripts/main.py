# main.py
# -------------------------
# Entry Point for Object Detection and Classification
# -------------------------

import os
from utils import show_image
from yolo_mod import load_yolov3_model, run_yolov3_inference
from resnet_mod import load_resnet50_model, run_resnet50_inference


# Constants
DATASET_DIR = './data/VisDrone/'
TEST_IMAGE = os.path.join(DATASET_DIR, 'images/0000001_00000_d_0000001.jpg')


def main():
    # Display the sample image
    show_image(TEST_IMAGE, "Sample Image")

    # YOLOv3 Inference
    print("\n--- YOLOv3 Results ---")
    yolov3_model = load_yolov3_model()
    yolov3_results = run_yolov3_inference(yolov3_model, TEST_IMAGE)

    # ResNet50 Inference
    print("\n--- ResNet50 Results ---")
    resnet_model = load_resnet50_model()
    resnet_results = run_resnet50_inference(resnet_model, TEST_IMAGE)
    for idx, (cls, label, prob) in enumerate(resnet_results):
        print(f"{idx+1}. {label}: {prob:.4f}")


if __name__ == '__main__':
    main()
