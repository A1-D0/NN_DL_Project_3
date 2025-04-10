# main.py
# -------------------------
# Entry Point for Object Detection and Classification
# -------------------------

import os
import utils
import yolo_mod as yolo
import resnet_mod as resnet


# Constants
DATASET_DIR = './data/VisDrone/'
TEST_IMAGE = os.path.join(DATASET_DIR, 'images/0000001_00000_d_0000001.jpg')


def main():
    # Display the sample image
    utils.show_image(TEST_IMAGE, "Sample Image")

    # YOLOv3 Inference
    print("\n--- YOLOv3 Results ---")
    yolov3_model = yolo.load_yolov3_model()
    yolov3_results = yolo.run_yolov3_inference(yolov3_model, TEST_IMAGE)

    # ResNet50 Inference
    print("\n--- ResNet50 Results ---")
    resnet_model = resnet.load_resnet50_model()
    resnet_results = resnet.run_resnet50_inference(resnet_model, TEST_IMAGE)
    for idx, (cls, label, prob) in enumerate(resnet_results):
        print(f"{idx+1}. {label}: {prob:.4f}")


if __name__ == '__main__':
    main()
