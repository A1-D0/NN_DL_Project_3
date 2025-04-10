# main.py
# -------------------------
# Object Detection and Classification Experiment
# -------------------------
#
# Experiment with pre-trained YOLOv3 and ResNet models on the VisDrone dataset.
# - Tune hyperparameters and decide on performance metrics.
# - Compare YOLOv3 and ResNet50:
#     * Hyperparameter settings
#     * Performance metrics (e.g., mAP for YOLO, accuracy for ResNet)
#     * Visualize ResNet feature maps
#     * Run inference on personal images with YOLO
#     * Provide result tables and visualizations to support findings


import os
import yolo_mod as yolo
import resnet_mod as resnet
import utils

# Constants
DATASET_DIR = './data/VisDrone2019-DET-train/images/'
OUTPUT_DIR = './output/yolo_predictions/train/'
CSV_OUTPUT = './output/yolo_results_train.csv'
TEST_IMAGE = os.path.join(DATASET_DIR, '0000001_00000_d_0000001.jpg')  # Example file


def main():
    # Load models
    yolov3_model = yolo.load_yolov3_model()
    resnet_model = resnet.load_resnet50_model()

    # Run YOLOv3 batch inference on VisDrone dataset
    print("Running YOLOv3 inference on VisDrone dataset...")
    results_list = yolo.run_yolov3_batch(
        model=yolov3_model,
        image_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        conf=0.25,
        iou=0.45
    )
    print(f"Saving YOLOv3 results to {CSV_OUTPUT}...")
    yolo.save_yolo_predictions(results_list, CSV_OUTPUT)

    # Run ResNet50 classification on sample VisDrone image
    print("\nRunning ResNet50 classification on sample VisDrone image...")
    utils.show_image(TEST_IMAGE, "Sample Image")
    predictions = resnet.run_resnet50_inference(resnet_model, TEST_IMAGE)
    for idx, (cls, label, prob) in enumerate(predictions):
        print(f"{idx + 1}. {label}: {prob:.4f}")

    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
