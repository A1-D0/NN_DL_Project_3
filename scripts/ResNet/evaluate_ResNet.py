'''
Description:
How to Run: 
'''

import os
import json
import csv
import torch
import argparse
import torchvision
import visdrone_to_coco
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor


def show_prediction(model, image_path, threshold=0.5, device="cuda", label_map=None):
    '''
    Run prediction on a single image and display results with bounding boxes and labels.
    
    :param model: Pretrained object detection model
    :param image_path: Path to input image
    :param threshold: Confidence threshold to display boxes
    :param device: 'cuda' or 'cpu'
    '''
    if label_map is None:
        print("Label map is not provided. Please provide a label map.")
        return None

    # load and prepare image
    img = Image.open(image_path).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # inference
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)[0]

    # setup plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        label_name = label_map.get(label.item(), str(label.item()))
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        ))
        ax.text(x1, y1 - 5, f'{label_name} ({score:.2f})',
                color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')
    plt.title("Predictions")
    plt.show()



def load_model(device: torch.device):
    '''
    Load a pre-trained Faster R-CNN model and move it to the specified device.
    
    :param device: Device to load the model onto ('cpu' or 'cuda').
    :return: Torch model on specified device.
    '''
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

def prepare_image(img_path: str) -> torch.Tensor:
    '''
    Load and preprocess an image into a PyTorch tensor.

    :param img_path: Path to the image file.
    :return: Tensor representing the image.
    '''
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def run_inference_on_dataset(model, image_dir: str, coco_gt: COCO, device: torch.device, test_size: int=0) -> list:
    '''
    Run object detection inference on the dataset using specified device.

    :param model: Pre-trained detection model.
    :param image_dir: Directory containing images.
    :param coco_gt: COCO object for referencing image IDs.
    :param device: Device to perform inference on ('cpu' or 'cuda').
    :param test_size: Number of images to test.
    :return: List of predictions in COCO detection format.
    '''
    results = []

    for img_info in tqdm(coco_gt.dataset['images'], desc="Running Inference"):
        img_path = os.path.join(image_dir, img_info['file_name'])
        image_tensor = prepare_image(img_path).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(image_tensor)[0]

        for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1
            results.append({
                "image_id": img_info['id'],
                "category_id": label.item(),
                "bbox": [x1, y1, width, height],
                "score": score.item()
            })
        
        if len(results) >= test_size and test_size > 0:
            break

    return results

def evaluate_coco(coco_gt: COCO, coco_dt_json: str, save_csv_path: str = "converted/eval_summary.csv") -> None:
    '''
    Evaluate predictions using COCO metrics and save results to CSV.

    :param coco_gt: COCO object containing ground-truth annotations.
    :param coco_dt_json: Path to the prediction results in COCO format.
    :param save_csv_path: Path to save the evaluation results as a CSV.
    '''
    coco_dt = coco_gt.loadRes(coco_dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # save summarized metrics to CSV
    metric_names = [
        "AP@[0.50:0.95]",
        "AP@0.50",
        "AP@0.75",
        "AP (small)",
        "AP (medium)",
        "AP (large)",
        "AR@[0.50:0.95]",
        "AR (small)",
        "AR (medium)",
        "AR (large)"
    ]

    with open(save_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for name, value in zip(metric_names, coco_eval.stats):
            writer.writerow([name, round(value, 4)])

    print(f"Evaluation summary saved to: {save_csv_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet on VisDrone dataset")
    parser.add_argument('-test_size', type=int, required=False, default=0, help="Test size (number of images to processes) (Optional)")
    args = parser.parse_args()
    test_size = args.test_size

    if test_size > 0: print(f"Test size: {test_size}")

    # set paths
    VisDrone2019_dir = os.path.join(os.pardir, os.pardir, 'data', 'VisDrone2019-DET-train')
    image_dir = os.path.join(VisDrone2019_dir, 'images')
    anno_dir = os.path.join(VisDrone2019_dir, 'annotations')

    resnet_data_dir = os.path.join(os.pardir, os.pardir, 'data', 'ResNet_data')
    os.makedirs(resnet_data_dir, exist_ok=True)
    annotation_json = os.path.join(resnet_data_dir, 'visdrone_train_coco.json')

    resnet_output_dir = os.path.join(os.pardir, os.pardir, 'output', 'ResNet')
    os.makedirs(resnet_output_dir, exist_ok=True)
    output_json = os.path.join(resnet_output_dir, 'predictions.json')
    save_results = os.path.join(resnet_output_dir, 'evaluation_results.csv')

    # convert VisDrone annotations to COCO
    # visdrone_to_coco.convert_visdrone_to_coco(
    #     image_dir=image_dir,
    #     anno_dir=anno_dir,
    #     output_json=annotation_json
    # )

    # check CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

    # load annotations and model
    coco_gt = COCO(annotation_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # enable GPU use if available
    model = load_model(device)

    # test one example
    label_map = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor", 10: "others"
    }
    show_prediction(model, os.path.join(image_dir, '0000002_00005_d_0000014.jpg'), threshold=0.5, device=device, label_map=label_map)

    # run inference and save predictions
    predictions = run_inference_on_dataset(model, image_dir, coco_gt, device, test_size)

    # print five predictions
    print("First five predictions:")
    for pred in predictions[:5]:
        print(pred)

    exit(3)

    # save predictions to JSON
    with open(output_json, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_json}")

    # evaluate and save results
    evaluate_coco(coco_gt, output_json, save_csv_path=save_results)

    exit(0)

if __name__ == "__main__":
    main()
