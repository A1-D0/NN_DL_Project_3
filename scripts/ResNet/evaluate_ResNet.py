'''
Description:
'''

import os
import json
import csv
import torch
import torchvision
import visdrone_to_coco

from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm

def load_model() -> torchvision.models.detection.FasterRCNN:
    '''
    Load a pre-trained Faster R-CNN model with ResNet50-FPN backbone.

    :return: PyTorch Faster R-CNN model in evaluation mode.
    '''
    model = fasterrcnn_resnet50_fpn(pretrained=True)
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

def run_inference_on_dataset(model, image_dir: str, coco_gt: COCO) -> list:
    '''
    Run object detection inference on the dataset.

    :param model: Pre-trained detection model.
    :param image_dir: Directory containing VisDrone images.
    :param coco_gt: COCO ground-truth object for referencing image IDs.
    :return: List of predictions in COCO detection format.
    '''
    results = []
    for img_info in tqdm(coco_gt.dataset['images'], desc="Running Inference"):
        img_path = os.path.join(image_dir, img_info['file_name'])
        image_tensor = prepare_image(img_path).unsqueeze(0)
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
    visdrone_to_coco.convert_visdrone_to_coco(
        image_dir=image_dir,
        anno_dir=anno_dir,
        output_json=annotation_json
    )

    exit(1)

    # load annotations and model
    coco_gt = COCO(annotation_json)
    model = load_model()

    # run inference and save predictions
    predictions = run_inference_on_dataset(model, image_dir, coco_gt)

    # save predictions to JSON
    with open(output_json, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_json}")

    # evaluate and save results
    evaluate_coco(coco_gt, output_json, save_csv_path=save_results)

if __name__ == "__main__":
    main()
