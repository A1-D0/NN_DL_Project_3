'''
Description: Evaluate a pre-trained ResNet model (Faster R-CNN with ResNet50-FPN backbone) on the VisDrone dataset using COCO metrics.
This script runs inference, computes evaluation metrics (saving summary CSVs and plots), and (optionally) extracts feature maps from the backbone and saves them to a directory.
How to Run: 
    python evaluate_ResNet.py -test_size n -extract_fmaps (optional),
    where n is the number of images to process.
'''

import numpy as np
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

def extract_feature_maps(model, image_dir: str, fmaps_output_dir: str, device: torch.device, test_size: int=0) -> None:
    '''
    Extract feature maps from the FPN backbone of the Faster R-CNN model for each image
    and save the visualizations to the specified output directory.
    
    :param model: Pre-trained Faster R-CNN model.
    :param image_dir: Directory containing input images.
    :param fmaps_output_dir: Directory to save the extracted feature map images.
    :param device: Device to run the model on.
    :param test_size: Number of images to process (if 0, process all).
    '''
    os.makedirs(fmaps_output_dir, exist_ok=True)
    feature_maps = {}

    def fpn_hook(module, input, output):
        # output is a dict of feature maps from different FPN layers
        for key, fmap in output.items():
            feature_maps[key] = fmap.detach().cpu()

    hook_handle = model.backbone.register_forward_hook(fpn_hook)
    image_files = sorted(os.listdir(image_dir))
    processed = 0
    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(input_tensor)

        # save feature maps for this image
        for layer_key, fmap in feature_maps.items():
            fmap = fmap.squeeze(0)  # shape: [C, H, W]
            num_channels = min(6, fmap.shape[0])
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i in range(num_channels):
                axes[i].imshow(fmap[i], cmap='viridis')
                axes[i].axis("off")
                axes[i].set_title(f"Channel {i}")
            plt.suptitle(f"{file_name} - Layer {layer_key}")
            save_path = os.path.join(fmaps_output_dir, f"{os.path.splitext(file_name)[0]}_layer{layer_key}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved feature map: {save_path}")
        feature_maps.clear()
        processed += 1
        if test_size > 0 and processed >= test_size:
            break
    hook_handle.remove()

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

def save_summary_metrics(coco_eval: COCOeval, output_dir: str) -> None:
    '''
    Save summary metrics (mAP and AR) from COCOeval to a CSV file in the output directory.

    :param coco_eval: COCOeval object that has been evaluated and accumulated.
    :param output_dir: Directory where the summary CSV will be saved.
    '''
    summary_csv = os.path.join(output_dir, "eval_summary.csv")
    metric_names = [
        "AP@[0.50:0.95]", "AP@0.50", "AP@0.75", 
        "AP (small)", "AP (medium)", "AP (large)",
        "AR@[0.50:0.95]", "AR (small)", "AR (medium)", "AR (large)"
    ]
    with open(summary_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for name, value in zip(metric_names, coco_eval.stats):
            writer.writerow([name, round(value, 4)])
    print(f"Saved mAP/AR summary to {summary_csv}")

def save_precision_recall_f1_vs_conf(coco_eval: COCOeval, output_dir: str) -> None:
    '''
    Compute precision, recall, and F1 score versus confidence using COCOeval results,
    and save these values to a CSV file in the output directory.

    :param coco_eval: COCOeval object after evaluation.
    :param output_dir: Directory where the CSV file will be saved.
    '''
    # extract precision array: shape [T, R, K, A, M]
    precision = coco_eval.eval['precision']
    # use the first IoU threshold, first area range, first maxDet index.
    iou_thr_idx, area_idx, max_det_idx = 0, 0, 0
    precisions = precision[iou_thr_idx, :, :, area_idx, max_det_idx]  # shape: [R, K]
    # average precision over all classes (mean over axis 1 of recall thresholds)
    precision_avg = np.mean(precisions, axis=1)
    recall = coco_eval.params.recThrs  # array of recall thresholds (usually 101 values)
    # use "1 - recall" as a proxy for confidence (for plotting)
    confidence = 1 - recall
    f1_scores = 2 * (precision_avg * recall) / (precision_avg + recall + 1e-6)

    prf_csv = os.path.join(output_dir, "precision_recall_f1_vs_conf.csv")
    with open(prf_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Confidence", "Precision", "Recall", "F1"])
        for c, p, r, f1 in zip(confidence, precision_avg, recall, f1_scores):
            writer.writerow([round(c, 4), round(p, 4), round(r, 4), round(f1, 4)])
    print(f"Saved precision/recall/F1 vs confidence to {prf_csv}")

def save_pr_curve(coco_eval: COCOeval, output_dir: str) -> None:
    '''
    Generate a Precision-Recall curve from COCOeval results and save the plot as an image.

    :param coco_eval: COCOeval object after evaluation.
    :param output_dir: Directory where the PR curve image will be saved.
    '''
    precision = coco_eval.eval['precision']
    iou_thr_idx, area_idx, max_det_idx = 0, 0, 0
    precisions = precision[iou_thr_idx, :, :, area_idx, max_det_idx]  # shape: [R, K]
    precision_avg = np.mean(precisions, axis=1)
    recall = coco_eval.params.recThrs

    pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision_avg, label="PR Curve", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (IoU=0.50:0.95)")
    plt.grid(True)
    plt.savefig(pr_curve_path, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve image to {pr_curve_path}")

def save_per_class_metrics(coco_eval: COCOeval, coco_gt: COCO, output_dir: str) -> None:
    '''
    Compute per-class Average Precision (AP) at IoU=0.50:0.95 and save the results to a CSV file.

    :param coco_eval: COCOeval object after evaluation.
    :param coco_gt: COCO object containing ground-truth annotations.
    :param output_dir: Directory where the per-class metrics CSV will be saved.
    '''
    # retrieve class names from COCO ground truth
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [cat["name"] for cat in cats]
    num_classes = len(class_names)
    # extract precision array.
    precision = coco_eval.eval['precision']
    iou_thr_idx, area_idx, max_det_idx = 0, 0, 0
    per_class_ap = []
    for k in range(num_classes):
        class_prec = precision[iou_thr_idx, :, k, area_idx, max_det_idx]
        valid = class_prec[class_prec > -1]
        ap = np.mean(valid) if valid.size else float('nan')
        per_class_ap.append(ap)

    per_class_csv = os.path.join(output_dir, "per_class_metrics.csv")
    with open(per_class_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "AP@[0.50:0.95]"])
        for name, ap in zip(class_names, per_class_ap):
            writer.writerow([name, round(ap, 4)])
    print(f"Saved per-class AP to {per_class_csv}")

def save_confusion_matrix(coco_eval: COCOeval, coco_gt: COCO, coco_dt, output_dir: str) -> None:
    '''
    Compute a raw confusion matrix from ground truth and detected annotations, and save it to a CSV file.
    This function uses an IoU threshold of 0.5 to match predictions to ground truth.

    :param coco_eval: COCOeval object that has been evaluated.
    :param coco_gt: COCO object containing ground-truth annotations.
    :param coco_dt: COCO object containing detected annotations.
    :param output_dir: Directory where the confusion matrix CSV will be saved.
    '''
    # retrieve class names
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [cat["name"] for cat in cats]
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    img_ids = coco_gt.getImgIds()

    iou_threshold = 0.5
    for img_id in img_ids:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])
        gts = coco_gt.loadAnns(gt_ann_ids)
        dts = coco_dt.loadAnns(dt_ann_ids)
        if not gts or not dts:
            continue

        # compute IoUs for current image
        ious = coco_eval.computeIoU(imgId=img_id, catId=None)
        if ious is None or len(ious) == 0: continue
        if isinstance(ious, list): ious = ious[0]

        matched_gt = set()
        for dt_idx, dt in enumerate(dts):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = ious[dt_idx, gt_idx]
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0:
                gt_cls = gts[best_gt_idx]['category_id']
                dt_cls = dt['category_id']
                matched_gt.add(best_gt_idx)
                confusion[gt_cls][dt_cls] += 1
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(cm_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, row in enumerate(confusion):
            writer.writerow([class_names[i]] + list(row))
    print(f"Saved confusion matrix to {cm_path}")

def evaluate_coco(coco_gt: COCO, coco_dt_json: str, output_dir: str) -> None:
    '''
    Evaluate predictions using COCO metrics and save multiple outputs:
    - Summary mAP/AR metrics
    - Precision, Recall, and F1 vs Confidence data
    - Precision-Recall curve image
    - Per-class AP metrics
    - Confusion matrix data
    
    :param coco_gt: COCO object containing ground-truth annotations.
    :param coco_dt_json: Path to the detection results in COCO format.
    :param output_dir: Directory where all result files will be saved.
    '''
    os.makedirs(output_dir, exist_ok=True)
    print("Evaluating with COCOeval...")
    
    # load detection results
    coco_dt = coco_gt.loadRes(coco_dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # save each output
    save_summary_metrics(coco_eval, output_dir)
    save_precision_recall_f1_vs_conf(coco_eval, output_dir)
    save_pr_curve(coco_eval, output_dir)
    save_per_class_metrics(coco_eval, coco_gt, output_dir)
    save_confusion_matrix(coco_eval, coco_gt, coco_dt, output_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet on VisDrone dataset")
    parser.add_argument('-test_size', type=int, required=False, default=0, help="Test size (number of images to processes) (Optional)")
    parser.add_argument('-extract_fmaps', required=False, action='store_true', help="Extract feature maps from the model")
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

    # convert VisDrone annotations to COCO
    if os.path.exists(annotation_json): print(f"Annotation JSON already exists at {annotation_json}\tSkipping conversion.")
    else:
        print(f"Converting VisDrone annotations to COCO format...")
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

    # # run inference and save predictions
    # predictions = run_inference_on_dataset(model, image_dir, coco_gt, device, test_size)

    # # print five predictions
    # print("First five predictions:")
    # for pred in predictions[:5]:
    #     print(pred)

    # # save predictions to JSON
    # with open(output_json, 'w') as f:
    #     json.dump(predictions, f)
    # print(f"Saved predictions to {output_json}")

    # # evaluate and save results
    # coco_gt = COCO(annotation_json)
    # evaluate_coco(coco_gt, output_json, resnet_output_dir)


    if args.extract_fmaps:
        fmaps_output_dir = os.path.join(resnet_output_dir, "feature_maps")
        print("Extracting feature maps...")
        extract_feature_maps(model, image_dir, fmaps_output_dir, device, test_size)


    exit(0)

if __name__ == "__main__":
    main()
