'''
Description: Plot evaluation metrics from COCO-formatted annotations and predictions, including mAP, Precision-Recall curves, and confusion matrix.
How to Run: python plot_evaluations.py
'''
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from pycocotools.coco import COCO

def plot_eval_summary(eval_csv_path: str, save_path: str) -> None:
    '''
    Plot and save a bar chart of mAP/AR summary metrics.

    :param eval_csv_path: Path to eval_summary.csv
    :param save_path: Path to save the plot image
    '''
    df = pd.read_csv(eval_csv_path)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Metric"], df["Value"], color='skyblue')
    plt.xticks(rotation=45)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Overall Evaluation Summary (mAP/AR)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved evaluation summary plot to: {save_path}")

def plot_prf_vs_conf(prf_csv_path: str, save_path: str) -> None:
    '''
    Plot and save line plots for Precision, Recall, and F1 vs Confidence.

    :param prf_csv_path: Path to precision_recall_f1_vs_conf.csv
    :param save_path: Path to save the plot image
    '''
    df = pd.read_csv(prf_csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Confidence"], df["Precision"], label="Precision", marker='o')
    plt.plot(df["Confidence"], df["Recall"], label="Recall", marker='x')
    plt.plot(df["Confidence"], df["F1"], label="F1 Score", marker='s')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs Confidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved Precision, Recall, and F1 vs Confidence plot to: {save_path}")

def display_pr_curve(pr_curve_image_path: str) -> None:
    '''
    Load and show the pre-saved PR curve image.

    :param pr_curve_image_path: Path to precision_recall_curve.png
    '''
    img = Image.open(pr_curve_image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Precision-Recall Curve")
    plt.show()

def plot_per_class_ap(per_class_csv_path: str, save_path: str) -> None:
    '''
    Plot and save per-class AP as a bar chart.

    :param per_class_csv_path: Path to per_class_metrics.csv
    :param save_path: Path to save the plot image
    '''
    df = pd.read_csv(per_class_csv_path)
    plt.figure(figsize=(12, 6))
    plt.bar(df["Class"], df["AP@[0.50:0.95]"], color='salmon')
    plt.xlabel("Class")
    plt.ylabel("AP@[0.50:0.95]")
    plt.title("Per-Class Average Precision")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved per-class AP plot to: {save_path}")

def plot_normalize_confusion_matrix(cm_csv_path: str, save_path: str) -> None:
    '''
    Plot and save a normalized confusion matrix heatmap.

    :param cm_csv_path: Path to confusion_matrix.csv
    :param save_path: Path to save the plot image
    '''
    df = pd.read_csv(cm_csv_path, index_col=0)

    # normalize by row (true class)
    normalized_df = df.div(df.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_df, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved normalized confusion matrix heatmap to: {save_path}")

def plot_confusion_matrix(cm_csv_path: str, save_path: str) -> None:
    '''
    Plot and save a confusion matrix heatmap.

    :param cm_csv_path: Path to confusion_matrix.csv
    :param save_path: Path to save the plot image
    '''
    df = pd.read_csv(cm_csv_path, index_col=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved confusion matrix heatmap to: {save_path}")

def compute_confusion_matrix(coco_gt_path: str, dt_json_path: str, output_csv: str, iou_thresh: float = 0.5) -> None:
    '''
    Compute a confusion matrix from COCO-formatted ground truth and detection predictions,
    and save it as a CSV file.

    :param coco_gt_path: Path to COCO-formatted ground truth annotations.
    :param dt_json_path: Path to detection predictions in COCO format.
    :param output_csv: File path to save the computed confusion matrix CSV.
    :param iou_thresh: IoU threshold for matching a detection with a ground truth.
    '''
    if os.path.exists(output_csv):
        print(f"Confusion matrix CSV already exists at {output_csv}. Skipping computation.")
        return

    # load COCO ground truth and predictions
    gt = COCO(coco_gt_path)
    dt = gt.loadRes(dt_json_path)
    
    # get categories and prepare mappings
    cat_ids = gt.getCatIds()
    cats = gt.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    class_idx_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    num_classes = len(cat_ids)
    
    # init the confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    # for each image, compute IoU between GT and DT boxes
    for img_id in gt.getImgIds():
        gt_ann_ids = gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = dt.getAnnIds(imgIds=[img_id])
        gts = gt.loadAnns(gt_ann_ids)
        dts = dt.loadAnns(dt_ann_ids)
        
        if not gts or not dts:
            continue

        # convert bboxes from [x, y, w, h] to [x1, y1, x2, y2]
        def to_xyxy(bbox):
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        
        gt_boxes = [to_xyxy(gt_ann['bbox']) for gt_ann in gts]
        dt_boxes = [to_xyxy(dt_ann['bbox']) for dt_ann in dts]
        gt_ids = [gt_ann['category_id'] for gt_ann in gts]
        dt_ids = [dt_ann['category_id'] for dt_ann in dts]
        
        used_gt = set()
        for dt_idx, dt_box in enumerate(dt_boxes):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt: continue

                # compute intersection coordinates
                ixmin = max(dt_box[0], gt_box[0])
                iymin = max(dt_box[1], gt_box[1])
                ixmax = min(dt_box[2], gt_box[2])
                iymax = min(dt_box[3], gt_box[3])
                iw = max(ixmax - ixmin, 0)
                ih = max(iymax - iymin, 0)
                inter = iw * ih
                
                # compute union area
                dt_area = (dt_box[2]-dt_box[0]) * (dt_box[3]-dt_box[1])
                gt_area = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                union = dt_area + gt_area - inter
                iou = inter / union if union > 0 else 0
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0:
                used_gt.add(best_gt_idx)
                true_cls = gt_ids[best_gt_idx]
                pred_cls = dt_ids[dt_idx]


                if true_cls in class_idx_map and pred_cls in class_idx_map:
                    confusion[class_idx_map[true_cls], class_idx_map[pred_cls]] += 1
                else:
                    print(f"[Warning] Skipping unmatched category: GT={true_cls}, Pred={pred_cls}")
                    continue
                # confusion[class_idx_map[true_cls], class_idx_map[pred_cls]] += 1

    # save the confusion matrix to CSV
    df_cm = pd.DataFrame(confusion, index=cat_names, columns=cat_names)
    df_cm.to_csv(output_csv)
    print(f"Saved computed confusion matrix to: {output_csv}")


def main() -> None:

    output_dir = os.path.join(os.pardir, os.pardir, 'output', 'ResNet')
    
    # define file paths for each output file
    eval_summary_csv = os.path.join(output_dir, "eval_summary.csv")
    prf_csv = os.path.join(output_dir, "precision_recall_f1_vs_conf.csv")
    pr_curve_png = os.path.join(output_dir, "precision_recall_curve.png")
    per_class_csv = os.path.join(output_dir, "per_class_metrics.csv")
    confusion_csv = os.path.join(output_dir, "confusion_matrix.csv")
    
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # plot each output in its own figure
    print("Plotting evaluation summary metrics...")
    plot_eval_summary(eval_summary_csv, os.path.join(output_dir, "plots", "eval_summary.png"))
    
    print("Plotting Precision, Recall and F1 vs Confidence...")
    plot_prf_vs_conf(prf_csv, os.path.join(output_dir, "plots", "precision_recall_f1_vs_conf.png"))
    
    print("Displaying Precision-Recall Curve image...")
    display_pr_curve(pr_curve_png)
    
    print("Plotting per-class AP metrics...")
    plot_per_class_ap(per_class_csv, os.path.join(output_dir, "plots", "per_class_metrics.png"))

    print("Computing confusion matrix...")    
    compute_confusion_matrix(os.path.join(os.pardir, os.pardir, "data", "ResNet_data", "visdrone_train_coco.json"), 
                             os.path.join(output_dir, "predictions.json"), 
                             confusion_csv)
    print("Plotting confusion matrix as a heatmap...")
    plot_confusion_matrix(confusion_csv, os.path.join(output_dir, "plots", "confusion_matrix.png"))

    print("Plotting normalized confusion matrix as a heatmap...")
    plot_normalize_confusion_matrix(confusion_csv, os.path.join(output_dir, "plots", "normalized_confusion_matrix.png"))

    exit(0)

if __name__ == "__main__":
    main()
