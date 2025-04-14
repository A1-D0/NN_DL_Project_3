'''
Description: This script reads the detection (prediction) results from a json file, labels images with bounding boxes, and saves the labeled images to a specified directory.
It uses the COCO format for the predictions and the VisDrone dataset for the images.
How to Run: python label_images.py -conf_thresh n (where n is the confidence threshold between 0.0 and 1.0)
'''

import os
import json
import argparse

from PIL import Image, ImageDraw, ImageFont

def load_predictions_and_images(pred_json_path: str, gt_json_path: str) -> (list, dict):
    '''
    Load the predicted detections and ground truth image metadata from COCO-formatted JSON files.

    :param pred_json_path: Path to the predictions JSON file.
    :param gt_json_path: Path to the ground truth COCO JSON file.
    :return: A tuple containing:
             - predictions: a list of predicted detection dictionaries.
             - image_id_to_file: a dictionary mapping image_id to file_name.
    '''
    with open(pred_json_path, "r") as f:
        predictions = json.load(f)
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    # build a mapping from image_id to file name
    image_id_to_file = {img["id"]: img["file_name"] for img in gt_data.get("images", [])}
    return predictions, image_id_to_file

def draw_predictions_on_image(image_path: str, predictions: list, image_id: int,
                              label_map: dict, confidence_threshold: float = 0.5) -> Image.Image:
    '''
    Draw bounding boxes, labels, and confidence scores on a single image based on predictions.

    :param image_path: Path to the image file.
    :param predictions: List of prediction dictionaries in COCO format.
    :param image_id: The image ID to filter predictions.
    :param label_map: A dictionary mapping category IDs to class names.
    :param confidence_threshold: Minimum detection confidence to draw the prediction.
    :return: A PIL Image object with overlaid predictions.
    '''
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try: # attempt to load a TrueType font; fallback to default if unavailable.
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for pred in predictions:
        if pred["image_id"] != image_id: continue
        if pred["score"] < confidence_threshold: continue

        # get bounding box and other prediction values
        x, y, w, h = pred["bbox"]
        category_id = pred["category_id"]
        score = pred["score"]
        label = label_map.get(category_id, str(category_id))

        # draw the bounding box (red rectangle)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # prepare text (label and confidence score)
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # draw a filled rectangle behind the text for contrast
        draw.rectangle([x, y - text_height, x + text_width, y], fill="red")
        draw.text((x, y - text_height), text, fill="white", font=font)

    return image

def label_all_images(predictions: list, image_id_to_file: dict, image_dir: str,
                     output_dir: str, label_map: dict,
                     confidence_threshold: float = 0.5) -> None:
    '''
    Label all images with predicted bounding boxes and save the resulting images in a separate directory.
    This function leaves the original images intact.

    :param predictions: List of predictions in COCO format.
    :param image_id_to_file: Dictionary mapping image_id to file_name.
    :param image_dir: Directory containing the original images.
    :param output_dir: Directory where the labeled images will be saved.
    :param label_map: Dictionary mapping category IDs to class names.
    :param confidence_threshold: Minimum confidence required to display a detection.
    '''
    os.makedirs(output_dir, exist_ok=True)
    for image_id, file_name in image_id_to_file.items():
        image_path = os.path.join(image_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        labeled_img = draw_predictions_on_image(
            image_path=image_path,
            predictions=predictions,
            image_id=image_id,
            label_map=label_map,
            confidence_threshold=confidence_threshold
        )
        # save the labeled image into the output directory
        labeled_img.save(output_path)
        print(f"Labeled image saved to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Label images with predictions.")
    parser.add_argument("-conf_thresh", type=float, default=0.5, help="Confidence threshold for displaying predictions (0.0-1.0).")
    args = parser.parse_args()

    confidence_threshold = args.conf_thresh

    if not (confidence_threshold <= 1.0 and confidence_threshold >= 0.0): 
        print("Confidence threshold must be between 0.0 and 1.0.")
        exit(1)

    # define the paths for predictions, ground truth, and image directories
    pred_json = os.path.join(os.pardir, os.pardir, "output", "ResNet", "predictions.json")
    gt_json = os.path.join(os.pardir, os.pardir, "data", "ResNet_data", "visdrone_train_coco.json")
    image_dir = os.path.join(os.pardir, os.pardir, "data", "VisDrone2019-DET-train", "images")
    
    output_dir = os.path.join(os.pardir, os.pardir, "output", "ResNet", f"labeled_images_{confidence_threshold}")
    os.makedirs(output_dir, exist_ok=True)
    
    # define the label mapping (update as needed)
    label_map = {
        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
        5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
        # , 10: "others"
    }
    
    # load the predictions and image file names
    predictions, image_id_to_file = load_predictions_and_images(pred_json, gt_json)
    
    # process each image to draw detections, then save the labeled image to the output directory
    label_all_images(predictions, image_id_to_file, image_dir, output_dir, label_map, confidence_threshold=confidence_threshold)
    
    print("Images have been labeled and saved.")

    exit(0)

if __name__ == "__main__":
    main()
