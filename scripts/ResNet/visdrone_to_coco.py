'''
Description: Convert VisDrone annotations to COCO format for evaluate_ResNet.py.
How to Run: python visdrone_to_coco.py
'''
import os
import json
import glob

from PIL import Image
from typing import List, Dict

def convert_visdrone_to_coco(image_dir: str, anno_dir: str, output_json: str, category_map: List[Dict] = None)-> None:
    '''
    Convert VisDrone annotations to COCO format.

    :param image_dir: Directory containing images.
    :param anno_dir: Directory containing annotations.
    :param output_json: Output path for COCO JSON file.
    :param category_map: Mapping of category IDs to category names (default is VisDrone2019 10 classes).
    '''
    print(f"Converting VisDrone annotations to COCO format...")

    if category_map is None:
        category_map = [
            {"id": 0, "name": "pedestrian"},
            {"id": 1, "name": "people"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
            {"id": 4, "name": "van"},
            {"id": 5, "name": "truck"},
            {"id": 6, "name": "tricycle"},
            {"id": 7, "name": "awning-tricycle"},
            {"id": 8, "name": "bus"},
            {"id": 9, "name": "motor"}
            # {"id": 10, "name": "others"},
        ]

    category_ids = {cat["id"]: cat for cat in category_map}

    images = []
    annotations = []
    annotation_id = 1

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    for img_id, img_path in enumerate(image_files):
        file_name = os.path.basename(img_path)
        img = Image.open(img_path)
        width, height = img.size

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        anno_file = os.path.join(anno_dir, file_name.replace(".jpg", ".txt"))
        if not os.path.exists(anno_file):
            continue

        with open(anno_file) as f:
            for line in f:
                try:
                    vals = list(map(int, line.strip().split(',')[:8]))
                    x, y, w, h, _, cls_id, _, _ = vals
                    if cls_id not in category_ids:
                        continue
                    annotations.append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                except Exception as e:
                    print(f"Error parsing line in {anno_file}: {line}")
                    continue

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": category_map
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"COCO-style annotations saved to: {output_json}")