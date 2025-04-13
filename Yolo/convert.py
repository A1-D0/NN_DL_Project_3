import os
import cv2

def convert_visdrone_annotations(ann_dir, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".txt"):
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        img_file = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(img_path):
            print(f"Skipping {img_file} (image not found)")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image {img_path}")
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                x, y, bw, bh, score, class_id = map(int, parts[:6])
                if score != 1 or class_id >= 10:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                w_norm = bw / w
                h_norm = bh / h

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        out_path = os.path.join(out_dir, ann_file)
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"Converted {len(os.listdir(ann_dir))} files in {ann_dir} to YOLO format → {out_dir}")

# Apply conversion to both train and val
convert_visdrone_annotations(
    ann_dir="/home/sahau24/csc790project/DL/data/VisDrone/labels/train",
    img_dir="/home/sahau24/csc790project/DL/data/VisDrone/images/train",
    out_dir="/home/sahau24/csc790project/DL/data/VisDrone/labels/train_yolo"
)

convert_visdrone_annotations(
    ann_dir="/home/sahau24/csc790project/DL/data/VisDrone/labels/val",
    img_dir="/home/sahau24/csc790project/DL/data/VisDrone/images/val",
    out_dir="/home/sahau24/csc790project/DL/data/VisDrone/labels/val_yolo"
)
