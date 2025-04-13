## NN_DL_Project_3
* please note that due to LFS bandwidth and storage issues, the drone image and label data are ignored via .gitignore
for the scripts to work simply move the .zip files to data directory and then unzip *

# VisDrone Object Detection with YOLOv8

A deep learning pipeline to detect small, occluded, and densely packed objects in aerial footage using the [VisDrone](http://www.aiskyeye.com/) dataset and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

---

## 📁 Project Structure

visdrone-yolov8/
├── data/
│   └── VisDrone/     ## Dataset Format (YOLO)        
│       ├── images/
            ├── /train
            ├── /val
│       └── labels/
            ├── /train
            ├── /val
├── scripts/
│   ├── convert_annotations.py  ## VisDrone → YOLO ( As required )
│   ├── yolo.py
│   ├── evaluate.py
│   └── utils.py
├── jobs/
│   ├── train.slurm
├── runs/                  
├── results/
│   ├── confusion_matrix.png
│   ├── val_predictions.csv
│   └── val_per_class_metrics.csv
├── data.yaml

**Requirements before run:**

ultralytics
matplotlib
pandas
opencv-python

