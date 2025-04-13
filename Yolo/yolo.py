# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt based on your GPU
# model.train(
#     data='data.yaml',
#     epochs=50,
#     imgsz=640, # multiple of 32, or 256
#     batch=16,
#     workers=1  
# )
# metrics = model.val()
# results = model('/home/sahau24/csc790project/DL/data/VisDrone/images/val/0000001_02999_d_0000005.jpg')
# results.show()



import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# ------------------------------
# Step 1: Load Pretrained YOLOv8 Model
# ------------------------------
model = YOLO('yolov8n.pt')  # Use yolov8s.pt or yolov8m.pt if you prefer

# ------------------------------
# Step 2: Train the Model
# ------------------------------
model.train(
    data='data.yaml',      # points to YAML config
    epochs=50,
    imgsz=640,
    batch=16,
    workers=1
)

# ------------------------------
# Step 3: Evaluate Model on Val Set
# ------------------------------
metrics = model.val()

# ------------------------------
# Step 4: Save Per-Class Metrics to CSV
# ------------------------------
names = model.names
results = metrics.box

df = pd.DataFrame({
    'Class ID': list(range(len(results.precision))),
    'Class Name': [names[i] for i in range(len(results.precision))],
    'Precision': results.precision,
    'Recall': results.recall,
    'mAP50': results.map50,
    'mAP50-95': results.map
})
df.round(3).to_csv("val_per_class_metrics.csv", index=False)
print("Saved: val_per_class_metrics.csv")

# ------------------------------
# Step 5: Plot and Show Confusion Matrix
# ------------------------------
metrics.plot(confusion_matrix=True)

# Attempt to locate and display the confusion matrix
val_folders = sorted([d for d in os.listdir("runs/val") if os.path.isdir(os.path.join("runs/val", d))], reverse=True)
for folder in val_folders:
    path = os.path.join("/home/sahau24/csc790project/DL/runs/val", folder, "confusion_matrix.png")
    if os.path.exists(path):
        img = Image.open(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.show()
        break

# ------------------------------
# Step 6: Predict on All Validation Images
# ------------------------------
val_results = model.predict(
    source='/home/sahau24/csc790project/DL/data/VisDrone/images/val',
    imgsz=256,
    conf=0.25,
    save=True,
    save_txt=True
)

# ------------------------------
# Step 7: Save Predictions to CSV
# ------------------------------
with open("val_predictions.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'class_id', 'confidence', 'x_center', 'y_center', 'width', 'height'])

    for r in val_results:
        image_name = os.path.basename(r.path)
        for box in r.boxes.data.tolist():
            x_center, y_center, width, height, conf, class_id = box[:6]
            writer.writerow([image_name, int(class_id), round(conf, 4), x_center, y_center, width, height])

print("Saved: val_predictions.csv")
