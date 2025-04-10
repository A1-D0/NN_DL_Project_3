# utils.py
# -------------------------
# Shared Utility Functions
# -------------------------

import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np


def load_image_for_resnet(img_path):
    """
    Load and preprocess an image for ResNet50.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x)


def show_image(img_path, title="Image"):
    """
    Display an image using OpenCV and Matplotlib.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
