# utils.py
# -------------------------
# Shared Utility Functions
# -------------------------

import cv2
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

def load_image_for_resnet(img_path):
    '''
    Load and preprocess an image for ResNet50.

    :param img_path: Path to the image file.
    :return: Preprocessed image ready for ResNet50.
    '''
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def show_image(img_path, title="Image"):
    """
    Display an image using OpenCV and Matplotlib.

    :param img_path: Path to the image file.
    :param title: Title for the displayed image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
