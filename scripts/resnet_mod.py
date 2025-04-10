# resnet_module.py
# -------------------------
# ResNet50 Inference Module
# -------------------------

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
import numpy as np


def load_resnet50_model():
    """
    Load pretrained ResNet50 model.
    """
    return ResNet50(weights='imagenet')


def load_image_for_resnet(img_path):
    """
    Load and preprocess an image for ResNet50.
    """
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def run_resnet50_inference(model, img_path):
    """
    Run ResNet50 classification on a single image.
    Returns top-3 predictions.
    """
    x = load_image_for_resnet(img_path)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded
