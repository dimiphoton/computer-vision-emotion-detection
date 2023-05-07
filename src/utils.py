import os
import cv2
import numpy as np
from src.config import IMAGE_SIZE

def load_image(path):
    """
    Load an image from a given path and preprocess it.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == "__main__":
    print("Utility functions loaded.")
