import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for deepfake detection model
    """
    # Read image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize and normalize
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    
    # Expand dimensions for model input (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array