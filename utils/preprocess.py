import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced

def vessel_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    added = cv2.add(gray, tophat)
    subtracted = cv2.subtract(added, blackhat)
    _, vessels = cv2.threshold(subtracted, 15, 255, cv2.THRESH_BINARY)
    return vessels

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clahe_img = apply_clahe(image)
    vessels = vessel_segmentation(clahe_img)
    vessels_colored = cv2.cvtColor(vessels, cv2.COLOR_GRAY2RGB)
    combined = cv2.addWeighted(clahe_img, 0.8, vessels_colored, 0.2, 0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor_img = transform(combined)
    return tensor_img
