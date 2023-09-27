
import torch.nn.functional as F

import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify
from optimAUROC import infer_a_sample
import cv2


# Define your inference function
def infer_image(image_path):

    image = Image.open(image_path)

    result_images = infer_a_sample(image)
    
    result_images = [Image.fromarray(img) for img in result_images]

    # Convert PIL images to numpy arrays for Gradio output
    result_images = [np.array(img) for img in result_images]
    
    return result_images

# infer_image("/home/wiseyak/saumya/Chest_XRay_Model/Imgs/view1_frontal.jpg")
infer_image("/home/optimus/Downloads/Datasets/CheXpert-v1.0-small/train/patient00007/study2/view1_frontal.jpg")