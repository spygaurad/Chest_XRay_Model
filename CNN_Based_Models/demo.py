import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify
from optimAUROC import infer_a_sample
import cv2

app = Flask(__name__)

def infer_image(image_path):
    image = Image.open(image_path)
    class_activation_map, possible_diagnoses = infer_a_sample(image)
    return class_activation_map, possible_diagnoses


@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        class_activation_map, possible_diagnoses = infer_image(image_file)
        # You can process the results as needed here
        return jsonify({'class_activation_map': class_activation_map, 'possible_diagnoses': possible_diagnoses})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
