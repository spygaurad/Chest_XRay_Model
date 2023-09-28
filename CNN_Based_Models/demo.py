import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify
from optimAUROC import infer_a_sample
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

def infer_image(image):
    image = Image.open(image)
    class_activation_map, possible_diagnoses, elapsed_time = infer_a_sample(image)
    return class_activation_map, possible_diagnoses, elapsed_time


@app.route('/infer', methods=['POST'])
def infer():

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        class_activation_map, possible_diagnoses, elapsed_time = infer_image(image_file)
        image_buffer = BytesIO()
        class_activation_map.save(image_buffer, format='PNG')
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        print(possible_diagnoses)
        
        response_data = {
            'time_taken': elapsed_time,
            'possible_diagnoses': possible_diagnoses,
            'class_activation_map': image_base64,
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
