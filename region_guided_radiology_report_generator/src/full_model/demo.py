import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from generate_reports_for_images import infer
import os

IMAGE_INPUT_SIZE = 512
mean = 0.471  # see get_transforms in src/dataset/compute_mean_std_dataset.py
std = 0.302

def get_image_tensor(image_path):
    # cv2.imread by default loads an image with 3 channels
    # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape (3056, 2544)

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    transform = val_test_transforms(image=image)
    image_transformed = transform["image"]  # shape (1, 512, 512)
    image_transformed_batch = image_transformed.unsqueeze(0)  # shape (1, 1, 512, 512)

    return image_transformed_batch


app = Flask(__name__)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'})

        image_file = request.files['image']

        # Create a temporary file to save the image
        temp_image_path = 'temp_image.jpg'
        image_file.save(temp_image_path)

        # Get the image tensor
        image_tensor = get_image_tensor(temp_image_path)

        # Call the infer function to generate the report
        generated_report, elapsed_time = infer(image_tensor)

        # Remove the temporary image file
        os.remove(temp_image_path)

        response = {
            'generated_report': generated_report,
            'time_taken': elapsed_time,
        }
        # Return the generated report as a JSON response
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
