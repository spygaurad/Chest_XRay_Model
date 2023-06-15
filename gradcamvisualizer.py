import json
import torch
from torchvision import transforms
from PIL import Image as PilImage
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
import csv
import random 

from network import EfficientNet


# If available, move the input tensors to the GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load the pre-trained model
model = EfficientNet().to(DEVICE)
model.load_state_dict(torch.load('saved_model/EfficientNet_11.pth', map_location=torch.device(DEVICE)))


# Load the test.csv file and select a random row
csv_file = '/home/optimus/Downloads/Dataset/ChestXRays/NIH/test.csv'
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    rows = list(reader)
    random_row = random.choice(rows)

# Get the image path from the first column of the random row
image_path = random_row[0]

# Load the image
image = PilImage.open(f'/home/optimus/Downloads/Dataset/ChestXRays/NIH/images/{image_path}').convert('RGB')

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

input_batch = input_batch.to(DEVICE)

# Load the Grad-CAM explainer
explainer = GradCAM(model=model, target_layer=model.classficationLayer, preprocess_function=preprocess)

# Perform Grad-CAM
gradcam_map = explainer.explain(input_batch)

# Convert the Grad-CAM map to a heatmap image
heatmap = explainer.heatmap(gradcam_map)

# Overlay the heatmap on the original image
overlay = explainer.overlay_heatmap(image, heatmap)

# Save the visualization images
heatmap.save('heatmap.jpg')
overlay.save('overlay.jpg')
