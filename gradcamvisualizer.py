import torch
from torch.autograd import Function
import cv2
import numpy as np
from network import EfficientNet
from torchvision import transforms
from PIL import Image
import csv
import random
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.gradient = None
        self.model.eval()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output):
            self.feature_map = output.detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input_image):
        return self.model(input_image)

    def backward(self, target_class):
        if self.gradient is None:
            raise RuntimeError("Gradient is not available. Make sure to call forward() before backward().")

        gradients = torch.mean(self.gradient, dim=(2, 3))
        target_feature_map = self.feature_map
        weights = torch.sum(gradients * target_feature_map, dim=1)
        weights = torch.relu(weights)

        # Perform global average pooling on the weights
        weights = torch.mean(weights, dim=(1, 2))

        # Expand dimensions for compatibility with the feature map size
        weights = weights.unsqueeze(1).unsqueeze(2)

        # Obtain the weighted combination of the feature maps
        grad_cam = torch.sum(weights * target_feature_map, dim=0)

        # Apply ReLU to eliminate negative values
        grad_cam = torch.relu(grad_cam)

        # Normalize the grad_cam values between 0 and 1
        grad_cam = grad_cam - torch.min(grad_cam)
        grad_cam = grad_cam / torch.max(grad_cam)

        return grad_cam

    def generate_heatmap(self, grad_cam):
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam.cpu()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() / 255
        return heatmap

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        image = np.array(image)
        overlay = np.uint8(255 * heatmap.cpu().numpy()).transpose(1, 2, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return output


model = EfficientNet().to(DEVICE)
# model.load_state_dict(torch.load('/mnt/media/wiseyak/Chest_XRays/saved_model/EfficientNet_1_25.pth', map_location=DEVICE))
model.eval()

csv_file = 'Datasets/multilabel_classification/test.csv'
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    rows = list(reader)
    random_row = random.choice(rows)

# Get the image path from the first column of the random row
image_path = random_row[0]



# Load the image and preprocess it
image = Image.open(f'/home/optimus/Downloads/Dataset/ChestXRays/NIH/images/{image_path}').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Choose the target layer for Grad-CAM
target_layer = model.classficationLayer

# Create the Grad-CAM object
grad_cam = GradCAM(model, target_layer)


# Forward pass through the model
features, output = grad_cam.forward(input_tensor)

# Apply Grad-CAM for each target class with probability > 0.7
threshold = 0.5
for class_idx, prob in enumerate(output.squeeze()):
    if prob > threshold:
        # Convert class_idx to one-hot encoded tensor
        target_class = torch.eye(len(output.squeeze()))[class_idx].unsqueeze(0).to(DEVICE)

        # Backward pass to obtain Grad-CAM
        grad_cam.backward(features)

        # Generate the heatmap
        heatmap = grad_cam.generate_heatmap(grad_cam)

        # Overlay the heatmap on the original image
        overlay = grad_cam.overlay_heatmap(image, heatmap)

        # Save the visualization images
        heatmap.save(f'heatmap_class_{class_idx}.jpg')
        overlay.save(f'overlay_class_{class_idx}.jpg')
