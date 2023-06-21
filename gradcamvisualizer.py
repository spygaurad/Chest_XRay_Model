import torch
from torch.autograd import Function
import cv2
import numpy as np
from network import EfficientNet


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
        gradients = torch.mean(self.gradient, dim=(2, 3))[0]
        target_feature_map = self.feature_map[0]
        weights = torch.sum(gradients * target_feature_map, dim=0)
        weights = torch.relu(weights)

        # Perform global average pooling on the weights
        weights = torch.mean(weights, dim=(1, 2))

        # Expand dimensions for compatibility with the feature map size
        weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Obtain the weighted combination of the feature maps
        grad_cam = torch.sum(weights * target_feature_map, dim=1).squeeze(0)

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


# Usage example
model = EfficientNet().to(DEVICE)
model.load_state_dict(torch.load('saved_model/EfficientNet_11.pth', map_location=DEVICE))

# Load the image and preprocess it
image = PilImage.open(f'/home/optimus/Downloads/Dataset/ChestXRays/NIH/images/{image_path}').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Create the Grad-CAM object
grad_cam = GradCAM(model, model.effnet)

# Forward pass through the model
output = grad_cam.forward(input_tensor)

# Backward pass to obtain Grad-CAM
grad_cam.backward(target_class=output.argmax())

# Generate the heatmap
heatmap = grad_cam.generate_heatmap(grad_cam)

# Overlay the heatmap on the original image
overlay = grad_cam.overlay_heatmap(image, heatmap)

# Save the visualization images
heatmap.save('heatmap.jpg')
overlay.save('overlay.jpg')
