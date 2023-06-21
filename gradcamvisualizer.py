import torch
from torchvision import transforms
from PIL import Image
import csv
import random
import torch.nn.functional as F

from network import EfficientNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EfficientNet().to(DEVICE)
model.load_state_dict(torch.load('mnt/media/wiseyak/Chest_XRays/saved_model/EfficientNet_1_25.pth', map_location=torch.device(DEVICE)))


csv_file = '/home/optimus/Downloads/Dataset/ChestXRays/NIH/test.csv'
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    rows = list(reader)
    random_row = random.choice(rows)

# Get the image path from the first column of the random row
image_path = random_row[0]


# Load the image
image = Image.open(f'/home/optimus/Downloads/Dataset/ChestXRays/NIH/images/{image_path}').convert('RGB')

# Forward pass
model.eval()
with torch.no_grad():
    features, output = model(input_batch)


# Calculate gradients
grads = torch.autograd.grad(output, features)[0]


# Global average pooling of the gradients
weights = torch.mean(grads, dim=(2, 3))

# Resize the weights to match the feature map size
weights = weights[:, :, None, None]
weights = F.interpolate(weights, size=(input_batch.size(2), input_batch.size(3)), mode='bilinear', align_corners=False)

# Normalize the weights
weights = F.relu(weights)
weights /= torch.max(weights)


# Compute the weighted combination of the features
gradcam = torch.sum(features * weights, dim=1, keepdim=True)

# Apply ReLU to focus on the positive contributions
gradcam = F.relu(gradcam)

# Normalize the Grad-CAM
gradcam /= torch.max(gradcam)

# Convert Grad-CAM to an RGB image
gradcam = gradcam.repeat(1, 3, 1, 1)
gradcam = gradcam.detach().cpu().numpy()[0]
gradcam = Image.fromarray((gradcam * 255).astype('uint8'))


# Resize the original image to Grad-CAM size
image = image.resize((gradcam.size[1], gradcam.size[0]))

# Convert the image to RGBA
image_rgba = image.convert('RGBA')

# Convert the Grad-CAM to RGBA with transparency
gradcam_rgba = gradcam.convert('RGBA')

# Blend the images using alpha compositing
overlay = Image.alpha_composite(image_rgba, gradcam_rgba)


gradcam.save('gradcam.jpg')
overlay.save('overlay.jpg')
