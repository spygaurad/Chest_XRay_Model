# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import csv
# import random
# from network import EfficientNet

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# if DEVICE == "cuda":
#     torch.cuda.empty_cache()

# large_file_dir = 'Datasets/'

# # Initialize the model and load the weights
# model = EfficientNet().to(DEVICE)
# model.load_state_dict(torch.load(f'{large_file_dir}/saved_model/EfficientNet_1_SampleDataset.pth', map_location=DEVICE))
# model.eval()

# # Path to the CSV file containing image paths
# csv_file = 'Datasets/multilabel_modified/out_test.csv'

# with open(csv_file, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row
#     rows = list(reader)
#     random_row = random.choice(rows)

# # Get the image path from the first column of the random row
# image_path = random_row[0]
# image_path = 'image7843.jpg'
# print(image_path)

# # Load the image and preprocess it
# image = Image.open(f'{large_file_dir}/multilabel_modified/images/{image_path}').convert('RGB')
# preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
# input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# input_tensor.requires_grad_()

# # Perform forward pass and get the output predictions
# features, output = model(input_tensor)
# print(output)
# # Choose the target class index for visualization
# target_class = torch.argmax(output, dim=1)


# pred = target_class.item()
# # original = torch.argmax(label, )
# # Calculate the gradients using guided backpropagation
# model.zero_grad()
# output[0, target_class].backward(retain_graph=True)

# # Get the gradients of the input tensor
# gradients = input_tensor.grad.squeeze().cpu().detach().numpy()

# # Convert the gradients to grayscale
# grayscale_gradients = np.sum(gradients, axis=0)  # Convert RGB to grayscale

# # Normalize the gradients
# grayscale_gradients -= np.min(grayscale_gradients)
# grayscale_gradients /= np.max(grayscale_gradients)
# grayscale_gradients = np.uint8(grayscale_gradients * 255)

# # Apply the color map to the grayscale gradients
# heatmap = cv2.applyColorMap(grayscale_gradients, cv2.COLORMAP_COOL)


# input_array = input_tensor.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
# # Apply the heatmap to the original image
# # gradcam = cv2.addWeighted(cv2.cvtColor(np.uint8(255 * input_array), cv2.COLOR_RGB2BGR), 0.7, heatmap, 0.3, 0)

# # # Display the guided backpropagation visualization
# # plt.imshow(gradcam)
# # plt.axis('off')
# # plt.show()

# # Apply the heatmap to the original image
# gradcam = cv2.addWeighted(cv2.cvtColor(np.uint8(255 * input_array), cv2.COLOR_RGB2BGR), 1, heatmap, 0.3, 0)
# print(pred)
# # Display the guided backpropagation visualization
# cv2.imshow("Guided Backpropagation", gradcam)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import random
from network import EfficientNet



# Your existing code
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()

large_file_dir = 'Datasets/'

# Initialize the model and load the weights
model = EfficientNet().to(DEVICE)
model.load_state_dict(torch.load(f'{large_file_dir}/saved_model/EfficientNet_1_SampleDataset.pth', map_location=DEVICE))
model.eval()

# Path to the CSV file containing image paths
csv_file = 'Datasets/multilabel_modified/out_test.csv'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    rows = list(reader)
    random_row = random.choice(rows)

# Get the image path from the first column of the random row
image_path = random_row[0]
print(image_path)

# Load the image and preprocess it
image = Image.open(f'{large_file_dir}/multilabel_modified/images/{image_path}').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Perform forward pass and get the output predictions
features, output = model(input_tensor)

# Choose the target class (class with the highest prediction score)
target_class = torch.argmax(output).item()

# Calculate the gradients for the target class
model.zero_grad()
output[0, target_class].backward(retain_graph=True)

# Get the gradients and activations of the last convolutional layer
gradients = model.effnet.conv_head.weight.grad
activations = features[0]

# Calculate the weights using global average pooling of the gradients
weights = torch.mean(gradients, dim=(2, 3))

# Perform the weighted sum of the activations
weighted_activations = torch.einsum('ij,ijk->ik', weights, activations)

# Normalize the weighted activations
normalized_activations = torch.relu(weighted_activations)

# Upsample the normalized activations to the size of the input image
upsampled_activations = torch.nn.functional.interpolate(normalized_activations.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)

# Normalize the upsampled activations
normalized_upsampled_activations = (upsampled_activations - torch.min(upsampled_activations)) / (torch.max(upsampled_activations) - torch.min(upsampled_activations))

# Convert the normalized upsampled activations to a numpy array
heatmap = normalized_upsampled_activations.squeeze().cpu().detach().numpy()

# Apply the heatmap to the original image
image_array = np.array(image)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
gradcam = cv2.addWeighted(heatmap, 0.5, image_array, 0.5, 0)

# Display the Grad-CAM visualization
plt.imshow(gradcam)
plt.axis('off')
plt.show()