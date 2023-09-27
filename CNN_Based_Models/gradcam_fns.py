import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def gradcam_pcam(model, x, target_class):
    """
    Generate GradCAM and PCAM heatmaps for a given image.

    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input image tensor.
        target_class (int): Index of the target class.

    Returns:
        torch.Tensor: GradCAM heatmap.
        torch.Tensor: PCAM heatmap.
    """
    model.eval()
    logits, features, pcams = model(x)
    
    # Create a one-hot tensor for the target class
    one_hot = torch.zeros_like(logits)
    one_hot[:, target_class] = 1
    
    # Backpropagate gradients with respect to the target class
    logits.backward(gradient=one_hot, retain_graph=True)
    gradients = model.features[0].weight.grad
    
    # Calculate the pooled gradients
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    
    # Multiply the feature maps by the pooled gradients
    for i in range(pooled_gradients.shape[1]):
        features[:, i, :, :] *= pooled_gradients[0, i]
    
    # Calculate GradCAM heatmap
    heatmap = torch.mean(features, dim=1).unsqueeze(0)
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap, pcams, logits



def overlay_heatmap(image, heatmap, class_labels):
    """
    Overlay a heatmap on top of an image and add class label.

    Args:
        image (PIL.Image): Input image.
        heatmap (torch.Tensor): Heatmap to overlay.
        class_labels (str): Class label.

    Returns:
        PIL.Image: Image with overlayed heatmap and label.
    """
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    
    # Resize heatmap to match the dimensions of the input image
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    # Convert the input image to RGB mode
    image = image.convert("RGB")
    
    # Create a custom colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list('custom_colormap', [(0, 'blue'), (0.5, 'green'), (1, 'red')])
    
    # Normalize the heatmap values to map to colors
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # Apply the colormap to the normalized heatmap
    colored_heatmap = cmap(normalized_heatmap)
    
    # Convert the colored heatmap to an RGB image
    colored_heatmap_rgb = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    
    # Create an Image object from the RGB heatmap
    heatmap_image = Image.fromarray(colored_heatmap_rgb, mode='RGB')
    
    # Ensure that the heatmap and input image have the same size
    if heatmap_image.size != image.size:
        heatmap_image = heatmap_image.resize(image.size, Image.BILINEAR)
    
    # Blend the heatmap with the input image
    overlayed_image = Image.blend(image, heatmap_image, alpha=0.5)
    
    # Add class label to the image
    label_text = f"Class: {class_labels}"
    draw = ImageDraw.Draw(overlayed_image)
    text_width, text_height = draw.textsize(label_text)
    draw.rectangle([(0, overlayed_image.size[1] - text_height), (overlayed_image.size[0], overlayed_image.size[1])], fill="black")
    draw.text(((overlayed_image.size[0] - text_width) // 2, overlayed_image.size[1] - text_height), label_text, fill="white")

    return overlayed_image
