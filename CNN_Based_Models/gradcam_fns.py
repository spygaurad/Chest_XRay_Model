import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw



def gradcam_pcam(model, x, target_class):
    model.eval()

    logits, features, pcams = model(x)

    one_hot = torch.zeros_like(logits)
    one_hot[:, target_class] = 1

    logits.backward(gradient=one_hot, retain_graph=True)

    gradients = model.features[0].weight.grad

    pooled_gradients = torch.mean(gradients, dim=[2, 3])

    for i in range(pooled_gradients.shape[1]):
        features[:, i, :, :] *= pooled_gradients[0, i]

    # Calculate the heatmap for the target class
    heatmap = torch.mean(features, dim=1).unsqueeze(0)
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap, pcams


def overlay_heatmap(image, heatmap, target_class, class_labels):

    heatmap_image = transforms.ToPILImage()(heatmap.squeeze().cpu())

    overlayed_image = Image.blend(image, heatmap_image, alpha=0.25)

    label_text = f"Class: {class_labels[target_class]}"
    draw = ImageDraw.Draw(overlayed_image)
    text_width, text_height = draw.textsize(label_text)
    draw.rectangle([(0, overlayed_image.size[1] - text_height), (overlayed_image.size[0], overlayed_image.size[1])], fill="black")
    draw.text(((overlayed_image.size[0] - text_width) // 2, overlayed_image.size[1] - text_height), label_text, fill="white")

    return overlayed_image