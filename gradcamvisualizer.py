import json
import torch
from torchvision import transforms
from PIL import Image as PilImage
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

from network import EfficientNet
#write code to visualize the grad-cam process here

img = Image(PilImage.open('../'))