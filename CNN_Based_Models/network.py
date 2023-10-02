# Define the model architecture
import torch
import torch.nn as nn
from torchvision.models import densenet121
import torch.nn.functional as F
import importlib


network_path = '/home/wiseyak/saumya/Chest_XRay_Model/region_guided_radiology_report_generator/src/full_model/generate_reports_for_images.py'
network_module = importlib.util.spec_from_file_location('generate_reports_for_images', network_path)
network = importlib.util.module_from_spec(network_module)
network_module.loader.exec_module(network)
get_model = network.get_model



device = "cuda" if torch.cuda.is_available() else "cpu"

class DenseNetWithPCAM(nn.Module):
    def __init__(self, num_classes=5):
        super(DenseNetWithPCAM, self).__init__()
        self.features = densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(50176, 768, bias=True)  # Adjust the input size here
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Forward pass through the convolutional layers (features)
        features = self.features(x)
        
        # PCAM Pooling
        pcams = torch.zeros(features.size(0), features.size(1), features.size(2), features.size(3))
        pcams = pcams.to(device)
        pcams = features * torch.max(features, dim=1, keepdim=True).values

        # Flatten pcams before passing to the fully connected layers
        pcams = pcams.view(pcams.size(0), -1)
        
        # Forward pass through the classifier
        logits = self.fc2(self.relu(self.fc1(pcams)))
        
        return logits, features, pcams


class ResNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet, self).__init__()
        checkpoint_path = "/home/wiseyak/saumya/Chest_XRay_Model/region_guided_radiology_report_generator/full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt"
        model = get_model(checkpoint_path)
        self.resnet = model.object_detector.backbone
        # for params in self.resnet.parameters():
        #     params.requires_grad = False
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(2048, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits, None, None

image = torch.rand(1, 1, 512, 512).to('cuda')
model = ResNet().to('cuda')
output = model(image)
print(output)
