# Define the model architecture
import torch
import torch.nn as nn
from torchvision.models import densenet121
import torch.nn.functional as F

class DenseNetWithPCAM(nn.Module):
    def __init__(self, num_classes=5):
        super(DenseNetWithPCAM, self).__init__()
        self.features = densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 768, bias=True)
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Forward pass through the convolutional layers (features)
        features = self.features(x)
        
        # PCAM Pooling
        pcams = torch.zeros(features.size(0), features.size(1), features.size(2), features.size(3))

        for i in range(features.size(0)):
            for j in range(features.size(1)):
                pcams[i, j] = features[i, j] * torch.max(features[i, j])

        # Forward pass through the classifier
        pcams = pcams.view(pcams.size(0), -1)
        logits = self.fc2(self.relu(self.fc1(pcams)))
        
        return logits, features
