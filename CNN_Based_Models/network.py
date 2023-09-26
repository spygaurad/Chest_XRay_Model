# Define the model architecture
import torch
import torch.nn as nn
from torchvision.models import densenet121

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
        
        # Global Average Pooling (GAP)
        pooled_features = self.avgpool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Forward pass through the classifier
        logits = self.fc2(self.relu(self.fc1(pooled_features)))
        
        return logits, features