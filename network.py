from torchvision import models, transforms
import torch.nn as nn
import timm
import torch

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.preConv1 = nn.Conv2d(1, 3, 3, 1, padding='same')
        self.preConv2 = nn.Conv2d(3, 3, 3, 1, padding='same')
        self.effnet = timm.create_model('tf_efficientnetv2_b0', num_classes=64, pretrained=True)
        self.fc1 = nn.Linear(64, 32)
        self.classficationLayer = nn.Linear(32, 15)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.preConv2(self.relu(self.preConv1(x))))
        features = self.effnet(x)
        x = self.fc1(features)
        return  features, self.sigmoid(self.classficationLayer(x))

# inp = torch.rand((1, 3, 256, 256))
# model = EfficientNet()
# out = model(inp)
# print(out.shape)