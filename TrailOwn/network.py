from torchvision import models, transforms
import torch.nn as nn
import timm
import torch

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = timm.create_model('tf_efficientnetv2_b0', num_classes=64, pretrained=True)
        self.convFeatures = torch.nn.Sequential(*list(self.effnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.classficationLayer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        convfeatures = self.convFeatures(x)
        x = self.effnet(x)
        x = self.fc1(x)
        return  convfeatures, self.sigmoid(self.classficationLayer(x))

# inp = torch.rand((1, 3, 256, 256))
# model = EfficientNet()
# out = model(inp)
# print(out.shape)