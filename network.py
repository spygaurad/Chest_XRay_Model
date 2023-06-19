from torchvision import models, transforms
import torch.nn as nn
import timm
import torch

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = timm.create_model('tf_efficientnetv2_b0', num_classes=1024, pretrained=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.classficationLayer = nn.Linear(64, 16)
        # self.classIndexLayer = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.effnet(x)
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return  self.sigmoid(self.classficationLayer(x))

# inp = torch.rand((1, 3, 256, 256))
# model = ResNet50()
# out = model(inp)
# print(out.shape)