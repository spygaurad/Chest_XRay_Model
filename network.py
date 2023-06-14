from torchvision import models, transforms
import torch.nn as nn
import torch

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 64)
        self.classficationLayer = nn.linear(64, 10)
        # self.classIndexLayer = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return x

inp = torch.rand((1, 3, 256, 256))
model = ResNet50()
out = model(inp)
print(out.shape)