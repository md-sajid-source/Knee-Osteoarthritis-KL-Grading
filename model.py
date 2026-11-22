import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        options = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
            'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
            'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT)
        }
        make_model, weights = options[backbone]
        self.model = make_model(weights=weights if pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)
