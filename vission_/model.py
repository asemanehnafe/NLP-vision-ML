import torch
import torch.nn as nn # basic building block for neural neteorks
import torchvision.models as models

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        num_of_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_of_features, 8)
        

    def forward(self, x):
        x = torch.tanh(self.model(x))
        return x