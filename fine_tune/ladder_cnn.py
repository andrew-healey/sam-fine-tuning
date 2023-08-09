import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

class CNN_SAM(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_encoder = resnet18(pretrained=True)
    
    def forward(self, x):
        cnn_out = self.CNN_encoder.conv1(x)
        cnn_out = self.CNN_encoder.bn1(cnn_out)
        cnn_out = self.CNN_encoder.relu(cnn_out)
        cnn_out = self.CNN_encoder.maxpool(cnn_out)

        cnn_out = self.CNN_encoder.layer1(cnn_out)
        cnn_out = self.CNN_encoder.layer2(cnn_out)
        cnn_out = self.CNN_encoder.layer3(cnn_out)

        return cnn_out