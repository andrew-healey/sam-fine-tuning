import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from segment_anything.utils.transforms import ResizeLongestSide

class CNN_SAM(nn.Module):
    def __init__(self,resize_before_cnn: bool = True, feat_resolution: int = 64):
        super().__init__()

        self.CNN_encoder = resnet18(pretrained=False)
        self.resize_before_cnn = resize_before_cnn
        self.feat_resolution = feat_resolution
        self.resize_longest_side = ResizeLongestSide(self.feat_resolution)



    
    def forward(self, image: torch.Tensor, resized_image: torch.Tensor):

        curr_image = resized_image if self.resize_before_cnn else image
        # print("shape",curr_image.shape)

        cnn_out = self.CNN_encoder.conv1(curr_image)
        cnn_out = self.CNN_encoder.bn1(cnn_out)
        cnn_out = self.CNN_encoder.relu(cnn_out)
        cnn_out = self.CNN_encoder.maxpool(cnn_out)

        cnn_out = self.CNN_encoder.layer1(cnn_out)
        cnn_out = self.CNN_encoder.layer2(cnn_out)
        cnn_out = self.CNN_encoder.layer3(cnn_out)

        if not self.resize_before_cnn:
            cnn_out = self.resize_longest_side.apply_image_torch(cnn_out)
            cnn_out = preprocess_feature_map(cnn_out, self.feat_resolution)

        return cnn_out

# auto-resizing logic
# resize the longest side of the feature map to 256
# pad the other side with black on the right/bottom

from torch.nn import functional as F

def preprocess_feature_map(x: torch.Tensor, side_length: int) -> torch.Tensor:
    """Pad feature map to a square input."""
    # Normalize colors

    # Pad
    h, w = x.shape[-2:]
    padh = side_length - h
    padw = side_length - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
