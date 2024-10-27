import torch
import torch.nn as nn
import torch.nn.functional as F

"""
simple cnn
rpn
roi pooling
fast r-cnn
faster r-cnn
"""

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class RPN(nn.Module):

    def __init__(self, num_regions):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        # TODO (oliver); check dimensions of output conv
        self.bbox_reg = nn.Linear(in_features=256, out_features=num_regions * 4)
        self.cls = nn.Linear(in_features=256, out_features=num_regions * 2)

    def forward(self, feature_map):
        feature_map = F.relu(self.conv(feature_map))
        bbox_reg = self.bbox_reg(feature_map)
        cls = self.cls(feature_map)
        return cls, bbox_reg


class ROIPooling(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, feature_map, proposals):
        pooled_regions = []

        # proposals [n_regions, x1, x2, y1, y2]
        for proposal in proposals:
            x1, x2, y1, y2 = proposal
            region = feature_map[..., x1:x2, y1:y2]
            pooled_region = self.pooling(region)
            pooled_regions.append(pooled_region)

        return pooled_regions


class FastRCNN(nn.Module):
    pass


class FasterRCNN(nn.Module):

    def __init__(self):
        self.cnn = SimpleCNN()
        self.rpn = RPN()
        self.roi_pooling = ROIPooling()

    def forward(self, x):
        feature_map = self.cnn(x)
        ... = self.rpn(feature_map)
