import torch
import torch.nn as nn

from cnn3d import CNN
from pointnet import PointNetHead

class MultiTaskModel(nn.Module):

    def __init__(self,input_channels):
        super().__init__()

        self.cnn = CNN(input_channels)

        self.pointnet = PointNetHead(
            input_features=6,
            embedding_size=256
        )

        self.fusion = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.regression_head = nn.Linear(256,1)
        self.classification_head = nn.Linear(256,3)


    def forward(self,voxel,points):
        cnn_features = self.cnn(voxel)
        point_features = self.pointnet(points)

        features = torch.cat([cnn_features,point_features],dim=1)
        features = self.fusion(features)

        arsenic = self.regression_head(features)
        risk = self.classification_head(features)

        return {
            "arsenic": arsenic.squeeze(1),
            "risk": risk,
            "features": features
        }