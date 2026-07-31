import torch
import torch.nn as nn
from cnn3d import CNN
from pointnet import PointNetHead

class MultiTaskModel(nn.Module):
    def __init__(self,raster_info):
        super().__init__()

        self.cnn=CNN(dataset.raster_channels,extra_channels=17)

        self.pointnet=PointNetHead(
            input_features=15,
            embedding_size=256
        )

        self.fusion=nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.regression_head=nn.Linear(256,1)
        self.classification_head=nn.Linear(256,3)

    def forward(self,voxel,points):
        cnn_features=self.cnn(voxel)
        point_features=self.pointnet(points)

        features=self.fusion(
            torch.cat([cnn_features,point_features],dim=1)
        )

        return {
            "arsenic":self.regression_head(features).squeeze(1),
            "risk":self.classification_head(features),
            "features":features
        }