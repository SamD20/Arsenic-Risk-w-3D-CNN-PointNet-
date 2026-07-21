import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self,input_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_channels,32,kernel_size=3,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(128,256,3,padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool3d(
            (1,1,1)
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        return self.embedding(
            x
        )