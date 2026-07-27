import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, raster_channels, extra_channels):
        super().__init__()

        total_channels = raster_channels + extra_channels

        self.features = nn.Sequential(

            nn.Conv3d(total_channels, 24, 3, padding=1),
            nn.GroupNorm(8, 24),
            nn.ReLU(),

            nn.Conv3d(24, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(),

            # Only reduce horizontal resolution
            # Keep depth intact
            nn.MaxPool3d((1, 2, 2)),


            nn.Conv3d(48, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),

            nn.Conv3d(96, 192, 3, padding=1),
            nn.GroupNorm(8, 192),
            nn.ReLU(),

            # Another XY reduction, still preserve Z
            nn.MaxPool3d((1, 2, 2))
        )


        # Collapse only XY, keep depth information
        self.pool = nn.AdaptiveAvgPool3d((5, 1, 1))


        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


    def forward(self, x):
        x = self.features(x)

        # [B,192,5,1,1]
        x = self.pool(x)

        x = self.embedding(x)

        return x