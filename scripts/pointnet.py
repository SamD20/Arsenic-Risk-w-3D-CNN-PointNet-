import torch
import torch.nn as nn

class PointNetHead(nn.Module):

    def __init__(self,input_features=6,embedding_size=256):
        super().__init__()
        self.embedding_size = embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(input_features,64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64,128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128,embedding_size),
            nn.ReLU()
        )


    def forward(self,pointclouds):
        
        embeddings = []

        for points in pointclouds:
            points = points.float()
            if points.shape[0] == 0:
                embeddings.append(
                    torch.zeros(
                        self.embedding_size * 3,
                        device=points.device
                    )
                )
                continue

            x = self.mlp(points)
            x_max = torch.max(x, dim=0)[0]
            x_mean = torch.mean(x, dim=0)

            if points.shape[0] == 1:
                x_std = torch.zeros_like(x[0])
            else:
                x_std = torch.std(x, dim=0, unbiased=False)
                
            x = torch.cat([x_max,x_mean,x_std],dim=0)
            
            embeddings.append(x)

        return torch.stack(
            embeddings
        )