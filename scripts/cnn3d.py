import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, raster_info, extra_channels=0):
        super().__init__()

        self.embeddings = nn.ModuleDict()
        self.raster_names = list(raster_info.keys())
        self.embed_names = {}

        c = extra_channels

        for n, i in raster_info.items():
            if i["isEmbedded"]:
                key = n.replace(".", "_")
                self.embed_names[n] = key

                d = i["EmbeddingSize"]

                self.embeddings[key] = nn.Embedding(
                    i["classes"],
                    d,
                    padding_idx=0
                )

                c += d

            else:
                c += 1


        self.features = nn.Sequential(

            nn.Conv3d(c,24,3,padding=1),
            nn.GroupNorm(8,24),
            nn.ReLU(),

            nn.Conv3d(24,48,3,padding=1),
            nn.GroupNorm(8,48),
            nn.ReLU(),

            nn.MaxPool3d(2),


            nn.Conv3d(48,96,3,padding=1),
            nn.GroupNorm(8,96),
            nn.ReLU(),

            nn.Conv3d(96,192,3,padding=1),
            nn.GroupNorm(8,192),
            nn.ReLU()
        )


        self.pool = nn.AdaptiveAvgPool3d(1)


        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192,256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


    def forward(self,x):

        ch=[]

        for i,n in enumerate(self.raster_names):

            if n in self.embed_names:

                e=self.embeddings[
                    self.embed_names[n]
                ](x[:,i].long())

                ch.append(
                    e.permute(0,4,1,2,3)
                )

            else:
                ch.append(
                    x[:,i:i+1]
                )


        if x.shape[1] > len(self.raster_names):
            ch.append(
                x[:,len(self.raster_names):]
            )


        x=torch.cat(ch,1)

        x=self.features(x)

        x=self.pool(x)

        return self.embedding(x)