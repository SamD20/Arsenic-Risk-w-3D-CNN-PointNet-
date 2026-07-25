import numpy as np
import matplotlib.pyplot as plt
import random
from dataset import ArsenicDataset
from dataloader import RISK_CLASSES

dataset = ArsenicDataset()

print("\nCNN channels:")
print(dataset.empty_tensor.shape)

def get_random_index(risk):
    indexes = []

    for i in range(len(dataset)):
        arsenic = dataset.Arsenic[i]

        if arsenic <= RISK_CLASSES[0]:
            r = 0
        elif arsenic <= RISK_CLASSES[1]:
            r = 1
        else:
            r = 2

        if r == risk:
            indexes.append(i)

    return random.choice(indexes)

samples = {
    "LOW": get_random_index(0),
    "MEDIUM": get_random_index(1),
    "HIGH": get_random_index(2)
}

print("\nSelected wells:")

for k,v in samples.items():
    print(
        k,
        "index:",
        v,
        "arsenic:",
        dataset.Arsenic[v],
        "depth:",
        dataset.Depth[v]
    )

cnn_samples = {}

for name,index in samples.items():

    x = dataset.cnnInput(index)

    print(
        name,
        "CNN input:",
        x.shape
    )

    cnn_samples[name] = x

raster_channels = dataset.raster_channels
channel_names = []

for i in range(raster_channels):
    channel_names.append(
        f"Raster {i}"
    )

extra_names = [
    "well count",
    "arsenic mean",
    "arsenic median",
    "arsenic p10",
    "arsenic p90",
    "arsenic p25",
    "arsenic p75",
    "arsenic p95",
    "IQR",
    "p95-p90",
    "p25-p10",
    "depth mean",
    "depth std",
    "confidence",
    "relative X",
    "relative Y",
    "absolute depth",
    "depth difference",
    "chem Fe",
    "chem Mn",
    "chem SO4",
    "chem Ca",
    "chem Mg",
    "chem Na",
    "chem Si",
    "chem P",
    "chem distance",
    "chem count"
]

channel_names.extend(extra_names)

def plot_channel(channel):

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(15,9)
    )

    fig.suptitle(
        f"Channel {channel}: {channel_names[channel]}",
        fontsize=16
    )

    for row,(name,x) in enumerate(cnn_samples.items()):
        volume = x[channel]
        for z in range(5):
            ax = axes[row,z]
            img = volume[:,:,z]

            im=ax.imshow(
                img,
                cmap="viridis"
            )

            ax.set_title(
                f"{name}\nLayer {z}"
            )
            ax.axis("off")

    plt.tight_layout()
    plt.show()

total_channels = dataset.empty_tensor.shape[0]

print(
    "\nTotal CNN channels:",
    total_channels
)

for c in range(total_channels):
    plot_channel(c)