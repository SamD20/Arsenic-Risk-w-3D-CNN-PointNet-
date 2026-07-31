import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

BATCH_SIZE = 256
NUM_WORKERS = 6
KEEP_WORKERS = True
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

def collate_fn(batch):
    return {
        "voxel": torch.from_numpy(np.stack([x["voxel"] for x in batch])).float(),
        "points": [torch.tensor(x["points"], dtype=torch.float32) for x in batch],
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.float32),
        "risk": torch.tensor([x["risk"] for x in batch], dtype=torch.long),
    }

def get_dataloaders(dataset, batch_size=BATCH_SIZE, workers=NUM_WORKERS):
    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.permutation(len(dataset))

    train_size = int(len(dataset) * 0.8)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=KEEP_WORKERS,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=KEEP_WORKERS,
        prefetch_factor=2,
    )

    return train_loader, val_loader