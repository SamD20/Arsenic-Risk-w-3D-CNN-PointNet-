import torch
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
import numpy as np

BATCH_SIZE = 256
NUM_WORKERS = 12
KEEP_WORKERS = True
RANDOM_SEED = 42
EPOCH_SIZE = 300000 #wells per epoch
VALIDATION_SIZE = 300000 #wells per validation
RISK_CLASSES = [10,50]
# <=10 = Low Risk, 10 to <=50 = Medium Risk, >50 = High Risk
rng = np.random.default_rng(RANDOM_SEED)

def collate_fn(batch):

    voxels = torch.from_numpy(np.stack([item["voxel"] for item in batch])).float()
    points = [torch.tensor(item["points"],dtype=torch.float32) for item in batch]
    labels = torch.tensor([item["label"] for item in batch],dtype=torch.float32)
    risk = torch.tensor([item["risk"] for item in batch],dtype=torch.long)

    return {
        "voxel": voxels,
        "points": points,
        "label": labels,
        "risk": risk
    }

def get_dataloader(dataset,batch_size=BATCH_SIZE,workers=NUM_WORKERS):
    val_indexes = rng.choice(len(dataset),VALIDATION_SIZE,replace=False)
    sampler = SubsetRandomSampler(val_indexes)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=KEEP_WORKERS,
        prefetch_factor=4
    )

    return loader

def get_validation_dataloader(dataset, batch_size=BATCH_SIZE, workers=NUM_WORKERS):
    val_indexes = rng.choice(len(dataset),VALIDATION_SIZE,replace=False)
    sampler = SubsetRandomSampler(val_indexes)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=KEEP_WORKERS,
        prefetch_factor=4
    )