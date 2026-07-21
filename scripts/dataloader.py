import torch
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
import numpy as np

BATCH_SIZE = 32
NUM_WORKERS = 12
RANDOM_SEED = 42
EPOCH_SIZE = 12000 #wells per epoch
VALIDATION_SIZE = 12000 #wells per validation
RISK_CLASSES = [10,50]
# <=10 = Low Risk, 10 to <=50 = Medium Risk, >50 = High Risk

class BalancedRiskSampler(Sampler):
    def __init__(self, dataset, epoch_size):
        arsenic = dataset.Arsenic

        LOW = RISK_CLASSES[0]
        HIGH = RISK_CLASSES[1]

        self.low = np.where(arsenic <= LOW)[0]
        self.medium = np.where((arsenic > LOW) & (arsenic <= HIGH))[0]
        self.high = np.where(arsenic > HIGH)[0]

        self.samples_per_class = epoch_size // 3
        self.length = self.samples_per_class * 3

    def __iter__(self):
        low = np.random.choice(
            self.low,
            self.length // 3,
            replace=True
        )

        medium = np.random.choice(
            self.medium,
            self.length // 3,
            replace=True
        )

        high = np.random.choice(
            self.high,
            self.length // 3,
            replace=True
        )

        indexes = np.concatenate([low,medium,high])

        np.random.shuffle(indexes)
        return iter(indexes.tolist())

    def __len__(self):
        return self.length

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
    sampler = BalancedRiskSampler(dataset, EPOCH_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    return loader

def get_validation_dataloader(dataset, batch_size=BATCH_SIZE, workers=NUM_WORKERS):
    rng = np.random.default_rng(RANDOM_SEED)
    val_indexes = rng.choice(len(dataset),VALIDATION_SIZE,replace=False)
    sampler = SubsetRandomSampler(val_indexes)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )