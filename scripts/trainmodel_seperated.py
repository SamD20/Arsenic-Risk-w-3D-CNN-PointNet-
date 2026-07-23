import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from dataset import ArsenicDataset
from dataloader import get_dataloader,get_validation_dataloader,NUM_WORKERS,BATCH_SIZE
from cnn3d import CNN
from pointnet import PointNetHead

EPOCHS=100
LR=5e-5
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CLASS_WEIGHT=0.1

choice=input("Pick Model:\n1. 3D CNN\n2. PointNet\n> ").strip()

dataset=ArsenicDataset()
train_loader=get_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)
val_loader=get_validation_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)

backbone=CNN(dataset.rasters,extra_channels=11) if choice=="1" else PointNetHead(input_features=15,embedding_size=256)

def ordinal_to_class(prob):

    gt10 = prob[:,0] > 0.5
    gt50 = prob[:,1] > 0.5

    pred = torch.zeros_like(prob[:,0]).long()

    pred[gt10 & ~gt50] = 1
    pred[gt50] = 2

    return pred

class Model(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.backbone=b
        f=256 if choice=="1" else 1024
        self.regression=nn.Linear(f,1)
        self.classification=nn.Linear(f,2)

    def forward(self,batch):
        if choice=="1":
            feat=self.backbone(batch["voxel"].to(DEVICE,non_blocking=True))
        else:
            feat=self.backbone([p.to(DEVICE,non_blocking=True) for p in batch["points"]])
        return {
            "arsenic":self.regression(feat).squeeze(1),
            "risk":self.classification(feat)
        }

model=Model(backbone).to(DEVICE)

reg_loss=nn.MSELoss()
cls_loss = nn.BCEWithLogitsLoss()

opt=optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scaler=torch.amp.GradScaler("cuda")

best_f1=-1
best=None

for epoch in range(EPOCHS):
    model.train()

    for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{EPOCHS}"):

        y=batch["label"].to(DEVICE,non_blocking=True)
        r=batch["ordinal"].to(DEVICE,non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            out=model(batch)
            loss=CLASS_WEIGHT*reg_loss(out["arsenic"],y)+cls_loss(out["risk"],r)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.eval()
    pred,true,rpred,rtrue=[],[],[],[]

    with torch.no_grad():
        for batch in val_loader:
            y=batch["label"].to(DEVICE,non_blocking=True)
            r=batch["risk"].to(DEVICE,non_blocking=True)

            with torch.amp.autocast("cuda"):
                out=model(batch)

            pred.extend(out["arsenic"].float().cpu().numpy())
            true.extend(y.cpu().numpy())
            rpred.extend(ordinal_to_class(prob).cpu().numpy())
            rtrue.extend(r.cpu().numpy())

    pred=np.array(pred)
    true=np.array(true)

    rmse=np.sqrt(mean_squared_error(true,pred))
    mae=mean_absolute_error(true,pred)
    r2=r2_score(true,pred)
    pear=np.corrcoef(true,pred)[0,1]

    acc=accuracy_score(rtrue,rpred)
    prec=precision_score(rtrue,rpred,average="macro",zero_division=0)
    rec=recall_score(rtrue,rpred,average="macro",zero_division=0)
    f1=f1_score(rtrue,rpred,average="macro",zero_division=0)

    print(f"Epoch {epoch+1}: F1={f1:.4f} Acc={acc:.4f} RMSE={rmse:.4f}")

    if f1>best_f1:
        best_f1=f1
        best=(epoch+1,rmse,mae,pear,r2,acc,prec,rec,f1,confusion_matrix(rtrue,rpred))

e,rmse,mae,pear,r2,acc,prec,rec,f1,cm=best

print(f"""
Best Epoch {e}

RMSE {rmse:.4f}
MAE {mae:.4f}
Pearson {pear:.4f}
R2 {r2:.4f}

Accuracy {acc:.4f}
Precision {prec:.4f}
Recall {rec:.4f}
F1 {f1:.4f}

{cm}
""")