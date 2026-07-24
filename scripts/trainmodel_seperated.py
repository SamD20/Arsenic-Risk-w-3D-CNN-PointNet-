import torch,torch.nn as nn,torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from dataset import ArsenicDataset
from dataloader import get_dataloader,get_validation_dataloader,NUM_WORKERS,BATCH_SIZE
from cnn3d import CNN
from pointnet import PointNetHead

EPOCHS,LR,DEVICE=100,5e-5,"cuda" if torch.cuda.is_available() else "cpu"

choice=input("Pick Model:\n1. 3D CNN\n2. PointNet\n> ").strip()

dataset=ArsenicDataset()
train_loader=get_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)
val_loader=get_validation_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)

backbone=CNN(dataset.raster_channels,28) if choice=="1" else PointNetHead(15,256)

BOUNDARIES=torch.tensor(np.log1p([10,25,50,100]),device=DEVICE)

def band_target(y):
    return torch.bucketize(y,BOUNDARIES)

def ordinal_target(x):
    return (x.unsqueeze(1)>torch.arange(4,device=x.device)).float()

def gaussian_nll(m,v,y):
    v=torch.clamp(v,-3,3)
    return (((y-m)**2)/torch.exp(v)+v).mean()

class Model(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.backbone=b
        f=256 if choice=="1" else 1024
        self.reg=nn.Linear(f,2)
        self.ordinal=nn.Linear(f,4)
        self.risk=nn.Linear(f,3)

    def forward(self,b):
        x=self.backbone(b["voxel"].to(DEVICE,non_blocking=True)) if choice=="1" else self.backbone([p.to(DEVICE,non_blocking=True) for p in b["points"]])
        r=self.reg(x)
        return {"mean":r[:,0],"var":r[:,1],"ord":self.ordinal(x),"risk":self.risk(x)}

model=Model(backbone).to(DEVICE)
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scaler=torch.amp.GradScaler("cuda")
weights=torch.tensor([1,2.0,1.5],device=DEVICE)

best=-1
best_data=None

for epoch in range(EPOCHS):
    model.train()
    total=0

    for b in tqdm(train_loader,desc=f"Epoch {epoch+1}/{EPOCHS}"):
        y=b["label"].to(DEVICE,non_blocking=True)
        r=b["risk"].to(DEVICE,non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            o=model(b)
            reg=gaussian_nll(o["mean"],o["var"],y)
            ord_loss=nn.functional.binary_cross_entropy_with_logits(o["ord"],ordinal_target(band_target(y)))
            cls=nn.functional.cross_entropy(o["risk"],r,weight=weights)
            loss=.2*reg+.7*ord_loss+cls

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(opt)
        scaler.update()
        total+=loss.item()

    model.eval()
    pred,true,rp,rt=[],[],[],[]

    with torch.no_grad():
        for b in val_loader:
            y=b["label"].to(DEVICE,non_blocking=True)
            r=b["risk"].to(DEVICE,non_blocking=True)

            with torch.amp.autocast("cuda"):
                o=model(b)

            pred.extend(o["mean"].cpu().numpy())
            true.extend(y.cpu().numpy())
            rp.extend(torch.argmax(o["risk"],1).cpu().numpy())
            rt.extend(r.cpu().numpy())

    pred,true=np.array(pred),np.array(true)
    rmse=np.sqrt(mean_squared_error(np.expm1(true),np.expm1(pred)))
    f1=f1_score(rt,rp,average="macro")
    acc=accuracy_score(rt,rp)
    cm=confusion_matrix(rt,rp)

    print(f"Epoch {epoch+1}: Loss={total/len(train_loader):.4f} F1={f1:.4f} Acc={acc:.4f} RMSE={rmse:.2f}")
    print(cm)

    if f1>best:
        best=f1
        best_data=(epoch+1,rmse,cm)

print(f"""
Best Epoch {best_data[0]}
RMSE {best_data[1]:.2f}
F1 {best:.4f}

{best_data[2]}
""")