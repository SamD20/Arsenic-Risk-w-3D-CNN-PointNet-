import torch,torch.nn as nn,torch.optim as optim
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from dataset import ArsenicDataset
from dataloader import get_dataloader,get_validation_dataloader,NUM_WORKERS,BATCH_SIZE
from cnn3d import CNN
from pointnet import PointNetHead

EPOCHS,LR,DEVICE,CLASS_WEIGHT=100,5e-5,"cuda" if torch.cuda.is_available() else "cpu",0.2
LOG_LOW,LOG_HIGH=np.log1p(10),np.log1p(50)

choice=input("Pick Model:\n1. 3D CNN\n2. PointNet\n> ").strip()
dataset=ArsenicDataset()
train_loader=get_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)
val_loader=get_validation_dataloader(dataset,BATCH_SIZE,NUM_WORKERS)
backbone=CNN(dataset.raster_channels,extra_channels=18) if choice=="1" else PointNetHead(input_features=15,embedding_size=256)

def gaussian_nll(mean,log_var,target):
    log_var=torch.clamp(log_var,-3,3)
    return (((target-mean)**2)/torch.exp(log_var)+log_var).mean()

def gaussian_class_probability(mean, log_var):

    log_var = torch.clamp(log_var, -3, 3)

    std = torch.exp(0.5 * log_var) + 1e-3

    n = Normal(mean, std)

    low = torch.tensor(LOG_LOW, device=mean.device)
    high = torch.tensor(LOG_HIGH, device=mean.device)

    p0 = n.cdf(low)
    p1 = n.cdf(high) - p0
    p2 = 1 - n.cdf(high)

    return torch.stack([p0, p1, p2], 1)

class Model(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.backbone=b
        self.regression=nn.Linear(256 if choice=="1" else 1024,2)

    def forward(self,b):
        x=self.backbone(b["voxel"].to(DEVICE,non_blocking=True)) if choice=="1" else self.backbone([p.to(DEVICE,non_blocking=True) for p in b["points"]])
        y=self.regression(x)
        return {"mean":y[:,0],"log_var":y[:,1]}

model=Model(backbone).to(DEVICE)
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scaler=torch.amp.GradScaler("cuda")
weights=torch.tensor([1,1.7,1.2],device=DEVICE)

best_f1,best=-1,None

for epoch in range(EPOCHS):
    model.train();total=0

    for b in tqdm(train_loader,desc=f"Epoch {epoch+1}/{EPOCHS}"):
        y,r=b["label"].to(DEVICE,non_blocking=True),b["risk"].to(DEVICE,non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            o=model(b)
            reg=gaussian_nll(o["mean"],o["log_var"],y)
            p=gaussian_class_probability(o["mean"],o["log_var"])
            cls=-(weights[r]*torch.log(p[torch.arange(len(r),device=DEVICE),r]+1e-8)).mean()
            loss=CLASS_WEIGHT*reg+cls

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(opt);scaler.update()
        total+=loss.item()

    model.eval()
    pred,true,rpred,rtrue=[],[],[],[]

    with torch.no_grad():
        for b in val_loader:
            y,r=b["label"].to(DEVICE,non_blocking=True),b["risk"].to(DEVICE,non_blocking=True)
            with torch.amp.autocast("cuda"):
                o=model(b)

            pred.extend(o["mean"].float().cpu().numpy())
            true.extend(y.cpu().numpy())
            rpred.extend(torch.argmax(gaussian_class_probability(o["mean"],o["log_var"]),1).cpu().numpy())
            rtrue.extend(r.cpu().numpy())

    pred,true=np.array(pred),np.array(true)
    raw_pred,raw_true=np.expm1(pred),np.expm1(true)

    log_rmse=np.sqrt(mean_squared_error(true,pred))
    rmse=np.sqrt(mean_squared_error(raw_true,raw_pred))
    mae=mean_absolute_error(raw_true,raw_pred)
    r2=r2_score(true,pred)
    pear=np.corrcoef(true,pred)[0,1]

    acc=accuracy_score(rtrue,rpred)
    prec=precision_score(rtrue,rpred,average="macro",zero_division=0)
    rec=recall_score(rtrue,rpred,average="macro",zero_division=0)
    f1=f1_score(rtrue,rpred,average="macro",zero_division=0)
    cm=confusion_matrix(rtrue,rpred)

    print(f"Epoch {epoch+1}: Loss={total/len(train_loader):.4f} F1={f1:.4f} Acc={acc:.4f} LogRMSE={log_rmse:.4f} RMSE={rmse:.2f}")
    print(cm)

    if f1>best_f1:
        best_f1=f1
        best=(epoch+1,rmse,mae,pear,r2,acc,prec,rec,f1,cm)

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