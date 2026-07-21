
# Combined training script skeleton
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from dataset import ArsenicDataset
from dataloader import get_dataloader,get_validation_dataloader
from cnn3d import CNN
from pointnet import PointNetHead

EPOCHS=50
LR=1e-4
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=32
CLASS_WEIGHT=0.5

print("Pick Model:\n 1. 3D CNN\n 2. PointNet")
choice=input("> ").strip()

dataset=ArsenicDataset()
train_loader=get_dataloader(dataset,BATCH_SIZE,12)
val_loader=get_validation_dataloader(dataset,BATCH_SIZE,12)

sample=next(iter(train_loader))
input_channels=sample["voxel"].shape[1]

backbone=CNN(input_channels) if choice=="1" else PointNetHead(input_features=6,embedding_size=256)

class Model(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.backbone=b

        if choice=="1": # CNN
            feature_size = 256
        else: # PointNet
            feature_size = 768

        self.regression = nn.Linear(feature_size,1)
        self.classification = nn.Linear(feature_size,3)

    def forward(self,batch):
        if choice=="1":
            feat=self.backbone(batch["voxel"].to(DEVICE))
        else:
            feat=self.backbone([p.to(DEVICE) for p in batch["points"]])
        return {"arsenic":self.regression(feat).squeeze(1),"risk":self.classification(feat)}

model=Model(backbone).to(DEVICE)
reg_loss=nn.MSELoss();cls_loss=nn.CrossEntropyLoss()
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
best_f1=-1;best=None

for epoch in range(EPOCHS):
    model.train();train_loss=0

    for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{EPOCHS}"):
        y=batch["label"].to(DEVICE);r=batch["risk"].to(DEVICE)
        opt.zero_grad()
        out=model(batch)
        loss=CLASS_WEIGHT*reg_loss(out["arsenic"],y)+cls_loss(out["risk"],r)
        loss.backward();opt.step();train_loss+=loss.item()
    model.eval();pred=[];true=[];rpred=[];rtrue=[];vl=0

    with torch.no_grad():
        for batch in val_loader:
            y=batch["label"].to(DEVICE);r=batch["risk"].to(DEVICE)
            out=model(batch)
            vl+=(CLASS_WEIGHT*reg_loss(out["arsenic"],y)+cls_loss(out["risk"],r)).item()
            pred.extend(out["arsenic"].cpu().numpy());true.extend(y.cpu().numpy())
            rpred.extend(torch.argmax(out["risk"],1).cpu().numpy());rtrue.extend(r.cpu().numpy())

    pred=np.array(pred);true=np.array(true)
    rmse=np.sqrt(mean_squared_error(true,pred));mae=mean_absolute_error(true,pred);r2=r2_score(true,pred);pear=np.corrcoef(true,pred)[0,1]
    acc=accuracy_score(rtrue,rpred);prec=precision_score(rtrue,rpred,average="macro",zero_division=0);rec=recall_score(rtrue,rpred,average="macro",zero_division=0);f1=f1_score(rtrue,rpred,average="macro",zero_division=0)
    print(f"Epoch {epoch+1}: F1={f1:.4f} Acc={acc:.4f} RMSE={rmse:.4f}")

    if f1>best_f1:
        best_f1=f1
        best=(epoch+1,rmse,mae,pear,r2,acc,prec,rec,f1,confusion_matrix(rtrue,rpred))

e,rmse,mae,pear,r2,acc,prec,rec,f1,cm=best
print(f"\nBest Epoch {e}\nRMSE {rmse:.4f}\nMAE {mae:.4f}\nPearson {pear:.4f}\nR2 {r2:.4f}\nAccuracy {acc:.4f}\nPrecision {prec:.4f}\nRecall {rec:.4f}\nF1 {f1:.4f}\n{cm}")