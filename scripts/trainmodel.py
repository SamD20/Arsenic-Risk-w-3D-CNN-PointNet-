import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from dataset import ArsenicDataset
from dataloader import get_dataloader,get_validation_dataloader
from cnn3d import CNN
from pointnet import PointNetHead

EPOCHS=50
LR=1e-4
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=32
CLASS_WEIGHT=0.5

dataset=ArsenicDataset()
train_loader=get_dataloader(dataset,BATCH_SIZE,12)
val_loader=get_validation_dataloader(dataset,BATCH_SIZE,12)

sample=next(iter(train_loader))
input_channels=sample["voxel"].shape[1]

class MultiTaskModel(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.cnn=CNN(input_channels)
        self.pointnet=PointNetHead(input_features=15,embedding_size=256)
        self.fusion=nn.Sequential(nn.Linear(1280,256),nn.ReLU(),nn.Dropout(0.3))
        self.regression_head=nn.Linear(256,1)
        self.classification_head=nn.Linear(256,3)

    def forward(self,batch):
        cnn_features=self.cnn(batch["voxel"].to(DEVICE))
        point_features=self.pointnet([p.to(DEVICE) for p in batch["points"]])
        features=self.fusion(torch.cat([cnn_features,point_features],1))
        return {"arsenic":self.regression_head(features).squeeze(1),"risk":self.classification_head(features)}

model=MultiTaskModel(input_channels).to(DEVICE)

reg_loss=nn.MSELoss()
cls_loss=nn.CrossEntropyLoss()
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)

best_f1=-1
best=None

for epoch in range(EPOCHS):
    model.train()
    for batch in tqdm(train_loader,desc=f"Epoch {epoch+1}/{EPOCHS}"):
        y=batch["label"].to(DEVICE)
        r=batch["risk"].to(DEVICE)

        opt.zero_grad()
        out=model(batch)
        loss=CLASS_WEIGHT*reg_loss(out["arsenic"],y)+cls_loss(out["risk"],r)
        loss.backward()
        opt.step()

    model.eval()
    pred=[];true=[];rpred=[];rtrue=[]

    with torch.no_grad():
        for batch in val_loader:
            y=batch["label"].to(DEVICE)
            r=batch["risk"].to(DEVICE)
            out=model(batch)

            pred.extend(out["arsenic"].cpu().numpy())
            true.extend(y.cpu().numpy())
            rpred.extend(torch.argmax(out["risk"],1).cpu().numpy())
            rtrue.extend(r.cpu().numpy())

    pred=np.array(pred);true=np.array(true)

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

print(f"\nBest Epoch {e}\nRMSE {rmse:.4f}\nMAE {mae:.4f}\nPearson {pear:.4f}\nR2 {r2:.4f}\nAccuracy {acc:.4f}\nPrecision {prec:.4f}\nRecall {rec:.4f}\nF1 {f1:.4f}\n{cm}")