import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from dataset import ArsenicDataset
from dataloader import get_dataloaders,NUM_WORKERS,BATCH_SIZE
from cnn3d import CNN
from pointnet import PointNetHead


EPOCHS = 100
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE = 15


choice=input("Pick Model:\n1. 3D CNN\n2. PointNet\n> ").strip()
dataset=ArsenicDataset()
train_loader,val_loader=get_dataloaders(dataset,BATCH_SIZE,NUM_WORKERS)
backbone = CNN(dataset.raster_channels,34) if choice=="1" else PointNetHead(15,256)


class Model(nn.Module):
    def __init__(self,b):
        super().__init__()

        self.backbone=b

        f = 256 if choice=="1" else 1024

        self.reg = nn.Linear(f,2)

        # binary output
        self.binary = nn.Linear(f,1)


    def forward(self,b):

        if choice=="1":
            x=self.backbone(
                b["voxel"].to(
                    DEVICE,
                    non_blocking=True
                )
            )

        else:
            x=self.backbone(
                [
                    p.to(
                        DEVICE,
                        non_blocking=True
                    )
                    for p in b["points"]
                ]
            )


        r=self.reg(x)

        return {
            "mean":r[:,0],
            "var":r[:,1],
            "binary":self.binary(x).squeeze(1)
        }



def gaussian_nll(m,v,y):

    v=torch.clamp(v,-3,3)

    return (
        ((y-m)**2)/torch.exp(v)+v
    ).mean()



model=Model(backbone).to(DEVICE)


opt=optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


scaler=torch.amp.GradScaler("cuda")


best_f1=-1
best_epoch=0
patience_counter=0


for epoch in range(EPOCHS):

    model.train()

    total=0


    for b in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):

        y=b["label"].to(DEVICE,non_blocking=True)

        # <=10 = 0
        # >10 = 1
        target=(y>np.log1p(10)).float()


        opt.zero_grad(set_to_none=True)


        with torch.amp.autocast("cuda"):

            o=model(b)


            reg=gaussian_nll(
                o["mean"],
                o["var"],
                y
            )


            cls=nn.functional.binary_cross_entropy_with_logits(
                o["binary"],
                target
            )


            loss=0.3*reg+cls



        scaler.scale(loss).backward()

        scaler.unscale_(opt)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5
        )

        scaler.step(opt)
        scaler.update()


        total+=loss.item()



    # validation

    model.eval()

    pred_class=[]
    true_class=[]
    pred_reg=[]
    true_reg=[]


    with torch.no_grad():

        for b in val_loader:

            y=b["label"].to(DEVICE)

            with torch.amp.autocast("cuda"):

                o=model(b)


            probs=torch.sigmoid(
                o["binary"]
            )


            pred_class.extend(
                (probs>0.5)
                .cpu()
                .numpy()
            )


            true_class.extend(
                (y>np.log1p(10))
                .cpu()
                .numpy()
            )


            pred_reg.extend(
                o["mean"]
                .cpu()
                .numpy()
            )


            true_reg.extend(
                y.cpu()
                .numpy()
            )


    f1=f1_score(
        true_class,
        pred_class
    )


    acc=accuracy_score(
        true_class,
        pred_class
    )


    rmse=np.sqrt(
        mean_squared_error(
            np.expm1(true_reg),
            np.expm1(pred_reg)
        )
    )


    cm=confusion_matrix(
        true_class,
        pred_class
    )


    print(
        f"Epoch {epoch+1}: "
        f"Loss={total/len(train_loader):.4f} "
        f"F1={f1:.4f} "
        f"Acc={acc:.4f} "
        f"RMSE={rmse:.2f}"
    )

    print(cm)



    # save best model

    if f1 > best_f1:

        best_f1=f1
        best_epoch=epoch+1
        patience_counter=0

        torch.save(
            model.state_dict(),
            "best_binary_model.pt"
        )

        print("Saved best model")


    else:

        patience_counter+=1


    if patience_counter >= PATIENCE:

        print(
            f"Early stopping at epoch {epoch+1}"
        )

        break



print(
f"""
Best Epoch {best_epoch}
Best F1 {best_f1:.4f}

Model saved:
best_binary_model.pt
"""
)