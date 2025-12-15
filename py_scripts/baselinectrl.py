#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CTRL-only LatentAdditive Baseline
---------------------------------
Train and evaluate model on control cells ONLY.

Metrics:
    - RMSE
    - Mean Pearson (per-cell)
    - Mean Cosine (per-cell)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from torch import nn

# ====================================================
# 0) Reproducibility
# ====================================================
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====================================================
# 1) Load AnnData
# ====================================================
adata_path = "/gpfs/home/junxif/xin_lab/perturbench/data/boli_anndata/boli_with_GETembedding_celltypeaware_filled.h5ad"
split_path = "/gpfs/home/junxif/xin_lab/perturbench/data/boli_251006_1_qual_high_amt_high_split.csv"

adata = sc.read_h5ad(adata_path)
print("Loaded AnnData:", adata)

# ====================================================
# 2) Load train/val/test split
# ====================================================
df_split = pd.read_csv(split_path, header=None, names=["barcode", "split"])
df_split.index = df_split["barcode"]
adata.obs["split"] = adata.obs.index.map(df_split["split"])

# ====================================================
# 3) Use *only* control cells
# ====================================================
ctrl_filter = adata.obs["condition"] == "ctrl"

ctrl_train = adata[(adata.obs["split"] == "train") & ctrl_filter]
ctrl_val   = adata[(adata.obs["split"] == "val")   & ctrl_filter]
ctrl_test  = adata[(adata.obs["split"] == "test")  & ctrl_filter]

print("CTRL counts:")
print("Train:", ctrl_train.n_obs)
print("Val:",   ctrl_val.n_obs)
print("Test:",  ctrl_test.n_obs)

n_genes = adata.n_vars

# ====================================================
# 4) Convert X matrices
# ====================================================
def to_numpy(X):
    return X.toarray() if not isinstance(X, np.ndarray) else X

X_train = torch.tensor(to_numpy(ctrl_train.X), dtype=torch.float32)
X_val   = torch.tensor(to_numpy(ctrl_val.X),   dtype=torch.float32)
X_test  = torch.tensor(to_numpy(ctrl_test.X),  dtype=torch.float32)

# ====================================================
# 5) Covariates (still needed)
# ====================================================
cov = pd.get_dummies(adata.obs["celltype_mapped"].astype("category"))

cov_train = torch.tensor(cov.loc[ctrl_train.obs.index].values, dtype=torch.float32)
cov_val   = torch.tensor(cov.loc[ctrl_val.obs.index].values,   dtype=torch.float32)
cov_test  = torch.tensor(cov.loc[ctrl_test.obs.index].values,  dtype=torch.float32)

n_cov = cov_train.shape[1]

# ====================================================
# 6) Dummy perturbation vector (all zeros)
# ====================================================
p_train = torch.zeros((ctrl_train.n_obs, 1))
p_val   = torch.zeros((ctrl_val.n_obs, 1))
p_test  = torch.zeros((ctrl_test.n_obs, 1))
n_perts = 1

# ====================================================
# 7) Build DataLoaders
# ====================================================
train_ds = TensorDataset(X_train, p_train, cov_train, cov_train)
val_ds   = TensorDataset(X_val,   p_val,   cov_val,   cov_val)
test_ds  = TensorDataset(X_test,  p_test,  cov_test,  cov_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
test_loader  = DataLoader(test_ds,  batch_size=64)

# ====================================================
# 8) LatentAdditive baseline
# ====================================================
class MLP(nn.Module):
    def __init__(self, in_dim, width, out_dim, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LatentAdditiveBaseline(nn.Module):
    """
    z_rna  = gene_encoder([x, cov])
    z_pert = pert_encoder(p)   <- p is always zero here

    x_hat = decoder([z_rna + z_pert, cov_dec])
    """
    def __init__(self, n_genes, n_perts, n_cov, latent_dim=160, width=3072, dropout=0.1):
        super().__init__()
        self.softplus_output = True

        self.gene_encoder = MLP(n_genes + n_cov, width, latent_dim, dropout=dropout)
        self.pert_encoder = MLP(n_perts, width, latent_dim, dropout=dropout)
        self.decoder      = MLP(latent_dim + n_cov, width, n_genes, dropout=dropout)

    def forward(self, x, p, cov_enc, cov_dec):
        z_rna  = self.gene_encoder(torch.cat([x, cov_enc], dim=1))
        z_pert = self.pert_encoder(p)
        z      = z_rna + z_pert
        out    = self.decoder(torch.cat([z, cov_dec], dim=1))
        return F.softplus(out)


# ====================================================
# 9) Train CTRL-only model
# ====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LatentAdditiveBaseline(n_genes, n_perts, n_cov).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(20):
    model.train()
    train_loss = 0

    for xb, pb, cenc, cdec in tqdm(train_loader, desc=f"E{epoch+1}/20 Train"):
        xb, pb, cenc, cdec = xb.to(device), pb.to(device), cenc.to(device), cdec.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred = model(xb, pb, cenc, cdec)
            loss = F.mse_loss(pred, xb)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

print("Training complete.\n")

# ====================================================
# 10) CTRL-only evaluation
# ====================================================
def eval_ctrl_reconstruction(model, test_loader, device="cuda"):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for xb, pb, cenc, cdec in test_loader:
            xb, pb, cenc, cdec = xb.to(device), pb.to(device), cenc.to(device), cdec.to(device)
            pred = model(xb, pb, cenc, cdec)
            all_true.append(xb.cpu())
            all_pred.append(pred.cpu())

    true = torch.cat(all_true).numpy()
    pred = torch.cat(all_pred).numpy()

    rmse = np.sqrt(np.mean((true - pred)**2))

    # Per-cell metrics
    def cosine(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    pearsons = []
    cosines  = []

    from scipy.stats import pearsonr
    for i in range(true.shape[0]):
        try:
            pearsons.append(pearsonr(true[i], pred[i])[0])
        except:
            pearsons.append(np.nan)
        cosines.append(cosine(true[i], pred[i]))

    return {
        "rmse": float(rmse),
        "pearson_mean": float(np.nanmean(pearsons)),
        "cosine_mean": float(np.mean(cosines)),
    }

results = eval_ctrl_reconstruction(model, test_loader, device=device)
print("CTRL-only reconstruction results:")
print(results)
