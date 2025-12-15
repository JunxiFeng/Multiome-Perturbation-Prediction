#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline LatentAdditive (NO GET, NO ATAC, NO gating)
----------------------------------------------------

z_rna = gene_encoder([RNA, covariate])
z_pert = pert_encoder(perturbation_onehot)

z_fused = z_rna + z_pert

x_hat = decoder([z_fused, covariate_dec])
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# =========================================
# 0️⃣ Reproducibility
# =========================================
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================
# 1️⃣ Load AnnData (same file)
# =========================================
adata_path = "/gpfs/home/junxif/xin_lab/perturbench/data/boli_anndata/boli_with_GETembedding_celltypeaware_filled.h5ad"
split_path = "/gpfs/home/junxif/xin_lab/perturbench/data/boli_251006_1_qual_high_amt_high_split.csv"

adata = sc.read_h5ad(adata_path)
print("Loaded AnnData:", adata)

# =========================================
# 2️⃣ Load split CSV
# =========================================
df_split = pd.read_csv(split_path, header=None, names=["barcode", "split"])
df_split.index = df_split["barcode"]
adata.obs["split"] = adata.obs.index.map(df_split["split"])

print("Split counts:")
print(adata.obs["split"].value_counts())

# =========================================
# 3️⃣ Build train/val/test
# =========================================
train_adata = adata[adata.obs["split"] == "train"].copy()
val_adata   = adata[adata.obs["split"] == "val"].copy()
test_adata  = adata[adata.obs["split"] == "test"].copy()

n_genes = adata.n_vars
print(f"Train={train_adata.n_obs} | Val={val_adata.n_obs} | Test={test_adata.n_obs}")
print(f"Genes={n_genes}")

# =========================================
# 4️⃣ Convert X matrices to tensors
# =========================================
def to_numpy(X):
    return X.toarray() if not isinstance(X, np.ndarray) else X

X_train = torch.tensor(to_numpy(train_adata.X), dtype=torch.float32)
X_val   = torch.tensor(to_numpy(val_adata.X),   dtype=torch.float32)
X_test  = torch.tensor(to_numpy(test_adata.X),  dtype=torch.float32)

# =========================================
# 5️⃣ Perturbation one-hot
# =========================================
pert = adata.obs["condition"].astype("category")
pert_onehot = pd.get_dummies(pert)

p_train = torch.tensor(pert_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
p_val   = torch.tensor(pert_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
p_test  = torch.tensor(pert_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_perts = p_train.shape[1]
print("Pert dim:", n_perts)

# =========================================
# 6️⃣ Celltype covariates (one-hot)
# =========================================
celltypes = adata.obs["celltype_mapped"].astype("category")
cov_onehot = pd.get_dummies(celltypes)

cov_train = torch.tensor(cov_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
cov_val   = torch.tensor(cov_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
cov_test  = torch.tensor(cov_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_cov = cov_train.shape[1]
print("Cov dim:", n_cov)

cov_train_enc = cov_train_dec = cov_train
cov_val_enc   = cov_val_dec   = cov_val
cov_test_enc  = cov_test_dec  = cov_test

# =========================================
# 7️⃣ DataLoaders
# =========================================
train_ds = TensorDataset(X_train, p_train, cov_train_enc, cov_train_dec)
val_ds   = TensorDataset(X_val,   p_val,   cov_val_enc,   cov_val_dec)
test_ds  = TensorDataset(X_test,  p_test,  cov_test_enc,  cov_test_dec)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print("DataLoaders ready.")

# =========================================
# 8️⃣ Baseline LatentAdditive Model
# =========================================
from torch import nn

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
    BASELINE:
      z_rna  = gene_encoder([x, cov])
      z_pert = pert_encoder(p)

      z_total = z_rna + z_pert
      out = decoder([z_total, cov_dec])
    """

    def __init__(
        self,
        n_genes,
        n_perts,
        n_covariates_enc,
        n_covariates_dec,
        latent_dim=160,
        width=3072,
        n_layers=3,
        dropout=0.1,
        softplus_output=True,
    ):
        super().__init__()

        self.softplus_output = softplus_output

        self.gene_encoder = MLP(n_genes + n_covariates_enc,
                                width, latent_dim, n_layers, dropout)

        self.pert_encoder = MLP(n_perts,
                                width, latent_dim, n_layers, dropout)

        self.decoder = MLP(
            latent_dim + n_covariates_dec,
            width,
            n_genes,
            n_layers,
            dropout,
        )

    def forward(self, x, p, cov_enc, cov_dec):
        z_rna  = self.gene_encoder(torch.cat([x, cov_enc], dim=1))
        z_pert = self.pert_encoder(p)

        z = z_rna + z_pert
        z_full = torch.cat([z, cov_dec], dim=1)

        out = self.decoder(z_full)
        if self.softplus_output:
            out = F.softplus(out)

        return out


# =========================================
# 9️⃣ Initialize model + optimizer
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = LatentAdditiveBaseline(
    n_genes=n_genes,
    n_perts=n_perts,
    n_covariates_enc=n_cov,
    n_covariates_dec=n_cov,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
scaler = torch.cuda.amp.GradScaler()

# =========================================
# 🔟 Training Loop
# =========================================
n_epochs = 20
for epoch in range(n_epochs):

    # ---- TRAIN ----
    model.train()
    train_loss = 0.

    for xb, pb, cenc, cdec in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1}/{n_epochs} (train)"):

        xb, pb, cenc, cdec = (
            xb.to(device), pb.to(device),
            cenc.to(device), cdec.to(device),
        )

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred = model(xb, pb, cenc, cdec)
            loss = F.mse_loss(pred, xb)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for xb, pb, cenc, cdec in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{n_epochs} (val)"
        ):
            xb, pb, cenc, cdec = (
                xb.to(device), pb.to(device),
                cenc.to(device), cdec.to(device),
            )

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = model(xb, pb, cenc, cdec)
                loss = F.mse_loss(pred, xb)

            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    sched.step(val_loss)

    print(f"Epoch {epoch+1:02d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

print("Training complete.")

# =========================================
# 1️⃣2️⃣ Evaluate
# =========================================
import sys
sys.path.append("/gpfs/home/junxif/xin_lab/multiome")

from utils.eval import evaluate_model
results = evaluate_model(model, test_loader, test_adata, device=device, k=50)

print(results)
