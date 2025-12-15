#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LatentAdditiveGET (Strategy A):
  X, p, cov + GET fused via:
      cell_GET = X @ GET_ct

Memory-safe: does NOT expand (N × genes × d_get).
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
# 1️⃣ Load AnnData WITH GET embeddings
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
print("Split counts:\n", adata.obs["split"].value_counts())

# =========================================
# 3️⃣ Build train / val / test sets
# =========================================
train_adata = adata[adata.obs["split"] == "train"].copy()
val_adata   = adata[adata.obs["split"] == "val"].copy()
test_adata  = adata[adata.obs["split"] == "test"].copy()

print(f"Train: {train_adata.n_obs} | Val: {val_adata.n_obs} | Test: {test_adata.n_obs}")
n_genes = adata.n_vars

# =========================================
# 4️⃣ Convert expression matrix
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
# 6️⃣ Celltype covariates one-hot
# =========================================
celltypes_all = adata.obs["celltype_mapped"].astype("category")
cov_onehot = pd.get_dummies(celltypes_all)

cov_train = torch.tensor(cov_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
cov_val   = torch.tensor(cov_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
cov_test  = torch.tensor(cov_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_cov = cov_train.shape[1]
print("Cov dim:", n_cov)

# encoder/decoder identical covs
cov_train_enc = cov_train_dec = cov_train
cov_val_enc   = cov_val_dec   = cov_val
cov_test_enc  = cov_test_dec  = cov_test

# =========================================
# 7️⃣ GET embeddings (celltype-level only)
# =========================================
get_dict_raw = adata.uns["GET_embeddings"]

# Map celltype → index
unique_cts = sorted(list(get_dict_raw.keys()))
ct2idx = {ct: i for i, ct in enumerate(unique_cts)}

# Build CT-index vectors
ct_train_idx = torch.tensor(
    [ct2idx[ct] for ct in train_adata.obs["celltype_mapped"].astype(str).values],
    dtype=torch.long,
)
ct_val_idx = torch.tensor(
    [ct2idx[ct] for ct in val_adata.obs["celltype_mapped"].astype(str).values],
    dtype=torch.long,
)
ct_test_idx = torch.tensor(
    [ct2idx[ct] for ct in test_adata.obs["celltype_mapped"].astype(str).values],
    dtype=torch.long,
)

# Convert GET matrices to tensors but DO NOT expand
get_list = []
for ct in unique_cts:
    M = get_dict_raw[ct]  # (n_genes, d_get)
    get_list.append(torch.tensor(M, dtype=torch.float32))
GET_tensor = torch.stack(get_list, dim=0)  # (n_celltypes × n_genes × d_get)
GET_tensor = GET_tensor.cuda() if torch.cuda.is_available() else GET_tensor
n_celltypes, n_genes_check, d_get = GET_tensor.shape

print(f"Loaded GET as CT-level tensor: {GET_tensor.shape}")

assert n_genes_check == n_genes, "Gene dimension mismatch!"

# =========================================
# 8️⃣ DataLoaders (no large GET inside dataset)
# =========================================
train_ds = TensorDataset(X_train, p_train, cov_train_enc, cov_train_dec, ct_train_idx)
val_ds   = TensorDataset(X_val,   p_val,   cov_val_enc,   cov_val_dec,   ct_val_idx)
test_ds  = TensorDataset(X_test,  p_test,  cov_test_enc,  cov_test_dec,  ct_test_idx)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print("DataLoaders ready.")

# =========================================
# 9️⃣ Model: LatentAdditiveGET (Strategy A)
# =========================================
from torch import nn
import torch.nn.functional as F

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


class LatentAdditiveGET(nn.Module):
    """
    Strategy A:
        cell_GET = X @ GET_ct
        z = encoder(x,p,cov) || cell_GET
    """

    def __init__(
        self,
        n_genes,
        n_perts,
        n_covariates_enc,
        n_covariates_dec,
        d_get,
        latent_dim=160,
        encoder_width=3072,
        n_layers=3,
        dropout=0.1,
        softplus_output=True,
        get_tensor=None,
    ):
        super().__init__()
        self.softplus_output = softplus_output

        # base encoders
        self.gene_encoder = MLP(
            n_genes + n_covariates_enc, encoder_width, latent_dim,
            n_layers, dropout
        )
        self.pert_encoder = MLP(
            n_perts, encoder_width, latent_dim,
            n_layers, dropout
        )

        # expanded latent dimension = latent_dim + d_get
        self.decoder = MLP(
            latent_dim + d_get + n_covariates_dec,
            encoder_width,
            n_genes,
            n_layers,
            dropout,
        )

        # store GET
        self.GET = get_tensor  # (n_celltypes × n_genes × d_get)

    def forward(self, x, p, cov_enc, cov_dec, ct_idx):
        """
        ct_idx: (batch,) long
        """
        # latent ctrl + pert
        latent_ctrl = self.gene_encoder(torch.cat([x, cov_enc], dim=1))
        latent_pert = self.pert_encoder(p)
        latent_sum = latent_ctrl + latent_pert  # (batch × latent_dim)

        # GET fusion
        get_ct_mat = self.GET[ct_idx]          # (batch × n_genes × d_get)
        cell_GET = torch.bmm(
            x.unsqueeze(1),                    # (batch × 1 × n_genes)
            get_ct_mat                         # (batch × n_genes × d_get)
        ).squeeze(1)                           # (batch × d_get)

        # combine
        z = torch.cat([latent_sum, cell_GET, cov_dec], dim=1)
        out = self.decoder(z)

        if self.softplus_output:
            out = F.softplus(out)

        return out


# =========================================
# 10️⃣ Initialize model & optimizer
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = LatentAdditiveGET(
    n_genes=n_genes,
    n_perts=n_perts,
    n_covariates_enc=n_cov,
    n_covariates_dec=n_cov,
    d_get=d_get,
    get_tensor=GET_tensor,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
scaler = torch.cuda.amp.GradScaler()

# =========================================
# 11️⃣ Training loop
# =========================================
n_epochs = 20
for epoch in range(n_epochs):

    # ---- TRAIN ----
    model.train()
    train_loss = 0.

    for xb, pb, cenc, cdec, ct_idx in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{n_epochs} (train)"
    ):
        xb, pb, cenc, cdec, ct_idx = (
            xb.to(device), pb.to(device),
            cenc.to(device), cdec.to(device),
            ct_idx.to(device)
        )

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            recon = model(xb, p=pb, cov_enc=cenc, cov_dec=cdec, ct_idx=ct_idx)
            loss = F.mse_loss(recon, xb)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- VAL ----
    model.eval()
    val_loss = 0.

    with torch.no_grad():
        for xb, pb, cenc, cdec, ct_idx in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{n_epochs} (val)"
        ):
            xb, pb, cenc, cdec, ct_idx = (
                xb.to(device), pb.to(device),
                cenc.to(device), cdec.to(device),
                ct_idx.to(device)
            )

            with torch.cuda.amp.autocast(dtype=torch.float16):
                recon = model(xb, p=pb, cov_enc=cenc, cov_dec=cdec, ct_idx=ct_idx)
                loss = F.mse_loss(recon, xb)

            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    sched.step(val_loss)

    print(f"Epoch {epoch+1:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

print("Training complete.")

# =========================================
# 12️⃣ Evaluation (your eval function)
# =========================================
import sys, os
sys.path.append(os.path.abspath("/gpfs/home/junxif/xin_lab/multiome"))
from utils.eval import evaluate_model
import importlib
import utils.eval
importlib.reload(utils.eval)

results = evaluate_model(model, test_loader, test_adata, device=device, k=50)
print(results)
