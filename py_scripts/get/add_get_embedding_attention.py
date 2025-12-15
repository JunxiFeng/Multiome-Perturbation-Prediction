#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LatentAdditiveGET_CrossAttention:
    RNA latent (query) attends to GET embeddings (key/value).

    Q = Wq(z_rna)                (batch × d_att)
    K = Wk(GET_ct)               (batch × genes × d_att)
    V = Wv(GET_ct)               (batch × genes × d_att)

    Attention:
        A = softmax(QKᵀ / sqrt(d_att))
        z_cross = A @ V

    Final latent = concat(z_rna, z_cross, cov_dec)
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
# 1️⃣ Load AnnData
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
# 3️⃣ Build train/val/test sets
# =========================================
train_adata = adata[adata.obs["split"] == "train"].copy()
val_adata   = adata[adata.obs["split"] == "val"].copy()
test_adata  = adata[adata.obs["split"] == "test"].copy()

print(f"Train={train_adata.n_obs} | Val={val_adata.n_obs} | Test={test_adata.n_obs}")
n_genes = adata.n_vars

# =========================================
# 4️⃣ Convert X to tensors
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
# 6️⃣ Celltype covariates
# =========================================
celltypes_all = adata.obs["celltype_mapped"].astype("category")
cov_onehot = pd.get_dummies(celltypes_all)

cov_train = torch.tensor(cov_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
cov_val   = torch.tensor(cov_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
cov_test  = torch.tensor(cov_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_cov = cov_train.shape[1]
print("Cov dims:", n_cov)

cov_train_enc = cov_train_dec = cov_train
cov_val_enc   = cov_val_dec   = cov_val
cov_test_enc  = cov_test_dec  = cov_test

# =========================================
# 7️⃣ GET embeddings (celltype-level GET_ct: genes × d_get)
# =========================================
get_dict = adata.uns["GET_embeddings"]
unique_cts = sorted(list(get_dict.keys()))
ct2idx = {ct: i for i, ct in enumerate(unique_cts)}

ct_train_idx = torch.tensor([ct2idx[c] for c in train_adata.obs["celltype_mapped"]], dtype=torch.long)
ct_val_idx   = torch.tensor([ct2idx[c] for c in val_adata.obs["celltype_mapped"]],   dtype=torch.long)
ct_test_idx  = torch.tensor([ct2idx[c] for c in test_adata.obs["celltype_mapped"]], dtype=torch.long)

# Stack GET matrices
get_list = []
for ct in unique_cts:
    M = get_dict[ct]  # (genes × d_get)
    get_list.append(torch.tensor(M, dtype=torch.float32))

GET_tensor = torch.stack(get_list, dim=0)   # (n_celltypes × n_genes × d_get)
GET_tensor = GET_tensor.cuda() if torch.cuda.is_available() else GET_tensor

n_celltypes, n_genes_check, d_get = GET_tensor.shape
assert n_genes_check == n_genes
print("GET tensor:", GET_tensor.shape)

# =========================================
# 8️⃣ DataLoaders
# =========================================
train_ds = TensorDataset(X_train, p_train, cov_train_enc, cov_train_dec, ct_train_idx)
val_ds   = TensorDataset(X_val,   p_val,   cov_val_enc,   cov_val_dec,   ct_val_idx)
test_ds  = TensorDataset(X_test,  p_test,  cov_test_enc,  cov_test_dec,  ct_test_idx)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print("DataLoaders ready.")

# =========================================
# 9️⃣ Model: Cross-Attention GET Fusion
# =========================================
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, width, out_dim, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else width, width),
                nn.LayerNorm(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LatentAdditiveGET_CrossAttention(nn.Module):
    def __init__(
        self,
        n_genes,
        n_perts,
        n_covariates_enc,
        n_covariates_dec,
        d_get,
        get_tensor,
        latent_dim=160,
        att_dim=128,
        encoder_width=2048,
        n_layers=3,
        dropout=0.1,
        softplus_output=True,
    ):
        super().__init__()

        self.GET = get_tensor
        self.softplus_output = softplus_output

        # RNA encoders
        self.gene_encoder = MLP(
            n_genes + n_covariates_enc,
            encoder_width,
            latent_dim,
            n_layers,
            dropout,
        )
        self.pert_encoder = MLP(
            n_perts,
            encoder_width,
            latent_dim,
            n_layers,
            dropout,
        )

        # Cross-attention projections
        self.Wq = nn.Linear(latent_dim, att_dim)
        self.Wk = nn.Linear(d_get, att_dim)
        self.Wv = nn.Linear(d_get, att_dim)

        # Decoder
        self.decoder = MLP(
            latent_dim + att_dim + n_covariates_dec,
            encoder_width,
            n_genes,
            n_layers,
            dropout,
        )

    def forward(self, x, p, cov_enc, cov_dec, ct_idx):

        # Base latent
        latent_ctrl = self.gene_encoder(torch.cat([x, cov_enc], dim=1))
        latent_pert = self.pert_encoder(p)
        z_rna = latent_ctrl + latent_pert  # (batch × latent_dim)

        # Cross-attention
        Q = self.Wq(z_rna).unsqueeze(1)  # (batch × 1 × att_dim)

        GET_ct = self.GET[ct_idx]        # (batch × genes × d_get)
        K = self.Wk(GET_ct)              # (batch × genes × att_dim)
        V = self.Wv(GET_ct)              # (batch × genes × att_dim)

        attn_scores = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(K.shape[-1]),
            dim=-1,
        )  # (batch × 1 × genes)

        z_cross = torch.bmm(attn_scores, V).squeeze(1)  # (batch × att_dim)

        # concat + decode
        z_total = torch.cat([z_rna, z_cross, cov_dec], dim=1)
        out = self.decoder(z_total)

        if self.softplus_output:
            out = F.softplus(out)

        return out

# =========================================
# 🔟 Initialize model & optimizer
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = LatentAdditiveGET_CrossAttention(
    n_genes=n_genes,
    n_perts=n_perts,
    n_covariates_enc=n_cov,
    n_covariates_dec=n_cov,
    d_get=d_get,
    get_tensor=GET_tensor,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
scaler = torch.amp.GradScaler("cuda")

# =========================================
# 1️⃣1️⃣ Training loop
# =========================================
n_epochs = 20
for epoch in range(n_epochs):

    model.train()
    train_loss = 0.

    for xb, pb, cenc, cdec, ct_idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} (train)"):

        xb, pb, cenc, cdec, ct_idx = (
            xb.to(device), pb.to(device),
            cenc.to(device), cdec.to(device),
            ct_idx.to(device),
        )

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon = model(xb, pb, cenc, cdec, ct_idx)
            loss = F.mse_loss(recon, xb)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    val_loss = 0.

    with torch.no_grad():
        for xb, pb, cenc, cdec, ct_idx in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} (val)"):

            xb, pb, cenc, cdec, ct_idx = (
                xb.to(device), pb.to(device),
                cenc.to(device), cdec.to(device),
                ct_idx.to(device),
            )

            with torch.amp.autocast("cuda", dtype=torch.float16):
                recon = model(xb, pb, cenc, cdec, ct_idx)
                loss = F.mse_loss(recon, xb)

            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    sched.step(val_loss)

    print(f"Epoch {epoch+1:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

print("Training finished.")

# =========================================
# 1️⃣2️⃣ Evaluation
# =========================================
import sys
sys.path.append("/gpfs/home/junxif/xin_lab/multiome")
from utils.eval import evaluate_model

results = evaluate_model(model, test_loader, test_adata, device=device, k=50)
print("==== FINAL RESULTS ====")
print(results)
