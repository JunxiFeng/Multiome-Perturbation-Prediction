#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LatentAdditiveATAC_Gated_Global:
    A single global scalar α (shared across all cells)
    Controls contribution of ATAC latent:

        z_atac = α * ATAC_encoder(ATAC_ct)

    Fusion:
        z_total = concat(z_rna, z_atac, cov)
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
# 2️⃣ Load split
# =========================================
df_split = pd.read_csv(split_path, header=None, names=["barcode", "split"])
df_split.index = df_split["barcode"]
adata.obs["split"] = adata.obs.index.map(df_split["split"])

train_adata = adata[adata.obs["split"] == "train"].copy()
val_adata   = adata[adata.obs["split"] == "val"].copy()
test_adata  = adata[adata.obs["split"] == "test"].copy()

n_genes = adata.n_vars

# =========================================
# 3️⃣ Convert X to tensors
# =========================================
def to_numpy(X):
    return X.toarray() if not isinstance(X, np.ndarray) else X

X_train = torch.tensor(to_numpy(train_adata.X), dtype=torch.float32)
X_val   = torch.tensor(to_numpy(val_adata.X), dtype=torch.float32)
X_test  = torch.tensor(to_numpy(test_adata.X), dtype=torch.float32)

# =========================================
# 4️⃣ Perturbation one-hot
# =========================================
pert = adata.obs["condition"].astype("category")
pert_onehot = pd.get_dummies(pert)

p_train = torch.tensor(pert_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
p_val   = torch.tensor(pert_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
p_test  = torch.tensor(pert_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_perts = p_train.shape[1]

# =========================================
# 5️⃣ Covariates one-hot
# =========================================
cov_onehot = pd.get_dummies(adata.obs["celltype_mapped"].astype("category"))

cov_train = torch.tensor(cov_onehot.loc[train_adata.obs.index].values, dtype=torch.float32)
cov_val   = torch.tensor(cov_onehot.loc[val_adata.obs.index].values,   dtype=torch.float32)
cov_test  = torch.tensor(cov_onehot.loc[test_adata.obs.index].values,  dtype=torch.float32)

n_cov = cov_train.shape[1]

# =========================================
# 6️⃣ ATAC embeddings (vector per celltype)
# =========================================
atac_dict = adata.uns["ATAC_embeddings"]
unique_cts = sorted(list(atac_dict.keys()))
ct2idx = {ct: i for i, ct in enumerate(unique_cts)}

# Mapping Allen CTs → ATAC CTs
remap = {
    "astro-epen": "astro_epen",
    "cnu-mge_gaba": "ctx-mge_gaba",
    "cr_glut": "it_glut",
    "ctx-cge_gaba": "ctx-mge_gaba",
    "ctx-mge_gaba": "ctx-mge_gaba",
    "glioblast": "glioblast",
    "immune": "immune",
    "imn": "imn",
    "ip": "ip",
    "it_glut": "it_glut",
    "nec": "ip",
    "nonit_glut": "nonit_glut",
    "opc-oligo": "opc_oligo",
    "rg": "opc_oligo",
    "vascular": "vascular",
}

def map_ct(ct):
    if ct in atac_dict:
        return ct
    if ct in remap:
        return remap[ct]
    raise KeyError(f"No ATAC mapping for {ct}")

ct_train_idx = torch.tensor([ct2idx[map_ct(c)] for c in train_adata.obs["celltype_mapped"]], dtype=torch.long)
ct_val_idx   = torch.tensor([ct2idx[map_ct(c)] for c in val_adata.obs["celltype_mapped"]], dtype=torch.long)
ct_test_idx  = torch.tensor([ct2idx[map_ct(c)] for c in test_adata.obs["celltype_mapped"]], dtype=torch.long)

# Build ATAC tensor
ATAC_tensor = torch.stack(
    [torch.tensor(atac_dict[ct], dtype=torch.float32) for ct in unique_cts],
    dim=0
)
ATAC_tensor = ATAC_tensor.cuda() if torch.cuda.is_available() else ATAC_tensor
n_celltypes, d_atac = ATAC_tensor.shape

print("ATAC tensor:", ATAC_tensor.shape)

# =========================================
# 7️⃣ DataLoaders
# =========================================
train_ds = TensorDataset(X_train, p_train, cov_train, cov_train, ct_train_idx)
val_ds   = TensorDataset(X_val,   p_val,   cov_val,   cov_val,   ct_val_idx)
test_ds  = TensorDataset(X_test,  p_test,  cov_test,  cov_test,  ct_test_idx)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
test_loader  = DataLoader(test_ds,  batch_size=64)

# =========================================
# 8️⃣ Global-Gated ATAC Model
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
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LatentAdditiveATAC_GlobalGated(nn.Module):

    def __init__(self, n_genes, n_perts, n_cov, d_atac, atac_tensor,
                 latent_dim=160, atac_latent_dim=160):
        super().__init__()

        self.ATAC = atac_tensor                  # (n_ct × d_atac)

        # *** Single global α ***
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 init

        # RNA encoders
        self.gene_encoder = MLP(n_genes + n_cov, 3072, latent_dim)
        self.pert_encoder = MLP(n_perts, 3072, latent_dim)

        # ATAC encoder
        self.atac_encoder = MLP(d_atac, 1024, atac_latent_dim)

        # decoder
        self.decoder = MLP(latent_dim + atac_latent_dim + n_cov, 3072, n_genes)

    def forward(self, x, p, cov_enc, cov_dec, ct_idx):

        # RNA latent
        latent_ctrl = self.gene_encoder(torch.cat([x, cov_enc], dim=1))
        latent_pert = self.pert_encoder(p)
        z_rna = latent_ctrl + latent_pert

        # ATAC latent
        ATAC_ct = self.ATAC[ct_idx]              # (batch × d_atac)
        z_atac_raw = self.atac_encoder(ATAC_ct)

        # global gate α
        alpha = torch.sigmoid(self.alpha)        # scalar
        z_atac = alpha * z_atac_raw              # (batch × atac_latent_dim)

        # fuse
        z_total = torch.cat([z_rna, z_atac, cov_dec], dim=1)
        out = self.decoder(z_total)

        return F.softplus(out)


# =========================================
# 9️⃣ Training
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LatentAdditiveATAC_GlobalGated(
    n_genes=n_genes,
    n_perts=n_perts,
    n_cov=n_cov,
    d_atac=d_atac,
    atac_tensor=ATAC_tensor,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)
scaler = torch.cuda.amp.GradScaler()

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    train_loss = 0

    for xb, pb, cenc, cdec, ct_idx in tqdm(train_loader):
        xb, pb, cenc, cdec, ct_idx = xb.to(device), pb.to(device), cenc.to(device), cdec.to(device), ct_idx.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon = model(xb, pb, cenc, cdec, ct_idx)
            loss = F.mse_loss(recon, xb)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, pb, cenc, cdec, ct_idx in val_loader:
            xb, pb, cenc, cdec, ct_idx = xb.to(device), pb.to(device), cenc.to(device), cdec.to(device), ct_idx.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                recon = model(xb, pb, cenc, cdec, ct_idx)
                loss = F.mse_loss(recon, xb)

            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    sched.step(val_loss)

    print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

# =========================================
# 🔟 Evaluation
# =========================================
import sys
sys.path.append("/gpfs/home/junxif/xin_lab/multiome")
from utils.eval import evaluate_model

results = evaluate_model(model, test_loader, test_adata, device=device, k=50)
print(results)

print("\n=== Learned GLOBAL α ===")
print("alpha =", torch.sigmoid(model.alpha).item())
