# Multiome Perturbation Prediction with Gated Latent Fusion

## Overview

We implement and compare **three model families**:

- **Baseline Latent Additive Model (RNA-only)**
- **GET-augmented Latent Additive Model with Learnable Gate**
- **ATAC-augmented Latent Additive Model with Global Gate**

All models predict **gene expression under perturbation** using reconstruction loss and are evaluated using downstream perturbation metrics.

---

## Models

### 1️⃣ Baseline: Latent Additive RNA Model

**No multiome information.**

#### Inputs
- Gene expression **x**
- Perturbation one-hot **p**
- Cell-type covariates **cov**

#### Formulation
```text
z_ctrl = gene_encoder([x, cov])
z_pert = pert_encoder(p)
z      = z_ctrl + z_pert
x̂      = decoder([z, cov])
```

This serves as the **reference model**.

---

### 2️⃣ GET-Augmented Model (Cell-Type–Aware, Gated)

**File:** `LatentAdditiveGET_Encoder_Gated`

This model integrates **cell-type–level GET embeddings** *(genes × d_get)* using a **learnable scalar gate**.

#### Inputs
- Gene expression **x**
- Perturbation one-hot **p**
- Cell-type covariates **cov**
- Cell-type GET embedding **GET_ct**

#### GET Processing
```text
GET_ct (genes × d_get)
→ mean over d_get → (genes,)
→ MLP → z_get
```

#### Formulation
```text
z_ctrl  = gene_encoder([x, cov])
z_pert  = pert_encoder(p)
z_rna   = z_ctrl + z_pert

α       = softplus(alpha_param)   # α > 0
z_fused = z_rna + α · z_get

x̂       = decoder([z_fused, cov])
```

#### Key Properties
- **Cell-type–aware** (GET indexed by cell type)
- **Memory-safe** (no per-cell GET storage)
- **Interpretable**: α directly quantifies GET contribution
- **Stable training** via softplus gate

---

### 3️⃣ ATAC-Augmented Model (Global Gate)

**File:** `LatentAdditiveATAC_GlobalGated`

This model integrates **cell-type ATAC embeddings** using a **single global scalar gate**.

#### Inputs
- Gene expression **x**
- Perturbation one-hot **p**
- Cell-type covariates **cov**
- Cell-type ATAC embedding **ATAC_ct**

#### Formulation
```text
z_ctrl  = gene_encoder([x, cov])
z_pert  = pert_encoder(p)
z_rna   = z_ctrl + z_pert

z_atac  = ATAC_encoder(ATAC_ct)
α       = sigmoid(alpha_param)    # α ∈ (0, 1)

z_total = concat(z_rna, α · z_atac, cov)
x̂       = decoder(z_total)
```

#### Key Properties
- **Single global α** shared across all cells
- Tests whether ATAC adds **global predictive signal**
- **Minimal inductive bias**, easy to interpret
