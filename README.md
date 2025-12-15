📘 Baseline: Latent Additive Model

The baseline predicts expression under perturbations using:

expression vector x

perturbation vector p

cell-type covariates cov

The model computes:

z_ctrl = gene_encoder([x, cov])
z_pert = pert_encoder(p)
z = z_ctrl + z_pert
x_hat = decoder([z, cov])


This contains no multiome information.

🚀 GET-Augmented Architectures

Each mapped cell type has a GET embedding matrix:

adata.uns["GET_embeddings"][celltype] → (n_genes × d_get)


These matrices encode gene-level regulatory embedding vectors.

We test two fusion strategies.

🅰️ Option A — Cell-Level GET Projection
Concept

GET embeddings are projected into a cell-level regulatory summary:

cell_GET = X @ GET_ct        # (1 × d_get)


Then concatenated into the model latent space:

z = concat( z_ctrl + z_pert , cell_GET , cov )
x_hat = decoder(z)

Pros

Memory-efficient

Captures global regulatory influence

Simple & effective

🅱️ Option B — Gene-Level GET Bias
Concept

Use GET to supply a gene-specific correction:

bias_g = W_get @ GET_ct[g]      # scalar
x_hat_g = base_prediction_g + bias_g

Pros

Very interpretable

Adjusts each gene’s output individually

Good for improving DE/logFC accuracy