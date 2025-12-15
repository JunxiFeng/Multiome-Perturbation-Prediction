import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def evaluate_model(
    model,
    test_loader,
    test_adata,
    pseudocount=0.1,
    device="cuda",
    k=50
):
    """
    EXACT PerturBench-style evaluation for cosine(logFC).
    Works for BOTH:
        - baseline dataloaders: (xb, pb, cenc, cdec)
        - GET/ATAC models:      (xb, pb, cenc, cdec, ct_idx)
    """

    # --------------------------------------------------
    # AUTO-DETECT loader type (4 or 5 tensors per batch)
    # --------------------------------------------------
    first_batch = next(iter(test_loader))
    five_inputs = (len(first_batch) == 5)

    # ------------------------
    # 1) Run model on test set
    # ------------------------
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in test_loader:

            if five_inputs:
                xb, pb, cenc, cdec, ct_idx = batch
                xb, pb, cenc, cdec, ct_idx = (
                    xb.to(device), pb.to(device),
                    cenc.to(device), cdec.to(device),
                    ct_idx.to(device)
                )
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(
                        xb,
                        p=pb,
                        cov_enc=cenc,
                        cov_dec=cdec,
                        ct_idx=ct_idx,
                    )
            else:
                xb, pb, cenc, cdec = batch
                xb, pb, cenc, cdec = (
                    xb.to(device), pb.to(device),
                    cenc.to(device), cdec.to(device)
                )
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(
                        xb,
                        p=pb,
                        cov_enc=cenc,
                        cov_dec=cdec,
                    )

            all_true.append(xb.cpu())
            all_pred.append(pred.cpu())

    true = torch.cat(all_true).numpy()
    pred = torch.cat(all_pred).numpy()

    obs = test_adata.obs

    # ------------------------
    # 2) Global metrics
    # ------------------------
    rmse = np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))

    def cosine(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    pearsons = []
    cosines = []

    for i in range(true.shape[0]):
        try:
            pearsons.append(pearsonr(true[i], pred[i])[0])
        except:
            pearsons.append(np.nan)
        cosines.append(cosine(true[i], pred[i]))

    # -----------------------------
    # 3) PerturBench-style logFC
    # -----------------------------
    perts = obs["condition"].unique()

    # Compute control average
    ctrl_mask = (obs["condition"] == "ctrl")
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells 'ctrl' found in test set.")

    true_ctrl = true[ctrl_mask].mean(axis=0)
    pred_ctrl = pred[ctrl_mask].mean(axis=0)

    per_pert = {}

    for pert in perts:
        if pert == "ctrl":
            continue

        idx = np.where(obs["condition"] == pert)[0]
        if len(idx) < 2:
            continue

        avg_true = true[idx].mean(axis=0)
        avg_pred = pred[idx].mean(axis=0)

        # PerturBench EXACT definition
        true_lfc = np.log2(avg_true + pseudocount) - np.log2(true_ctrl + pseudocount)
        pred_lfc = np.log2(avg_pred + pseudocount) - np.log2(pred_ctrl + pseudocount)

        # Pearson
        try:
            lfc_corr = pearsonr(true_lfc, pred_lfc)[0]
        except:
            lfc_corr = np.nan

        # Cosine
        lfc_cosine = cosine(true_lfc, pred_lfc)

        # top-k recall
        true_top = np.argsort(true_lfc)[-k:]
        pred_top = np.argsort(pred_lfc)[-k:]
        recall_k = len(set(true_top) & set(pred_top)) / k

        per_pert[pert] = {
            "n_cells": int(len(idx)),
            "logfc_corr": float(lfc_corr),
            "logfc_cosine": float(lfc_cosine),
            f"top{k}_recall": float(recall_k),
        }

    # ------------------------------------------------
    # 4) Global averaged logFC cosine (PerturBench-like)
    # ------------------------------------------------
    stacked_true_lfc = []
    stacked_pred_lfc = []

    for pert in per_pert.keys():
        pert_idx = np.where(obs["condition"] == pert)[0]
        if len(pert_idx) < 2:
            continue

        avg_true = true[pert_idx].mean(axis=0)
        avg_pred = pred[pert_idx].mean(axis=0)

        stacked_true_lfc.append(
            np.log2(avg_true + pseudocount) - np.log2(true_ctrl + pseudocount)
        )
        stacked_pred_lfc.append(
            np.log2(avg_pred + pseudocount) - np.log2(pred_ctrl + pseudocount)
        )

    if len(stacked_true_lfc) > 0:
        global_true_lfc = np.mean(stacked_true_lfc, axis=0)
        global_pred_lfc = np.mean(stacked_pred_lfc, axis=0)
        global_logfc_cosine = cosine(global_true_lfc, global_pred_lfc)
    else:
        global_logfc_cosine = np.nan

    # ------------------------
    # 5) Return
    # ------------------------
    global_metrics = {
        "rmse": rmse,
        "pearson_mean": float(np.nanmean(pearsons)),
        "cosine_mean": float(np.nanmean(cosines)),
        "global_logfc_cosine": float(global_logfc_cosine),
    }

    return {
        "global": global_metrics,
        "per_pert": per_pert,
    }
