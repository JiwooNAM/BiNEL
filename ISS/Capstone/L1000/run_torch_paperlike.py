#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Torch reimplementation of paper-like MoA prediction with ranked/random gene sweep.

- Load paired treatment-level profiles (CP + L1000) via utils/readProfiles.py
- Harmonize MoA labels like the notebook
- Scale CP and GE separately: StandardScaler -> MinMax(0,1) fit on full set (paper-like)
- Select MoAs by counting unique compounds per MoA (sample 1 row per compound), keep MoAs with > nSamplesMOA compounds,
  remove multilabel MoAs containing '|'
- Outer CV: StratifiedGroupKFold by compound (PERT[:13]) (CDRP-bio default 10 folds, LINCS default 5)
- Oversampling: RandomOverSampler(sampling_strategy="not majority")
- Torch MLP for CP / GE / Early Fusion; Late Fusion = average of CP+GE probabilities
- Gene sweep:
    topn_list = [5,10,15,20,25,30,50,75,100,150,200,300,978]
    gene_mode: ranked (1 set) + random (n_random sets)
    GE uses top-n gene set; EF concatenates CP (all CP feats) + GE (top-n)
- Hyperparam tuning:
    "per_fold": inner StratifiedKFold on training set (after oversampling) to choose best hp
    "once": tune only on first outer fold per (topn, modality, gene_set_id), cache for remaining folds

Outputs:
- out_dir / f"sweep_torch_{dataset}.csv"  (fold-level results)
- out_dir / f"sweep_torch_{dataset}_summary.csv" (mean/std grouped)

Requirements:
- numpy, pandas, scikit-learn, imbalanced-learn, torch
- your repo: utils/readProfiles.py providing read_paired_treatment_level_profiles
- a ranked gene list: gene_sets/genes_ranked_{dataset}.npy (list of gene feature names)
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import f1_score

from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# utils import
# -------------------------
sys.path.insert(0, "./utils")
from readProfiles import read_paired_treatment_level_profiles  # noqa


# -------------------------
# Torch MLP
# -------------------------
class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden=(400,), dropout=0.0, act="tanh", n_classes=10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh() if act == "tanh" else nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=4096):
    model.eval()
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        pb = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(pb)
    return np.vstack(probs)


def train_torch_mlp(
    Xtr, ytr, n_classes, device,
    hidden=(400,), dropout=0.0, act="tanh",
    lr=1e-3, weight_decay=0.0, epochs=60, batch_size=1024,
    seed=0,
    lr_schedule="constant",  # {"constant","adaptive"}
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TorchMLP(
        in_dim=Xtr.shape[1],
        hidden=hidden,
        dropout=dropout,
        act=act,
        n_classes=n_classes,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # "adaptive" approximation of sklearn's learning_rate="adaptive":
    # Reduce LR when val loss plateaus (we'll step per epoch using a simple split when requested).
    # In training-only context, we'll step on training loss.
    if lr_schedule == "adaptive":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, verbose=False
        )
    else:
        sched = None

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)

    n = Xtr.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        running = 0.0
        for j in range(0, n, batch_size):
            b = idx[j:j + batch_size]
            xb = Xtr_t[b].to(device, non_blocking=True)
            yb = ytr_t[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item())

        if sched is not None:
            sched.step(running)

    return model


# -------------------------
# preprocessing helpers (paper-like)
# -------------------------
def scale_like_notebook(df, feat_cols):
    out = df.copy()
    ss = preprocessing.StandardScaler()
    mm = preprocessing.MinMaxScaler((0, 1))
    X = out[feat_cols].values.astype("float64")
    X = mm.fit_transform(ss.fit_transform(X))
    out.loc[:, feat_cols] = X
    return out


def prepare_merged_scaled(procProf_dir, dataset, profileType, filter_perts, repCorrFilePath, nRep=1):
    pertColName = "PERT"
    moa_col = "Metadata_MoA"
    filter_repCorr_params = [filter_perts, repCorrFilePath]

    merg, cp_features, l1k_features_full = read_paired_treatment_level_profiles(
        procProf_dir, dataset, profileType, filter_repCorr_params, nRep
    )

    merg = merg.copy()
    if dataset == "LINCS":
        # harmonize MoA field naming like your earlier code
        merg[moa_col] = merg["Metadata_moa"]
        merg.loc[merg["Metadata_moa"].isnull(), moa_col] = (
            merg.loc[merg["Metadata_moa"].isnull(), "moa"].astype(str).str.lower()
        )
        merg["Compounds"] = merg[pertColName].astype(str).str[0:13]
    else:
        merg[moa_col] = merg["Metadata_moa"].astype(str).str.lower()
        merg["Compounds"] = merg[pertColName].astype(str).str[0:13]

    cp = merg[[pertColName, "Compounds", moa_col] + list(cp_features)].copy()
    ge = merg[[pertColName, "Compounds", moa_col] + list(l1k_features_full)].copy()

    cp_s = scale_like_notebook(cp, list(cp_features))
    ge_s = scale_like_notebook(ge, list(l1k_features_full))

    merged_scaled = pd.concat([cp_s, ge_s], axis=1)
    merged_scaled = merged_scaled.loc[:, ~merged_scaled.columns.duplicated()].copy()
    merged_scaled["Compounds"] = merged_scaled[pertColName].astype(str).str[0:13]

    return merged_scaled, list(cp_features), list(l1k_features_full), moa_col, pertColName


def select_moas_like_notebook(merged_scaled, moa_col, nSamplesMOA):
    tmp = (
        merged_scaled.groupby(["Compounds"]).sample(1, random_state=0)
        .groupby([moa_col]).size()
        .reset_index(name="size")
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    selected = tmp[tmp["size"] > nSamplesMOA][moa_col].tolist()
    multi = [m for m in selected if isinstance(m, str) and "|" in m]
    selected = [m for m in selected if m not in multi]
    return selected, multi, tmp


# -------------------------
# tuning: inner CV (sklearn GridSearchCV analog)
# -------------------------
def tune_hp_inner_cv(
    Xtr, ytr,
    n_classes, device,
    hp_grid,
    cv_inner=3,
    avg="macro",
    lr=1e-3,
    epochs=60,
    batch_size=1024,
    seed=0,
):
    skf = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=seed)
    best = None
    best_score = -1.0

    for hp in hp_grid:
        scores = []
        for inner_fold, (itr, iva) in enumerate(skf.split(Xtr, ytr), start=1):
            m = train_torch_mlp(
                Xtr[itr], ytr[itr],
                n_classes=n_classes,
                device=device,
                hidden=hp["hidden"],
                dropout=hp["dropout"],
                act=hp["act"],
                lr=lr,
                weight_decay=hp["weight_decay"],
                epochs=epochs,
                batch_size=batch_size,
                seed=seed + inner_fold * 17,
                lr_schedule=hp["lr_schedule"],
            )
            prob = predict_proba(m, Xtr[iva], device=device)
            pred = prob.argmax(1)
            scores.append(f1_score(ytr[iva], pred, average=avg))

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best = hp

    return best, best_score


def make_hp_grid_like_sklearn():
    # sklearn grid:
    # hidden_layer_sizes: (100,), (200,), (400,)
    # activation: tanh, relu
    # alpha: 0.0001, 0.05   -> map to weight_decay
    # learning_rate: constant, adaptive -> map to lr_schedule
    grid = []
    for hidden in [(100,), (200,), (400,)]:
        for act in ["tanh", "relu"]:
            for wd in [1e-4, 5e-2]:
                for lr_schedule in ["constant", "adaptive"]:
                    grid.append(dict(
                        hidden=hidden,
                        act=act,
                        dropout=0.0,          # sklearn MLP had no dropout; keep 0 for closest match
                        weight_decay=wd,      # alpha ≈ L2
                        lr_schedule=lr_schedule,
                    ))
    return grid


# -------------------------
# main sweep
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["CDRP-bio", "LINCS"], required=True)

    ap.add_argument("--procProf_dir", default="./")
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--nRep", type=int, default=1)

    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--genes_ranked", default=None,
                    help="Path to ranked genes .npy; default gene_sets/genes_ranked_{dataset}.npy")

    ap.add_argument("--out_dir", default="./results/Jiwoo/MoA_TorchSweep")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nSamplesMOA", type=int, default=4)

    ap.add_argument("--topn_list", default="5,10,15,20,25,30,50,75,100,150,200,300,978")
    ap.add_argument("--n_random", type=int, default=5)

    # outer folds
    ap.add_argument("--n_folds_cdrp", type=int, default=10)
    ap.add_argument("--n_folds_lincs", type=int, default=5)

    # training
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=1024)

    # tuning
    ap.add_argument("--tune", action="store_true", help="If set, do inner-CV hp tuning (GridSearchCV-like).")
    ap.add_argument("--cv_inner", type=int, default=3)
    ap.add_argument("--tune_scope", choices=["per_fold", "once"], default="per_fold",
                    help="per_fold: tune every outer fold; once: tune on first outer fold then reuse cached hp.")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load + preprocess merged
    merged_scaled, cp_features, l1k_features_full, moa_col, pertColName = prepare_merged_scaled(
        procProf_dir=args.procProf_dir,
        dataset=args.dataset,
        profileType=args.profileType,
        filter_perts=args.filter_perts,
        repCorrFilePath=args.repCorrFilePath,
        nRep=args.nRep,
    )

    selected_moas, multi_removed, _ = select_moas_like_notebook(merged_scaled, moa_col, args.nSamplesMOA)
    df_all = merged_scaled[merged_scaled[moa_col].isin(selected_moas)].reset_index(drop=True).copy()

    le = preprocessing.LabelEncoder()
    le.fit(selected_moas)
    df_all["y"] = le.transform(df_all[moa_col].tolist())

    y_all = df_all["y"].values.astype(int)
    groups = df_all["Compounds"].values

    # fixed CP matrix
    X_cp = df_all[cp_features].values.astype("float32")

    # averaging choice like earlier
    avg = "macro" if args.dataset == "CDRP-bio" else "weighted"

    # outer CV
    n_splits = args.n_folds_cdrp if args.dataset == "CDRP-bio" else args.n_folds_lincs
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # oversampler
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)

    # ranked genes
    gene_dir = Path(args.gene_dir)
    gene_rank_path = args.genes_ranked or str(gene_dir / f"genes_ranked_{args.dataset}.npy")
    genes_ranked = np.load(gene_rank_path, allow_pickle=True).tolist()

    # topn list
    topn_list = [t.strip() for t in args.topn_list.split(",") if t.strip()]

    # hp grid
    hp_grid = make_hp_grid_like_sklearn()

    if args.verbose:
        print(f"[INFO] device={device}")
        print(f"[INFO] dataset={args.dataset}")
        print(f"[INFO] kept MoAs={len(selected_moas)} (multilabel removed={len(multi_removed)})")
        print(f"[INFO] rows={len(df_all)} compounds={df_all['Compounds'].nunique()} classes={df_all['y'].nunique()}")
        print(f"[INFO] cp_dim={len(cp_features)} l1k_full_dim={len(l1k_features_full)}")
        print(f"[INFO] ranked genes file={gene_rank_path}")

    # cache for tune_scope=once
    # key: (topn_tag, gene_mode, gene_id, modality) -> best_hp dict
    hp_cache = {}

    rows = []

    n_classes = int(df_all["y"].nunique())

    # sweep topn
    for topn_tag in topn_list:
        # decide ranked set size
        if topn_tag.lower() == "all" or topn_tag == "978":
            ranked_genes = [g for g in l1k_features_full if g in df_all.columns]
            k = len(ranked_genes)
        else:
            k = int(topn_tag)
            ranked_genes = [g for g in genes_ranked[:k] if g in df_all.columns]

        if len(ranked_genes) == 0:
            if args.verbose:
                print(f"[WARN] topn={topn_tag}: no matched ranked genes, skip")
            continue

        gene_pool = [g for g in l1k_features_full if g in df_all.columns]
        # reproducible random per topn
        rng_topn = np.random.default_rng(args.seed + k)

        gene_sets = [("ranked", "r0", ranked_genes)]
        for r in range(args.n_random):
            pick = rng_topn.choice(gene_pool, size=len(ranked_genes), replace=False).tolist()
            gene_sets.append(("random", f"rand{r+1}", pick))

        if args.verbose:
            print(f"\n[TOPN {topn_tag}] ranked_k={len(ranked_genes)} random_sets={args.n_random}")

        # outer folds
        for fold, (tr, te) in enumerate(sgkf.split(X_cp, y_all, groups=groups), start=1):
            y_tr, y_te = y_all[tr], y_all[te]

            # ---- CP (train once per fold; no sweep effect) ----
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)

            # hp for CP (tune or default)
            if args.tune:
                key = (topn_tag, "cp", "cp", "CP")
                use_cache = (args.tune_scope == "once") and (key in hp_cache)
                if use_cache:
                    hp_cp = hp_cache[key]
                else:
                    hp_cp, _ = tune_hp_inner_cv(
                        Xtr_cp, ytr_cp,
                        n_classes=n_classes,
                        device=device,
                        hp_grid=hp_grid,
                        cv_inner=args.cv_inner,
                        avg=avg,
                        lr=args.lr,
                        epochs=max(10, args.epochs // 2),   # inner CV cheaper
                        batch_size=args.batch_size,
                        seed=args.seed + fold * 100 + 11,
                    )
                    if args.tune_scope == "once":
                        hp_cache[key] = hp_cp
            else:
                hp_cp = dict(hidden=(400,), act="tanh", dropout=0.0, weight_decay=0.0, lr_schedule="constant")

            m_cp = train_torch_mlp(
                Xtr_cp, ytr_cp,
                n_classes=n_classes,
                device=device,
                hidden=hp_cp["hidden"],
                dropout=hp_cp["dropout"],
                act=hp_cp["act"],
                lr=args.lr,
                weight_decay=hp_cp["weight_decay"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + fold * 100 + 111,
                lr_schedule=hp_cp["lr_schedule"],
            )

            prob_cp = predict_proba(m_cp, X_cp[te], device=device)
            pred_cp = prob_cp.argmax(1)
            f1_cp = f1_score(y_te, pred_cp, average=avg)

            # ---- gene-set loop: GE / EF / LF ----
            for gene_mode, gene_id, genes_used in gene_sets:
                X_ge = df_all[genes_used].values.astype("float32")
                X_ef = np.concatenate([X_cp, X_ge], axis=1).astype("float32")

                # oversample GE/EF
                Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
                Xtr_ef, ytr_ef = ros.fit_resample(X_ef[tr], y_tr)

                # tune hp for GE and EF if requested
                if args.tune:
                    # GE
                    k_ge = (topn_tag, gene_mode, gene_id, "GE")
                    use_cache_ge = (args.tune_scope == "once") and (k_ge in hp_cache)
                    if use_cache_ge:
                        hp_ge = hp_cache[k_ge]
                    else:
                        hp_ge, _ = tune_hp_inner_cv(
                            Xtr_ge, ytr_ge, n_classes, device,
                            hp_grid=hp_grid, cv_inner=args.cv_inner, avg=avg,
                            lr=args.lr, epochs=max(10, args.epochs // 2),
                            batch_size=args.batch_size,
                            seed=args.seed + fold * 100 + 12,
                        )
                        if args.tune_scope == "once":
                            hp_cache[k_ge] = hp_ge

                    # EF
                    k_ef = (topn_tag, gene_mode, gene_id, "EF")
                    use_cache_ef = (args.tune_scope == "once") and (k_ef in hp_cache)
                    if use_cache_ef:
                        hp_ef = hp_cache[k_ef]
                    else:
                        hp_ef, _ = tune_hp_inner_cv(
                            Xtr_ef, ytr_ef, n_classes, device,
                            hp_grid=hp_grid, cv_inner=args.cv_inner, avg=avg,
                            lr=args.lr, epochs=max(10, args.epochs // 2),
                            batch_size=args.batch_size,
                            seed=args.seed + fold * 100 + 13,
                        )
                        if args.tune_scope == "once":
                            hp_cache[k_ef] = hp_ef
                else:
                    hp_ge = dict(hidden=(400,), act="tanh", dropout=0.0, weight_decay=0.0, lr_schedule="constant")
                    hp_ef = dict(hidden=(400,), act="tanh", dropout=0.0, weight_decay=0.0, lr_schedule="constant")

                # train GE
                m_ge = train_torch_mlp(
                    Xtr_ge, ytr_ge, n_classes, device,
                    hidden=hp_ge["hidden"], dropout=hp_ge["dropout"], act=hp_ge["act"],
                    lr=args.lr, weight_decay=hp_ge["weight_decay"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed + fold * 100 + 212,
                    lr_schedule=hp_ge["lr_schedule"],
                )
                prob_ge = predict_proba(m_ge, X_ge[te], device=device)
                pred_ge = prob_ge.argmax(1)
                f1_ge = f1_score(y_te, pred_ge, average=avg)

                # train EF
                m_ef = train_torch_mlp(
                    Xtr_ef, ytr_ef, n_classes, device,
                    hidden=hp_ef["hidden"], dropout=hp_ef["dropout"], act=hp_ef["act"],
                    lr=args.lr, weight_decay=hp_ef["weight_decay"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed + fold * 100 + 313,
                    lr_schedule=hp_ef["lr_schedule"],
                )
                prob_ef = predict_proba(m_ef, X_ef[te], device=device)
                pred_ef = prob_ef.argmax(1)
                f1_ef = f1_score(y_te, pred_ef, average=avg)

                # late fusion
                prob_lf = (prob_cp + prob_ge) / 2.0
                pred_lf = prob_lf.argmax(1)
                f1_lf = f1_score(y_te, pred_lf, average=avg)

                rows += [
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                         fold=fold, modality="CP", f1=f1_cp, n_genes=len(genes_used)),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                         fold=fold, modality="GE", f1=f1_ge, n_genes=len(genes_used)),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                         fold=fold, modality="Early Fusion", f1=f1_ef, n_genes=len(genes_used)),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                         fold=fold, modality="Late Fusion", f1=f1_lf, n_genes=len(genes_used)),
                ]

            if args.verbose:
                print(f"[{args.dataset}] topn={topn_tag} fold {fold}/{n_splits} CP_f1={f1_cp:.3f}")

    res = pd.DataFrame(rows)
    summary = (res.groupby(["dataset","top_n","gene_mode","modality"], as_index=False)
                 .agg(f1_mean=("f1","mean"), f1_std=("f1","std")))

    out_csv = out_dir / f"sweep_torch_{args.dataset}.csv"
    out_sum = out_dir / f"sweep_torch_{args.dataset}_summary.csv"
    res.to_csv(out_csv, index=False)
    summary.to_csv(out_sum, index=False)

    print("[SAVED]", out_csv)
    print("[SAVED]", out_sum)


if __name__ == "__main__":
    main()
