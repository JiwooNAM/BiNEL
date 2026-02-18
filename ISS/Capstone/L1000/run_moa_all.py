#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# utils import (IMPORTANT)
# -------------------------
sys.path.insert(0, "./utils")
from readProfiles import read_paired_treatment_level_profiles  # noqa

import warnings
warnings.filterwarnings("ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
    category=UserWarning,
)
warnings.filterwarnings("ignore",
    message="The least populated class in y has only",
    category=UserWarning,
)

# -------------------------
# Torch MLP
# -------------------------
class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 512), dropout=0.1, act="relu", n_classes=10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if act == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=2048):
    model.eval()
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        pb = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(pb)
    return np.vstack(probs)


def train_torch_mlp(Xtr, ytr, n_classes, device,
                    hidden=(400,), dropout=0.0, act="relu",
                    lr=1e-3, weight_decay=0.0, epochs=30, batch_size=1024,
                    seed=0):
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

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)

    n = Xtr.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for j in range(0, n, batch_size):
            b = idx[j:j+batch_size]
            xb = Xtr_t[b].to(device, non_blocking=True)
            yb = ytr_t[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


def _scale(df, cols):
    ss = preprocessing.StandardScaler()
    mm = preprocessing.MinMaxScaler((0, 1))
    X = df[cols].values.astype("float64")
    X = mm.fit_transform(ss.fit_transform(X))
    out = df.copy()
    out.loc[:, cols] = X
    return out


def _parse_hidden(hidden_str: str):
    return tuple(int(x) for x in str(hidden_str).split(",") if x.strip())


def load_hp_table(hp_path: Path):
    """
    hp csv columns expected:
      top_n,modality,hidden,act,dropout,wd
    modality in {CP,GE,EF} where EF means Early Fusion
    """
    hp = pd.read_csv(hp_path)
    # normalize
    hp["top_n"] = hp["top_n"].astype(str)
    hp["modality"] = hp["modality"].replace({"Early Fusion": "EF"}).astype(str)
    need = {"top_n","modality","hidden","act","dropout","wd"}
    miss = need - set(hp.columns)
    if miss:
        raise ValueError(f"HP file missing columns: {sorted(miss)} in {hp_path}")

    table = {}
    for _, r in hp.iterrows():
        key = (str(r["top_n"]), str(r["modality"]))
        table[key] = dict(
            hidden=_parse_hidden(r["hidden"]),
            act=str(r["act"]),
            dropout=float(r["dropout"]),
            wd=float(r["wd"]),
        )
    return table


def get_hp(table, topn_tag, modality, hp_fallback="error"):
    key = (str(topn_tag), modality)
    if key in table:
        return table[key]

    if hp_fallback == "default":
        # sensible default (너가 계속 쓰던 것 기반)
        return dict(hidden=(400,), act="tanh", dropout=0.0, wd=0.0)

    # error
    raise KeyError(f"missing hp for top_n={topn_tag}, modality={modality}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", choices=["LINCS", "CDRP-bio"], required=True)
    ap.add_argument("--procProf_dir", default="./")
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")

    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--out_dir", default="./results/Jiwoo/Genepanel_Torch")

    # train settings
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--nSamplesMOA", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)

    # comparison settings
    ap.add_argument("--topn_list", default="5,10,15,20,25,30,50,75,100,150,200,300,400,978")
    ap.add_argument("--n_random", type=int, default=5, help="how many random gene sets per top_n")
    ap.add_argument("--random_seed", type=int, default=123, help="base seed for random gene sampling")

    # hp
    ap.add_argument("--hp_fallback", choices=["error","default"], default="error",
                    help="what to do if hp missing: error or use default hp")

    # logging
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print("[DEVICE]", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_dir = Path(args.gene_dir)

    # hp load
    hp_path = gene_dir / f"hp_{args.dataset}.csv"
    hp_table = load_hp_table(hp_path)

    # folds
    n_splits = 5 if args.dataset == "LINCS" else 10
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # load profiles
    filter_repCorr_params = [args.filter_perts, args.repCorrFilePath]
    pertColName = "PERT"
    moa_col = "Metadata_MoA"

    merg, cp_features, l1k_features_full = read_paired_treatment_level_profiles(
        args.procProf_dir, args.dataset, args.profileType, filter_repCorr_params, 1
    )

    # harmonize
    merg = merg.copy()
    if args.dataset == "LINCS":
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
    cp_s = _scale(cp, list(cp_features))
    ge_s = _scale(ge, list(l1k_features_full))

    merged_scaled = pd.concat([cp_s, ge_s], axis=1)
    merged_scaled = merged_scaled.loc[:, ~merged_scaled.columns.duplicated()].copy()
    merged_scaled["Compounds"] = merged_scaled[pertColName].astype(str).str[0:13]

    # select MoAs by compound count
    tmp = (
        merged_scaled.groupby(["Compounds"]).sample(1, random_state=0)
        .groupby([moa_col]).size()
        .reset_index(name="size")
        .sort_values("size", ascending=False)
    )
    selected = tmp[tmp["size"] > args.nSamplesMOA][moa_col].tolist()
    multi = [m for m in selected if isinstance(m, str) and "|" in m]
    selected = [m for m in selected if m not in multi]

    df_all = merged_scaled[merged_scaled[moa_col].isin(selected)].reset_index(drop=True).copy()
    le = preprocessing.LabelEncoder()
    le.fit(selected)
    df_all["y"] = le.transform(df_all[moa_col].tolist())

    y_all = df_all["y"].values
    groups = df_all["Compounds"].values
    X_cp = df_all[list(cp_features)].values.astype("float32")

    # ranked genes
    gene_file = gene_dir / f"genes_ranked_{args.dataset}.npy"
    genes_ranked = np.load(gene_file, allow_pickle=True).tolist()

    # topn list parse
    topn_list = []
    for x in args.topn_list.split(","):
        x = x.strip()
        if not x:
            continue
        topn_list.append(x)

    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)
    avg = "weighted" if args.dataset == "LINCS" else "macro"

    rows = []

    # Precompute CP per fold ONCE (핵심: ranked/random 비교에서 CP 동일하게 유지)
    # fold마다 train indices가 top_n/gene-set과 무관하므로 재사용 가능
    if args.verbose:
        print(f"[INFO] rows={len(df_all)} compounds={df_all['Compounds'].nunique()} classes={df_all['y'].nunique()}")

    fold_cache = {}  # fold -> dict with trained CP model predictions per test set (by index list)

    for topn_raw in topn_list:
        # decide topn_tag + gene pool size
        if str(topn_raw).lower() == "all":
            topn_tag = "978"
            ranked_genes = list(l1k_features_full)
        else:
            topn_tag = str(topn_raw)
            if topn_tag == "978":
                ranked_genes = list(l1k_features_full)
            else:
                k = int(topn_tag)
                ranked_genes = [g for g in genes_ranked[:k] if g in df_all.columns]

        if len(ranked_genes) == 0:
            if args.verbose:
                print(f"[WARN] TOP_N={topn_tag}: no matched ranked genes, skip")
            continue

        # random generator for this top_n (reproducible)
        rng = np.random.default_rng(args.random_seed + int(topn_tag))
        gene_pool = [g for g in l1k_features_full if g in df_all.columns]

        # Build gene-sets to evaluate: ranked (1 set) + random (n_random sets)
        gene_sets = [("ranked", "r0", ranked_genes)]
        for r in range(args.n_random):
            pick = rng.choice(gene_pool, size=len(ranked_genes), replace=False).tolist()
            gene_sets.append(("random", f"rand{r+1}", pick))

        # For each fold: train CP once, then for each gene-set train GE/EF and do LF
        for fold, (tr, te) in enumerate(sgkf.split(X_cp, y_all, groups=groups), start=1):
            y_tr, y_te = y_all[tr], y_all[te]
            n_classes = int(df_all["y"].nunique())

            # ---- CP (train once per fold) ----
            if fold not in fold_cache:
                # oversample CP
                Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)

                hp_cp = get_hp(hp_table, topn_tag, "CP", hp_fallback=args.hp_fallback)
                m_cp = train_torch_mlp(
                    Xtr_cp, ytr_cp, n_classes, device,
                    hidden=hp_cp["hidden"], dropout=hp_cp["dropout"], act=hp_cp["act"],
                    lr=args.lr, weight_decay=hp_cp["wd"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed + fold * 100 + 11,
                )
                prob_cp_te = predict_proba(m_cp, X_cp[te], device)
                pred_cp_te = prob_cp_te.argmax(1)
                fold_cache[fold] = dict(prob_cp_te=prob_cp_te, pred_cp_te=pred_cp_te)

            prob_cp_te = fold_cache[fold]["prob_cp_te"]
            pred_cp_te = fold_cache[fold]["pred_cp_te"]

            # CP f1 (same for ranked/random by design)
            f1_cp = f1_score(y_te, pred_cp_te, average=avg)

            # ---- GE/EF per gene-set ----
            for mode, gene_id, genes_used in gene_sets:
                X_ge = df_all[genes_used].values.astype("float32")
                X_ef = np.concatenate([X_cp, X_ge], axis=1).astype("float32")

                # oversample GE/EF
                Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
                Xtr_ef, ytr_ef = ros.fit_resample(X_ef[tr], y_tr)

                hp_ge = get_hp(hp_table, topn_tag, "GE", hp_fallback=args.hp_fallback)
                hp_ef = get_hp(hp_table, topn_tag, "EF", hp_fallback=args.hp_fallback)

                m_ge = train_torch_mlp(
                    Xtr_ge, ytr_ge, n_classes, device,
                    hidden=hp_ge["hidden"], dropout=hp_ge["dropout"], act=hp_ge["act"],
                    lr=args.lr, weight_decay=hp_ge["wd"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed + fold * 100 + 12,
                )
                m_ef = train_torch_mlp(
                    Xtr_ef, ytr_ef, n_classes, device,
                    hidden=hp_ef["hidden"], dropout=hp_ef["dropout"], act=hp_ef["act"],
                    lr=args.lr, weight_decay=hp_ef["wd"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed + fold * 100 + 13,
                )

                prob_ge_te = predict_proba(m_ge, X_ge[te], device)
                prob_ef_te = predict_proba(m_ef, X_ef[te], device)

                pred_ge_te = prob_ge_te.argmax(1)
                pred_ef_te = prob_ef_te.argmax(1)

                prob_lf_te = (prob_cp_te + prob_ge_te) / 2.0
                pred_lf_te = prob_lf_te.argmax(1)

                f1_ge = f1_score(y_te, pred_ge_te, average=avg)
                f1_ef = f1_score(y_te, pred_ef_te, average=avg)
                f1_lf = f1_score(y_te, pred_lf_te, average=avg)

                rows += [
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                         fold=fold, modality="CP", f1=f1_cp),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                         fold=fold, modality="GE", f1=f1_ge),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                         fold=fold, modality="Early Fusion", f1=f1_ef),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                         fold=fold, modality="Late Fusion", f1=f1_lf),
                ]

        if args.verbose:
            print(f"[DONE] TOP_N={topn_tag} ranked_genes={len(ranked_genes)} random_sets={args.n_random}")

    res = pd.DataFrame(rows)

    # fold-level -> summary (mean/std)
    summary = (res.groupby(["dataset","top_n","gene_mode","modality"], as_index=False)
                 .agg(f1_mean=("f1","mean"), f1_std=("f1","std")))

    res_path = out_dir / f"compare_ranked_vs_random_{args.dataset}.csv"
    sum_path = out_dir / f"compare_ranked_vs_random_{args.dataset}_summary.csv"
    res.to_csv(res_path, index=False)
    summary.to_csv(sum_path, index=False)

    if args.verbose:
        print("[SAVED]", res_path)
        print("[SAVED]", sum_path)


if __name__ == "__main__":
    main()
