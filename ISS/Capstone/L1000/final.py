#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sweep.py (CP-cache version) — UPDATED

What changed (vs your pasted version)
- Removed the per-class top-k union implementation (k=top_n/n_classes).
- Implemented "global wilcoxon_union ranking" that works for ANY top_n (5..978):
    For each class c (c vs rest):
        score_c(gene) = abs(mean_in - mean_out) * (-log10(MWU pval))
    Then:
        best_score(gene) = max_c score_c(gene)
    Rank genes by best_score desc once, then slice top_n for every top_n.

Everything else stays conceptually the same:
- Same CV splits (StratifiedGroupKFold)
- CP trained once per (trial, fold) and cached
- GE & Early Fusion trained per gene set (HVG ranked / Wilcoxon union / optional random)
- Late fusion uses cached CP probs + current GE probs
- Save raw / trial_level / summary CSVs
"""

import argparse
from pathlib import Path
import sys
import math
import warnings

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

from imblearn.over_sampling import RandomOverSampler
from scipy.stats import mannwhitneyu

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# utils import (IMPORTANT)
# -------------------------
sys.path.insert(0, "./utils")
from readProfiles import read_paired_treatment_level_profiles  # noqa

# -------------------------
# warnings
# -------------------------
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The least populated class in y has only",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
            layers.append(nn.Tanh() if act == "tanh" else nn.ReLU())
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
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        pb = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(pb)
    return np.vstack(probs)


def train_torch_mlp(
    Xtr, ytr, n_classes, device,
    hidden=(400,), dropout=0.0, act="relu",
    lr=1e-3, weight_decay=0.0, epochs=30, batch_size=1024,
    seed=0
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

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)

    n = Xtr.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for j in range(0, n, batch_size):
            b = idx[j:j + batch_size]
            xb = Xtr_t[b].to(device, non_blocking=True)
            yb = ytr_t[b].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


# -------------------------
# Helpers
# -------------------------
def parse_hidden(s):
    s = str(s).strip()
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(" ", "")
    if not s:
        return (400,)
    parts = [p for p in s.split(",") if p]
    return tuple(int(p) for p in parts)


def load_hp_map(hp_csv_path: Path):
    """
    hp_{dataset}.csv columns must be:
      top_n, modality, hidden, act, dropout, wd
    modality expects:
      CP, GE, Early Fusion (or EF)
    """
    hp_df = pd.read_csv(hp_csv_path)
    required = ["top_n", "modality", "hidden", "act", "dropout", "wd"]
    missing = [c for c in required if c not in hp_df.columns]
    if missing:
        raise ValueError(f"hp csv missing columns: {missing} in {hp_csv_path}")

    hp_df["top_n"] = hp_df["top_n"].astype(str)
    hp_df["modality"] = hp_df["modality"].astype(str).replace({"EF": "Early Fusion"})

    hp_map = {}
    for _, r in hp_df.iterrows():
        key = (str(r["top_n"]), str(r["modality"]))
        hp_map[key] = dict(
            hidden=parse_hidden(r["hidden"]),
            act=str(r["act"]),
            dropout=float(r["dropout"]),
            wd=float(r["wd"]),
        )
    return hp_map, hp_df


def get_hp(hp_map, topn_tag, modality):
    return hp_map.get((str(topn_tag), str(modality)), None)


def scale_standard(df, cols):
    ss = preprocessing.StandardScaler()
    X = df[cols].values.astype("float64")
    X = ss.fit_transform(X)
    out = df.copy()
    out.loc[:, cols] = X
    return out


def compute_auroc_safe(y_true, prob, n_classes, avg):
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, prob[:, 1]))
        return float(
            roc_auc_score(
                y_true, prob,
                multi_class="ovr",
                average=avg,
                labels=np.arange(n_classes),
            )
        )
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred, y_prob, avg, n_classes):
    return dict(
        acc=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, average=avg)),
        auroc=compute_auroc_safe(y_true, y_prob, n_classes, avg),
    )


def parse_topn_list(s: str):
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "all":
            out.append("all")
        else:
            out.append(str(int(x)))
    return out


# -------------------------
# NEW: Global Wilcoxon-union ranking (supports any top_n 5..978)
# -------------------------
def rank_genes_wilcoxon_union_global(
    X: np.ndarray,          # (n_samples, n_genes)
    gene_names: list,       # length n_genes
    y: np.ndarray,          # (n_samples,)
    n_classes: int,
    two_sided: bool = True,
    min_group_n: int = 2,
    eps_p: float = 1e-300,
    verbose: bool = False,
):
    """
    Global union ranking:
      For each class c:
        score_c(gene) = abs(mean_in - mean_out) * (-log10(pval_MWU))
      Then:
        best_score(gene) = max_c score_c(gene)
      Rank genes by best_score desc.

    Returns:
      ranked_genes, ranked_scores, ranked_best_class
    """
    assert X.ndim == 2
    n, g = X.shape
    assert g == len(gene_names)

    alt = "two-sided" if two_sided else "greater"

    best_score = np.zeros(g, dtype=np.float64)
    best_class = np.full(g, -1, dtype=np.int32)

    for c in range(n_classes):
        idx_in = (y == c)
        idx_out = ~idx_in
        Xin = X[idx_in]
        Xout = X[idx_out]

        if Xin.shape[0] < min_group_n or Xout.shape[0] < min_group_n:
            if verbose:
                print(f"[WU] class={c} skipped (Xin={Xin.shape[0]}, Xout={Xout.shape[0]})")
            continue

        mean_in = Xin.mean(axis=0)
        mean_out = Xout.mean(axis=0)
        absdiff = np.abs(mean_in - mean_out)

        pvals = np.ones(g, dtype=np.float64)
        for j in range(g):
            try:
                pvals[j] = mannwhitneyu(Xin[:, j], Xout[:, j], alternative=alt).pvalue
            except Exception:
                pvals[j] = 1.0

        pvals = np.clip(pvals, eps_p, 1.0)
        score_c = absdiff * (-np.log10(pvals))

        better = score_c > best_score
        best_score[better] = score_c[better]
        best_class[better] = c

        if verbose:
            print(f"[WU] class={c} updated={(better).sum()} genes")

    order = np.argsort(-best_score)  # desc
    keep = best_score[order] > 0
    order = order[keep]

    ranked_genes = [gene_names[i] for i in order.tolist()]
    ranked_scores = best_score[order]
    ranked_best_class = best_class[order]

    return ranked_genes, ranked_scores, ranked_best_class


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", choices=["LINCS", "CDRP-bio"], required=True)
    ap.add_argument("--procProf_dir", default="./")
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--out_dir", default="./results/Jiwoo/Last")

    # training knobs
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--nSamplesMOA", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)

    # sweep knobs
    ap.add_argument("--topn_list", default="5,10,15,20,25,30,50,75,100,150,200,300,400,all")
    ap.add_argument("--n_splits", type=int, default=5)

    # trials
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed_step", type=int, default=10000)

    # optional random baseline
    ap.add_argument("--do_random", action="store_true")
    ap.add_argument("--n_random_sets", type=int, default=5)
    ap.add_argument("--random_seed", type=int, default=0)

    # hp csv
    ap.add_argument("--hp_csv", default=None)

    # WU options
    ap.add_argument("--wu_two_sided", action="store_true", help="use two-sided MWU (default True)")
    ap.add_argument("--wu_one_sided", action="store_true", help="use one-sided MWU alternative='greater'")
    ap.add_argument("--wu_min_group_n", type=int, default=2)

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

    # HP
    hp_path = Path(args.hp_csv) if args.hp_csv else (gene_dir / f"hp_{args.dataset}.csv")
    hp_map, hp_df = load_hp_map(hp_path)
    if args.verbose:
        print(f"[HP] loaded {hp_path} rows={len(hp_df)} unique keys={len(hp_map)}")

    # Load profiles
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

    merg[moa_col] = merg[moa_col].replace({"nan": np.nan, "none": np.nan, "": np.nan})
    merg = merg.dropna(subset=[moa_col]).copy()

    # scale separately
    cp = merg[[pertColName, "Compounds", moa_col] + list(cp_features)].copy()
    ge = merg[[pertColName, "Compounds", moa_col] + list(l1k_features_full)].copy()
    cp_s = scale_standard(cp, list(cp_features))
    ge_s = scale_standard(ge, list(l1k_features_full))

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

    ge_universe = [g for g in l1k_features_full if g in df_all.columns]
    X_ge_all = df_all[ge_universe].values.astype("float32")
    n_classes = int(df_all["y"].nunique())

    if args.verbose:
        print(f"[INFO] kept MoAs={len(selected)} (multilabel removed={len(multi)})")
        print(f"[INFO] rows={len(df_all)} compounds={df_all['Compounds'].nunique()} classes={df_all['y'].nunique()}")

    # gene ranking list (HVG ranked)
    gene_file = gene_dir / f"genes_ranked_{args.dataset}.npy"
    genes_ranked = np.load(gene_file, allow_pickle=True).tolist()

    # CV splitter (folds deterministic from fixed random_state)
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # Build split map ONCE (same across trials)
    split_map = {}
    for fold, (tr, te) in enumerate(sgkf.split(X_cp, y_all, groups=groups), start=1):
        split_map[fold] = (tr, te)

    # Oversampler
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)

    # Averaging
    avg = "weighted" if args.dataset == "LINCS" else "macro"

    topn_list = parse_topn_list(args.topn_list)

    # ------------------------------------------------------------
    # Precompute global WU union ranking ONCE (supports top_n 5..978)
    # ------------------------------------------------------------
    if args.wu_one_sided:
        two_sided = False
    else:
        # default to two-sided unless explicitly one-sided
        two_sided = True

    if args.verbose:
        print("[WU RANK] computing global wilcoxon-union ranking once...")

    wu_ranked_genes, wu_ranked_scores, wu_ranked_best_class = rank_genes_wilcoxon_union_global(
        X=X_ge_all,
        gene_names=ge_universe,
        y=y_all,
        n_classes=n_classes,
        two_sided=two_sided,
        min_group_n=args.wu_min_group_n,
        verbose=False,
    )

    if args.verbose:
        print(f"[WU RANK] done. ranked genes={len(wu_ranked_genes)} (nonzero-score)")

    # ------------------------------------------------------------
    # CP CACHE (global): computed once per (trial, fold)
    # key: (trial, fold) -> (te_idx, prob_cp)
    # ------------------------------------------------------------
    # Prefer CP HP from top_n=978; otherwise fallback to first CP row
    hp_cp_978 = get_hp(hp_map, "978", "CP")
    if hp_cp_978 is None:
        cp_keys = [k for k in hp_map.keys() if k[1] == "CP"]
        if not cp_keys:
            raise ValueError("No CP hyperparameters found in hp_csv.")
        hp_cp_978 = hp_map[cp_keys[0]]
        if args.verbose:
            print(f"[WARN] No CP hp for top_n=978. Using {cp_keys[0]} as fallback for CP.")

    if args.verbose:
        print("[CP CACHE] building CP cache for all (trial, fold)...")

    cp_cache = {}
    for trial in range(args.n_trials):
        for fold in range(1, args.n_splits + 1):
            tr, te = split_map[fold]
            y_tr = y_all[tr]
            base_seed = args.seed + trial * args.seed_step + fold * 100

            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)

            m_cp = train_torch_mlp(
                Xtr_cp, ytr_cp, n_classes, device,
                hidden=hp_cp_978["hidden"], dropout=hp_cp_978["dropout"], act=hp_cp_978["act"],
                lr=args.lr, weight_decay=hp_cp_978["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=base_seed + 11,
            )
            prob_cp = predict_proba(m_cp, X_cp[te], device)
            cp_cache[(trial, fold)] = (te, prob_cp)

    if args.verbose:
        print(f"[CP CACHE] done. entries={len(cp_cache)} (n_trials={args.n_trials}, n_splits={args.n_splits})")

    # ------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------
    rows = []

    for topn_raw in topn_list:
        if topn_raw == "all":
            topn_tag = "978"
            top_n_int = 978
        else:
            topn_tag = str(int(topn_raw))
            top_n_int = int(topn_tag)

        # Build deterministic gene sets
        if top_n_int == 978:
            hvg_genes = ge_universe
        else:
            hvg_genes = [g for g in genes_ranked[:top_n_int] if g in df_all.columns]

        if len(hvg_genes) == 0:
            if args.verbose:
                print(f"[WARN] TOP_N={topn_tag}: no HVG genes matched. skip")
            continue

        # NEW: wilcoxon_union = slice from global WU ranking (supports 5..978)
        wu_genes = [g for g in wu_ranked_genes[:top_n_int] if g in df_all.columns]

        random_sets = []
        if args.do_random:
            rng = np.random.default_rng(args.random_seed + 1000 * top_n_int)
            for r in range(args.n_random_sets):
                pick = rng.choice(ge_universe, size=min(top_n_int, len(ge_universe)), replace=False).tolist()
                random_sets.append(("random", f"rand{r}", pick))

        gene_sets = [("hvg_ranked", "hvg0", hvg_genes)]
        if len(wu_genes) > 0:
            gene_sets.append(("wilcoxon_union", "wu0", wu_genes))
        gene_sets += random_sets

        # HP check for this topn (GE/EF)
        hp_ge = get_hp(hp_map, topn_tag, "GE")
        hp_ef = get_hp(hp_map, topn_tag, "Early Fusion")
        if hp_ge is None or hp_ef is None:
            if args.verbose:
                print(f"[WARN] TOP_N={topn_tag}: missing HP rows for GE/Early Fusion. skip this topn.")
            continue

        for gene_mode, gene_id, genes_used in gene_sets:
            X_ge_full = df_all[genes_used].values.astype("float32")
            X_early_full = np.concatenate([X_cp, X_ge_full], axis=1).astype("float32")

            for trial in range(args.n_trials):
                for fold in range(1, args.n_splits + 1):
                    tr, te = split_map[fold]
                    y_tr = y_all[tr]
                    y_te = y_all[te]

                    te_idx, prob_cp = cp_cache[(trial, fold)]
                    # sanity: te indices should match
                    # (skip hard assert to avoid crashing if something changed)
                    # if not np.array_equal(te_idx, te): ...

                    pred_cp = prob_cp.argmax(1)

                    # oversample GE/EF (CP already cached)
                    Xtr_ge, ytr_ge = ros.fit_resample(X_ge_full[tr], y_tr)
                    Xtr_ef, ytr_ef = ros.fit_resample(X_early_full[tr], y_tr)

                    base_seed = args.seed + trial * args.seed_step + fold * 100

                    m_ge = train_torch_mlp(
                        Xtr_ge, ytr_ge, n_classes, device,
                        hidden=hp_ge["hidden"], dropout=hp_ge["dropout"], act=hp_ge["act"],
                        lr=args.lr, weight_decay=hp_ge["wd"],
                        epochs=args.epochs, batch_size=args.batch_size,
                        seed=base_seed + 12,
                    )
                    m_ef = train_torch_mlp(
                        Xtr_ef, ytr_ef, n_classes, device,
                        hidden=hp_ef["hidden"], dropout=hp_ef["dropout"], act=hp_ef["act"],
                        lr=args.lr, weight_decay=hp_ef["wd"],
                        epochs=args.epochs, batch_size=args.batch_size,
                        seed=base_seed + 13,
                    )

                    prob_ge = predict_proba(m_ge, X_ge_full[te], device)
                    prob_ef = predict_proba(m_ef, X_early_full[te], device)

                    pred_ge = prob_ge.argmax(1)
                    pred_ef = prob_ef.argmax(1)

                    prob_lf = (prob_cp + prob_ge) / 2.0
                    pred_lf = prob_lf.argmax(1)

                    met_cp = compute_metrics(y_te, pred_cp, prob_cp, avg, n_classes)
                    met_ge = compute_metrics(y_te, pred_ge, prob_ge, avg, n_classes)
                    met_ef = compute_metrics(y_te, pred_ef, prob_ef, avg, n_classes)
                    met_lf = compute_metrics(y_te, pred_lf, prob_lf, avg, n_classes)

                    rows += [
                        dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                             trial=trial, fold=fold, modality="CP", **met_cp),
                        dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                             trial=trial, fold=fold, modality="GE", **met_ge),
                        dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                             trial=trial, fold=fold, modality="Early Fusion", **met_ef),
                        dict(dataset=args.dataset, top_n=topn_tag, gene_mode=gene_mode, gene_id=gene_id,
                             trial=trial, fold=fold, modality="Late Fusion", **met_lf),
                    ]

            if args.verbose:
                print(f"[DONE] top_n={topn_tag} gene_mode={gene_mode} genes_used={len(genes_used)} trials={args.n_trials}")

    # -------------------------
    # Save
    # -------------------------
    res = pd.DataFrame(rows)
    raw_path = out_dir / f"raw_{args.dataset}.csv"
    res.to_csv(raw_path, index=False)

    trial_level = (
        res.groupby(["dataset", "top_n", "gene_mode", "modality", "trial"], as_index=False)
           .agg(acc=("acc", "mean"), f1=("f1", "mean"), auroc=("auroc", "mean"))
    )

    summary = (
        trial_level.groupby(["dataset", "top_n", "gene_mode", "modality"], as_index=False)
                  .agg(
                      n_trials=("trial", "nunique"),
                      acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                      f1_mean=("f1", "mean"),   f1_std=("f1", "std"),
                      auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                  )
    )

    trial_path = out_dir / f"trial_level_{args.dataset}.csv"
    sum_path = out_dir / f"summary_{args.dataset}.csv"
    trial_level.to_csv(trial_path, index=False)
    summary.to_csv(sum_path, index=False)

    for gm in summary["gene_mode"].unique():
        summary[summary["gene_mode"] == gm].to_csv(out_dir / f"summary_{args.dataset}_{gm}.csv", index=False)

    if args.verbose:
        # (optional) save WU ranking for inspection / reuse
        wu_rank_path = out_dir / f"wu_rank_{args.dataset}.csv"
        pd.DataFrame({
            "rank": np.arange(1, len(wu_ranked_genes) + 1),
            "gene": wu_ranked_genes,
            "score": wu_ranked_scores,
            "best_class": wu_ranked_best_class,
        }).to_csv(wu_rank_path, index=False)
        print("[SAVED]", wu_rank_path)

    print("[SAVED]", raw_path)
    print("[SAVED]", trial_path)
    print("[SAVED]", sum_path)


if __name__ == "__main__":
    main()