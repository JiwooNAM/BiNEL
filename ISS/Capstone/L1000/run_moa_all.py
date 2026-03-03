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
from collections import defaultdict
from scipy.stats import mannwhitneyu
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import json


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

def topk_hit_rate(y_true, prob, k=3):
    """
    y_true: (N,) int
    prob: (N, C) float probabilities (softmax)
    returns: float
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    k = int(k)
    if k <= 0:
        return np.nan
    k = min(k, prob.shape[1])

    topk = np.argpartition(-prob, kth=k-1, axis=1)[:, :k]
    hit = np.any(topk == y_true[:, None], axis=1)
    return float(hit.mean())


def select_genes_wilcoxon_union_per_moa(
    X: np.ndarray,          # (n_samples, n_genes)  float
    gene_names: list,       # length n_genes
    y: np.ndarray,          # (n_samples,) int labels
    n_classes: int,
    top_n: int,
    avg_round="round",      # "round" = 반올림, "floor"/"ceil"도 가능
    min_k: int = 1,
    two_sided: bool = True,
):
    """
    For each class c:
      compare X[y==c] vs X[y!=c] for every gene with Mann-Whitney U test
      score = abs(mean_diff) * -log10(p)
      pick top k genes per class
    union them, and cap to top_n by best score across classes.

    returns: (genes_union_list, meta_dict)
      meta_dict has: k_per_class, per_class_top_genes (dict), gene_best_score (dict)
    """
    assert X.ndim == 2
    n, g = X.shape
    assert g == len(gene_names)

    n_moa = n_classes
    if top_n <= 0:
        return [], dict(k=0, per_class_top_genes={}, gene_best_score={})

    # k = round(top_n / n_moa)
    raw = top_n / float(n_moa)
    if avg_round == "ceil":
        k = int(math.ceil(raw))
    elif avg_round == "floor":
        k = int(math.floor(raw))
    else:
        k = int(np.rint(raw))

    if k < min_k:
        return [], dict(k=k, per_class_top_genes={}, gene_best_score={})

    per_class_top = {}
    gene_best_score = {}

    # precompute global means for speed (optional)
    for c in range(n_classes):
        idx_in = (y == c)
        idx_out = ~idx_in
        Xin = X[idx_in]
        Xout = X[idx_out]

        if Xin.shape[0] < 2 or Xout.shape[0] < 2:
            per_class_top[c] = []
            continue

        mean_in = Xin.mean(axis=0)
        mean_out = Xout.mean(axis=0)
        diff = mean_in - mean_out  # signed
        absdiff = np.abs(diff)

        # p-values per gene
        pvals = np.ones(g, dtype=np.float64)
        alt = "two-sided" if two_sided else "greater"  # (보통 two-sided 추천)
        for j in range(g):
            # Mann-Whitney U (Wilcoxon rank-sum)
            try:
                pvals[j] = mannwhitneyu(Xin[:, j], Xout[:, j], alternative=alt).pvalue
            except Exception:
                pvals[j] = 1.0

        # 안정화: p=0 방지
        pvals = np.clip(pvals, 1e-300, 1.0)
        score = absdiff * (-np.log10(pvals))

        # top-k indices
        top_idx = np.argpartition(-score, kth=min(k, g-1)-1)[:min(k, g)]
        # 정렬(내림차순)
        top_idx = top_idx[np.argsort(-score[top_idx])]

        genes_c = [gene_names[j] for j in top_idx.tolist()]
        per_class_top[c] = genes_c

        # union score update (best score across classes)
        for j in top_idx:
            gn = gene_names[int(j)]
            sc = float(score[int(j)])
            if (gn not in gene_best_score) or (sc > gene_best_score[gn]):
                gene_best_score[gn] = sc

    # union + cap to top_n by best score
    union_genes = list(gene_best_score.keys())
    union_genes.sort(key=lambda gn: gene_best_score[gn], reverse=True)
    union_genes = union_genes[:min(top_n, len(union_genes))]

    meta = dict(
        k=k,
        per_class_top_genes=per_class_top,   # class index -> genes
        gene_best_score=gene_best_score,     # gene -> best score
    )
    return union_genes, meta


def _save_cm(cm, path, labels):
    payload = {
        "labels": labels,
        "cm": cm.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _safe_sheet_name(s: str) -> str:
    # Excel sheet name rules: <=31 chars, no : \ / ? * [ ]
    bad = [":", "\\", "/", "?", "*", "[", "]"]
    for b in bad:
        s = s.replace(b, "_")
    return s[:31]

def save_cm_store_npz(cm_store, out_path: Path):
    """
    cm_store: dict[(top_n, modality)] -> list of records
      record: {gene_mode, gene_id, fold, cm(np.array), labels(list[str])}

    Saves as a single .npz with:
      - meta.json (bytes) containing index of entries
      - each cm stored as cm__{idx}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = []
    arrays = {}

    idx = 0
    for (topn, modality), recs in cm_store.items():
        for r in recs:
            cm = np.asarray(r["cm"], dtype=np.float32)
            key = f"cm__{idx}"
            arrays[key] = cm
            meta.append(dict(
                idx=idx,
                top_n=str(topn),
                modality=str(modality),
                gene_mode=str(r.get("gene_mode", "")),
                gene_id=str(r.get("gene_id", "")),
                fold=int(r.get("fold", -1)),
                labels=list(r.get("labels", [])),
                cm_key=key,
            ))
            idx += 1

    meta_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    np.savez_compressed(out_path, meta=meta_json, **arrays)


def export_confusion_matrices_xlsx(cm_store, xlsx_path: Path, drop_nan_label=True):
    """
    Sheets: one per (top_n, modality)

    Each sheet contains ONLY 3 blocks (if present):
      - ranked
      - wilcoxon_union
      - random (mean over rand* per fold)

    For each block:
      1) SUM over folds (count CM)
      2) ROW-NORM of that SUM (truth-normalized)
    """

    def _sort_key(item):
        (topn, modality), _ = item
        return (int(topn) if str(topn).isdigit() else 10**9, str(modality))

    def _row_norm(cm):
        cm = np.asarray(cm, dtype=np.float64)
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return cm / denom

    def _maybe_drop_nan(cm, labels):
        if not drop_nan_label:
            return cm, labels
        nan_idx = [i for i, lab in enumerate(labels) if str(lab).lower() == "nan"]
        if not nan_idx:
            return cm, labels
        keep = [i for i in range(len(labels)) if i not in nan_idx]
        cm2 = cm[np.ix_(keep, keep)]
        labels2 = [labels[i] for i in keep]
        return cm2, labels2

    def _write_block(writer, sheet, row, title, labels, cm_sum):
        # title
        pd.DataFrame([[title]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=row)
        row += 1

        # SUM (count)
        cm_sum, labels2 = _maybe_drop_nan(np.asarray(cm_sum), labels)
        df_sum = pd.DataFrame(cm_sum, index=labels2, columns=labels2)
        pd.DataFrame([["SUM over folds (count)"]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=row)
        row += 1
        df_sum.to_excel(writer, sheet_name=sheet, startrow=row)
        row += (len(labels2) + 2)

        # ROW-NORM
        cm_rn = _row_norm(cm_sum)
        df_rn = pd.DataFrame(cm_rn, index=labels2, columns=labels2)
        pd.DataFrame([["ROW-NORM of SUM (truth-normalized)"]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=row)
        row += 1
        df_rn.to_excel(writer, sheet_name=sheet, startrow=row)
        row += (len(labels2) + 3)

        return row

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for (topn, modality), recs in sorted(cm_store.items(), key=_sort_key):
            if not recs:
                continue

            labels = list(recs[0]["labels"])
            sheet = _safe_sheet_name(f"top{topn}_{modality}")

            # header
            row = 0
            hdr = pd.DataFrame([[f"top_n={topn}", f"modality={modality}", f"n_records={len(recs)}"]],
                               columns=["key1", "key2", "key3"])
            hdr.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
            row += 2

            # Collect per gene_mode
            # We'll build:
            #  ranked_fold: fold -> [cm]  (usually one)
            #  wu_fold:     fold -> [cm]  (usually one)
            #  rnd_fold:    fold -> [cm]  (multiple rand*; we will average within fold)
            ranked = defaultdict(list)
            wu     = defaultdict(list)
            rnd    = defaultdict(list)

            for r in recs:
                gm = str(r.get("gene_mode", "")).lower()
                gid = str(r.get("gene_id", ""))
                fold = int(r.get("fold", -1))
                cm = np.asarray(r["cm"], dtype=np.float64)

                if gm == "ranked":
                    ranked[fold].append(cm)
                elif gm in ("wilcoxon_union", "wu", "wilcoxon"):
                    wu[fold].append(cm)
                elif gm == "random" or gid.startswith("rand"):
                    rnd[fold].append(cm)
                else:
                    # ignore others (e.g., CP if you don't want it here)
                    pass

            # collapse helpers
            def _sum_over_folds(d_fold_to_list):
                """If multiple per fold, average them within fold then SUM across folds."""
                if not d_fold_to_list:
                    return None
                fold_cms = []
                for f, cms in sorted(d_fold_to_list.items()):
                    if len(cms) == 1:
                        fold_cms.append(cms[0])
                    else:
                        # average within fold (e.g., random rand* per fold)
                        fold_cms.append(np.mean(np.stack(cms, axis=0), axis=0))
                return np.sum(np.stack(fold_cms, axis=0), axis=0)

            cm_rank_sum = _sum_over_folds(ranked)
            cm_wu_sum   = _sum_over_folds(wu)
            cm_rnd_sum  = _sum_over_folds(rnd)  # ✅ random: mean within fold, then sum folds

            # write 3 blocks
            if cm_rank_sum is not None:
                row = _write_block(writer, sheet, row, "ranked (r0)", labels, cm_rank_sum)

            if cm_wu_sum is not None:
                row = _write_block(writer, sheet, row, "wilcoxon_union (wu0)", labels, cm_wu_sum)

            if cm_rnd_sum is not None:
                row = _write_block(writer, sheet, row, "random (mean over rand* per fold)", labels, cm_rnd_sum)




def compute_metrics(y_true, y_pred, y_prob, avg, n_classes):
    """
    y_true: (n,)
    y_pred: (n,)
    y_prob: (n, C) softmax probs
    """
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"]  = float(f1_score(y_true, y_pred, average=avg))

    # multiclass AUROC (OVR). 클래스가 폴드에 빠진 경우 에러날 수 있어서 안전 처리
    try:
        # roc_auc_score는 (n, C) probs + labels가 모두 존재해야 안정적
        out["auroc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average=avg)
        )
    except Exception:
        out["auroc"] = np.nan

    # confusion matrix (MoA x MoA)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    out["cm"] = cm
    return out

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
    mm = preprocessing.StandardScaler()
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
    ap.add_argument("--out_dir", default="./results/Jiwoo/CM_HW")

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

    ap.add_argument("--topk", type=int, default=3, help="Top-K for hit rate (recall@K)")
    ap.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")


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
    #n_splits = 5 # if args.dataset == "CDRP-bio" else 10
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

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

    # moa_col 만든 직후
    merg[moa_col] = merg[moa_col].replace({"nan": np.nan, "none": np.nan, "": np.nan})
    merg = merg.dropna(subset=[moa_col]).copy()

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

    labels = list(le.classes_)  # index -> MoA string
    metrics_out_dir = out_dir / "metrics"
    cms_out_dir = out_dir / "confusion_matrices"
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    cms_out_dir.mkdir(parents=True, exist_ok=True)

    gene_pool = [g for g in l1k_features_full if g in df_all.columns]
    X_ge_all = df_all[gene_pool].values.astype("float32")
    n_classes = int(df_all["y"].nunique())
    


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

    cm_store = defaultdict(list)
    # key: (top_n, modality)  -> list of dicts:
    #   {gene_mode, gene_id, fold, cm(np.array), labels(list[str])}

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

        # ---- wilcoxon per-MoA union genes (NEW) ----
        if topn_tag.isdigit():
            top_n_int = int(topn_tag)
        else:
            top_n_int = 978

        wil_genes, wil_meta = select_genes_wilcoxon_union_per_moa(
            X=X_ge_all,
            gene_names=gene_pool,
            y=y_all,
            n_classes=n_classes,
            top_n=top_n_int,
            avg_round="round",
            min_k=1,
            two_sided=True,
        )

        # k<1이면 wil_genes가 []로 나오니까 자동으로 skip 가능
        if len(wil_genes) > 0:
            gene_sets.append(("wilcoxon_union", "wu0", wil_genes))
            (out_dir / "wilcoxon_union_genes").mkdir(parents=True, exist_ok=True)
            pd.Series(wil_genes).to_csv(out_dir / "wilcoxon_union_genes" / f"wilcoxon_union_top{topn_tag}_{args.dataset}.txt",
                                        index=False, header=False)

            # MoA별 top-k도 같이 저장
            per_moa_dir = out_dir / "wilcoxon_union_genes" / f"per_moa_topk_top{topn_tag}_{args.dataset}"
            per_moa_dir.mkdir(parents=True, exist_ok=True)
            for c, genes_c in wil_meta["per_class_top_genes"].items():
                moa_name = labels[c] if c < len(labels) else str(c)
                pd.Series(genes_c).to_csv(per_moa_dir / f"class{c}__{moa_name}.txt", index=False, header=False)


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
            m_cp_metrics = compute_metrics(y_te, pred_cp_te, prob_cp_te, avg, n_classes)

            topk_cp = topk_hit_rate(y_te, prob_cp_te, k=args.topk)


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

                m_ge_metrics = compute_metrics(y_te, pred_ge_te, prob_ge_te, avg, n_classes)
                m_ef_metrics = compute_metrics(y_te, pred_ef_te, prob_ef_te, avg, n_classes)
                m_lf_metrics = compute_metrics(y_te, pred_lf_te, prob_lf_te, avg, n_classes)

                topk_ge = topk_hit_rate(y_te, prob_ge_te, k=args.topk)
                topk_ef = topk_hit_rate(y_te, prob_ef_te, k=args.topk)
                topk_lf = topk_hit_rate(y_te, prob_lf_te, k=args.topk)


                # ---- store confusion matrices for Excel ----
                # NOTE: CP는 gene-set과 무관하지만, 탭 구성(TopN/Modality) 통일을 위해 gene_mode/gene_id 같이 저장해둠.
                #       (중복이 싫으면 CP 저장은 fold_cache 생성 시점에 1회만 저장하도록 옮기면 됨.)

                cm_store[(topn_tag, "CP")].append(dict(
                    gene_mode=mode, gene_id=gene_id, fold=fold, cm=m_cp_metrics["cm"], labels=labels
                ))
                cm_store[(topn_tag, "GE")].append(dict(
                    gene_mode=mode, gene_id=gene_id, fold=fold, cm=m_ge_metrics["cm"], labels=labels
                ))
                cm_store[(topn_tag, "Early Fusion")].append(dict(
                    gene_mode=mode, gene_id=gene_id, fold=fold, cm=m_ef_metrics["cm"], labels=labels
                ))
                cm_store[(topn_tag, "Late Fusion")].append(dict(
                    gene_mode=mode, gene_id=gene_id, fold=fold, cm=m_lf_metrics["cm"], labels=labels
                ))


                rows += [
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                        fold=fold, modality="CP",
                        acc=m_cp_metrics["acc"], f1=m_cp_metrics["f1"], auroc=m_cp_metrics["auroc"], topk=topk_cp),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                        fold=fold, modality="GE",
                        acc=m_ge_metrics["acc"], f1=m_ge_metrics["f1"], auroc=m_ge_metrics["auroc"], topk=topk_ge),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                        fold=fold, modality="Early Fusion",
                        acc=m_ef_metrics["acc"], f1=m_ef_metrics["f1"], auroc=m_ef_metrics["auroc"], topk=topk_ef),
                    dict(dataset=args.dataset, top_n=topn_tag, gene_mode=mode, gene_id=gene_id,
                        fold=fold, modality="Late Fusion",
                        acc=m_lf_metrics["acc"], f1=m_lf_metrics["f1"], auroc=m_lf_metrics["auroc"], topk=topk_lf),
                ]

                


                

                # 파일명에 top_n / mode / gene_id / fold / modality 포함
                base = f"ds={args.dataset}__topn={topn_tag}__mode={mode}__gid={gene_id}__fold={fold}"

                # CP는 ranked/random에 상관없이 동일하지만, gene_mode마다 중복 저장이 싫으면 mode/gid 없이 저장해도 됨.
                _save_cm(m_cp_metrics["cm"], cms_out_dir / f"{base}__mod=CP.json", labels)
                _save_cm(m_ge_metrics["cm"], cms_out_dir / f"{base}__mod=GE.json", labels)
                _save_cm(m_ef_metrics["cm"], cms_out_dir / f"{base}__mod=EF.json", labels)
                _save_cm(m_lf_metrics["cm"], cms_out_dir / f"{base}__mod=LF.json", labels)

        if args.verbose:
            print(f"[DONE] TOP_N={topn_tag} ranked_genes={len(ranked_genes)} random_sets={args.n_random}")

    res = pd.DataFrame(rows)

    # fold-level -> summary (mean/std)
    summary = (
        res.groupby(["dataset","top_n","gene_mode","modality"], as_index=False)
        .agg(
            acc_mean=("acc","mean"), acc_std=("acc","std"),
            f1_mean=("f1","mean"),   f1_std=("f1","std"),
            auroc_mean=("auroc","mean"), auroc_std=("auroc","std"),
            topk_mean=("topk","mean"), topk_std=("topk","std"),
        )
    )


    res_path = out_dir / f"compare_ranked_vs_random_{args.dataset}.csv"
    sum_path = out_dir / f"compare_ranked_vs_random_{args.dataset}_summary.csv"
    res.to_csv(res_path, index=False)
    summary.to_csv(sum_path, index=False)

    cm_xlsx_path = out_dir / f"confusion_matrices_by_topn_modality_{args.dataset}.xlsx"
    export_confusion_matrices_xlsx(cm_store, cm_xlsx_path)
    cm_npz_path = out_dir / f"cm_store_{args.dataset}.npz"
    save_cm_store_npz(cm_store, cm_npz_path)

    if args.verbose:
        print("[SAVED]", cm_npz_path)


        if args.verbose:
            print("[SAVED]", cm_xlsx_path)


        if args.verbose:
            print("[SAVED]", res_path)
            print("[SAVED]", sum_path)


if __name__ == "__main__":
    main()
