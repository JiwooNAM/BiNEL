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
from sklearn.model_selection import StratifiedShuffleSplit


# -------------------------
# utils import (IMPORTANT)
# -------------------------
# THIS_FILE = Path(__file__).resolve()
# 예: App3.Gene_panel/run_moa_topn_torch.py 라면
# utils 폴더가 L1000/utils 에 있다고 가정하고 2~3단계 위로 잡기
# UTILS_DIR = (THIS_FILE.parents[1] / "utils").resolve()  # 필요하면 parents[2]로 조정
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


def tune_torch_mlp(
    X, y, n_classes, device, avg,
    lr, batch_size,
    seed,
    # 후보들(논문 느낌)
    hidden_grid=((100,), (200,), (400,)),
    act_grid=("relu", "tanh"),
    wd_grid=(0.0, 1e-4, 1e-3),
    dropout_grid=(0.0, 0.1),
    # 튜닝 때는 짧게
    tune_epochs=20,
):
    """
    fold의 train set에서 stratified split으로 val을 만들고,
    val F1(best) 기준으로 hp를 고른 뒤 best dict 반환.
    """
    # stratified val split (train 내에서 10%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, va_idx = next(sss.split(X, y))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    best = None
    best_score = -1.0

    # 작은 grid
    for hidden in hidden_grid:
        for act in act_grid:
            for wd in wd_grid:
                for dr in dropout_grid:
                    m = train_torch_mlp(
                        Xtr, ytr, n_classes, device,
                        hidden=hidden, dropout=dr, act=act,
                        lr=lr, weight_decay=wd,
                        epochs=tune_epochs, batch_size=batch_size,
                        seed=seed,
                    )
                    prob = predict_proba(m, Xva, device)
                    pred = prob.argmax(1)
                    score = f1_score(yva, pred, average=avg)

                    if score > best_score:
                        best_score = score
                        best = dict(hidden=hidden, act=act, wd=wd, dropout=dr)

    return best, best_score



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
    for ep in range(epochs):
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



def proba_align(prob_src, classes_src, classes_ref):
    src_map = {c: i for i, c in enumerate(classes_src)}
    out = np.zeros((prob_src.shape[0], len(classes_ref)), dtype=float)
    for j, c in enumerate(classes_ref):
        if c in src_map:
            out[:, j] = prob_src[:, src_map[c]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["LINCS", "CDRP-bio"], required=True)
    ap.add_argument("--procProf_dir", default="./")  # readProfiles가 기대하는 root
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--out_dir", default="./results/Jiwoo/Genepanel_Torch")

    # training knobs (고정해서 빠르게 TOP_N sweep)
    ap.add_argument("--hidden", default="400")      # "400" or "512,512"
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--act", choices=["relu", "tanh"], default="relu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--nSamplesMOA", type=int, default=4)  # size > nSamplesMOA
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_dir = Path(args.gene_dir)
    TOP_N_LIST = [1] #, 5, 10, 15, 25, 30, 50, 75, 100, 150, 200, 300, 400, "all"]

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

    # harmonize (논문 스타일)
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

    # scale Standard -> MinMax (0,1) per feature
    def scale(df, cols):
        ss = preprocessing.StandardScaler()
        mm = preprocessing.MinMaxScaler((0, 1))
        X = df[cols].values.astype("float64")
        X = mm.fit_transform(ss.fit_transform(X))
        out = df.copy()
        out.loc[:, cols] = X
        return out

    cp = merg[[pertColName, "Compounds", moa_col] + list(cp_features)].copy()
    ge = merg[[pertColName, "Compounds", moa_col] + list(l1k_features_full)].copy()
    cp_s = scale(cp, list(cp_features))
    ge_s = scale(ge, list(l1k_features_full))

    merged_scaled = pd.concat([cp_s, ge_s], axis=1)
    merged_scaled = merged_scaled.loc[:, ~merged_scaled.columns.duplicated()].copy()
    merged_scaled["Compounds"] = merged_scaled[pertColName].astype(str).str[0:13]

    # select MoAs by compound count (논문)
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

    print(f"[INFO] kept MoAs={len(selected)} (multilabel removed={len(multi)})")
    print(f"[INFO] rows={len(df_all)} compounds={df_all['Compounds'].nunique()} classes={df_all['y'].nunique()}")

    y_all = df_all["y"].values
    groups = df_all["Compounds"].values
    X_cp = df_all[list(cp_features)].values.astype("float32")

    # ranked genes
    gene_file = gene_dir / f"genes_ranked_{args.dataset}.npy"
    genes_ranked = np.load(gene_file, allow_pickle=True)

    # parse hidden
    hidden = tuple(int(x) for x in str(args.hidden).split(",") if x.strip())

    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)

    rows = []
    dumps = []

    avg = "weighted" if args.dataset == "LINCS" else "macro"

    for TOP_N in TOP_N_LIST:
        if TOP_N == "all":
            # ✅ 논문처럼 진짜 978 강제
            available = list(l1k_features_full)
            topn_tag = "978"
        else:
            wanted = genes_ranked[:int(TOP_N)].tolist()
            # genes_ranked가 컬럼과 mismatch일 수 있으니 필터
            available = [g for g in wanted if g in df_all.columns]
            topn_tag = str(TOP_N)

        if len(available) == 0:
            print(f"[WARN] TOP_N={TOP_N}: no matched genes, skip")
            continue

        X_ge = df_all[available].values.astype("float32")
        X_early = np.concatenate([X_cp, X_ge], axis=1).astype("float32")


        for fold, (tr, te) in enumerate(sgkf.split(X_early, y_all, groups=groups), start=1):
            y_tr, y_te = y_all[tr], y_all[te]

            # oversample each modality like paper code
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)
            Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
            Xtr_ef, ytr_ef = ros.fit_resample(X_early[tr], y_tr)

            n_classes = int(df_all["y"].nunique())
            classes_ref = np.arange(n_classes)

            # train models on GPU
            # ---- (fold loop 안) ----
            n_classes = int(df_all["y"].nunique())

            # oversample each modality like paper code
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)
            Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
            Xtr_ef, ytr_ef = ros.fit_resample(X_early[tr], y_tr)

            # ✅ 논문 느낌: fold train에서 간단 튜닝 후 best로 full-train
            # 튜닝은 짧게, 최종은 args.epochs
            tune_epochs = min(20, args.epochs)  # 선발전

            best_cp, score_cp = tune_torch_mlp(
                Xtr_cp, ytr_cp, n_classes, device, avg,
                lr=args.lr, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 1,
                tune_epochs=tune_epochs,
            )
            best_ge, score_ge = tune_torch_mlp(
                Xtr_ge, ytr_ge, n_classes, device, avg,
                lr=args.lr, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 2,
                tune_epochs=tune_epochs,
            )
            best_ef, score_ef = tune_torch_mlp(
                Xtr_ef, ytr_ef, n_classes, device, avg,
                lr=args.lr, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 3,
                tune_epochs=tune_epochs,
            )

            # 최종 학습(full epochs)
            m_cp = train_torch_mlp(
                Xtr_cp, ytr_cp, n_classes, device,
                hidden=best_cp["hidden"], dropout=best_cp["dropout"], act=best_cp["act"],
                lr=args.lr, weight_decay=best_cp["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 11,
            )
            m_ge = train_torch_mlp(
                Xtr_ge, ytr_ge, n_classes, device,
                hidden=best_ge["hidden"], dropout=best_ge["dropout"], act=best_ge["act"],
                lr=args.lr, weight_decay=best_ge["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 12,
            )
            m_ef = train_torch_mlp(
                Xtr_ef, ytr_ef, n_classes, device,
                hidden=best_ef["hidden"], dropout=best_ef["dropout"], act=best_ef["act"],
                lr=args.lr, weight_decay=best_ef["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 13,
            )

            # (옵션) 로그
            print(f"[TUNE fold={fold} top_n={topn_tag}] "
                f"CP best={best_cp} valF1={score_cp:.4f} | "
                f"GE best={best_ge} valF1={score_ge:.4f} | "
                f"EF best={best_ef} valF1={score_ef:.4f}")


            # probs
            prob_cp = predict_proba(m_cp, X_cp[te], device)
            prob_ge = predict_proba(m_ge, X_ge[te], device)
            prob_ef = predict_proba(m_ef, X_early[te], device)

            pred_cp = prob_cp.argmax(1)
            pred_ge = prob_ge.argmax(1)
            pred_ef = prob_ef.argmax(1)

            # late fusion (classes are 0..C-1 already)
            prob_lf = (prob_cp + prob_ge) / 2.0
            pred_lf = prob_lf.argmax(1)

            f1_cp = f1_score(y_te, pred_cp, average=avg)
            f1_ge = f1_score(y_te, pred_ge, average=avg)
            f1_ef = f1_score(y_te, pred_ef, average=avg)
            f1_lf = f1_score(y_te, pred_lf, average=avg)

            rows += [
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="CP", f1=f1_cp),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="GE", f1=f1_ge),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="Early Fusion", f1=f1_ef),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="Late Fusion", f1=f1_lf),
            ]

            dumps.append(pd.DataFrame({
                "dataset": args.dataset,
                "top_n": topn_tag,
                "fold": fold,
                "PERT": df_all.loc[te, pertColName].values,
                "Compounds": df_all.loc[te, "Compounds"].values,
                "y_true": y_te,
                "CP": pred_cp,
                "GE": pred_ge,
                "Early Fusion": pred_ef,
                "Late Fusion": pred_lf,
            }))

        print(f"[DONE] TOP_N={topn_tag} (genes used={len(available)})")

    res = pd.DataFrame(rows)
    pred = pd.concat(dumps, ignore_index=True)

    res_path = out_dir / f"summary_{args.dataset}_torch_mlp_topn.csv"
    xlsx_path = out_dir / f"pred_{args.dataset}_torch_mlp_topn.xlsx"

    res.to_csv(res_path, index=False)
    pred.to_excel(xlsx_path, index=False)

    print("[SAVED]", res_path)
    print("[SAVED]", xlsx_path)


if __name__ == "__main__":
    main()
