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
        xb = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
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
            b = idx[j:j+batch_size]
            xb = Xtr_t[b].to(device, non_blocking=True)
            yb = ytr_t[b].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


def parse_hidden(s):
    """
    "400" -> (400,)
    "512,512" -> (512,512)
    "(512, 512)" -> (512,512)
    """
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
    Returns dict keyed by (top_n(str), modality(str)) -> hp dict
    """
    hp_df = pd.read_csv(hp_csv_path)
    required = ["top_n", "modality", "hidden", "act", "dropout", "wd"]
    missing = [c for c in required if c not in hp_df.columns]
    if missing:
        raise ValueError(f"hp csv missing columns: {missing} in {hp_csv_path}")

    hp_df["top_n"] = hp_df["top_n"].astype(str)
    hp_df["modality"] = hp_df["modality"].astype(str)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["LINCS", "CDRP-bio"], required=True)
    ap.add_argument("--procProf_dir", default="./")  # readProfiles가 기대하는 root
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--out_dir", default="./results/Jiwoo/Genepanel_Torch")

    # training knobs (global default; hp csv에는 lr/bs/epochs 없으니 여기값 씀)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--nSamplesMOA", type=int, default=4)  # size > nSamplesMOA
    ap.add_argument("--seed", type=int, default=1)

    # hp / gene mode
    ap.add_argument("--hp_csv", default=None,
                    help="ex) ./gene_sets/hp_LINCS.csv ; default=./gene_sets/hp_{dataset}.csv")
    ap.add_argument("--gene_mode", choices=["ranked", "random"], default="ranked")
    ap.add_argument("--random_rep", type=int, default=0, help="replicate index for random genes")
    ap.add_argument("--random_seed", type=int, default=0, help="seed base for random genes")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_dir = Path(args.gene_dir)

    # TOP_N sweep
    TOP_N_LIST = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 300, 400, "all"]

    # folds
    n_splits = 5 if args.dataset == "LINCS" else 10
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # hp map
    hp_path = Path(args.hp_csv) if args.hp_csv else (gene_dir / f"hp_{args.dataset}.csv")
    hp_map, hp_df = load_hp_map(hp_path)
    print(f"[HP] loaded {hp_path} rows={len(hp_df)} unique keys={len(hp_map)}")

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

    print(f"[INFO] kept MoAs={len(selected)} (multilabel removed={len(multi)})")
    print(f"[INFO] rows={len(df_all)} compounds={df_all['Compounds'].nunique()} classes={df_all['y'].nunique()}")

    y_all = df_all["y"].values
    groups = df_all["Compounds"].values
    X_cp = df_all[list(cp_features)].values.astype("float32")

    # ranked genes (for ranked mode)
    gene_file = gene_dir / f"genes_ranked_{args.dataset}.npy"
    genes_ranked = None
    if args.gene_mode == "ranked":
        genes_ranked = np.load(gene_file, allow_pickle=True)

    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)

    rows = []
    dumps = []
    avg = "weighted" if args.dataset == "LINCS" else "macro"

    # GE universe for random mode
    ge_universe = [g for g in l1k_features_full if g in df_all.columns]

    # helper: get GE genes for this TOP_N
    def pick_ge_genes(topn_tag, TOP_N):
        if TOP_N == "all":
            return [g for g in l1k_features_full if g in df_all.columns], "978"
        if args.gene_mode == "ranked":
            wanted = genes_ranked[:int(TOP_N)].tolist()
            available = [g for g in wanted if g in df_all.columns]
            return available, str(TOP_N)
        else:
            rng = np.random.default_rng(args.random_seed + 1000 * args.random_rep + int(TOP_N))
            available = rng.choice(ge_universe, size=int(TOP_N), replace=False).tolist()
            return available, str(TOP_N)

    # helper: get hp for modality/top_n
    def get_hp(topn_tag, modality):
        key = (str(topn_tag), str(modality))
        if key not in hp_map:
            return None
        return hp_map[key]

    for TOP_N in TOP_N_LIST:
        # get GE gene list + topn_tag
        available, topn_tag = pick_ge_genes(str(TOP_N), TOP_N)

        if len(available) == 0:
            print(f"[WARN] TOP_N={TOP_N}: no matched genes, skip")
            continue

        # hp availability check
        hp_cp = get_hp(topn_tag, "CP")
        hp_ge = get_hp(topn_tag, "GE")
        hp_ef = get_hp(topn_tag, "Early Fusion")
        if hp_cp is None or hp_ge is None or hp_ef is None:
            print(f"[WARN] TOP_N={topn_tag}: missing hp in {hp_path} "
                  f"(need CP/GE/Early Fusion). skip")
            continue

        X_ge = df_all[available].values.astype("float32")
        X_early = np.concatenate([X_cp, X_ge], axis=1).astype("float32")

        for fold, (tr, te) in enumerate(sgkf.split(X_early, y_all, groups=groups), start=1):
            y_tr, y_te = y_all[tr], y_all[te]

            # oversample each modality
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)
            Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
            Xtr_ef, ytr_ef = ros.fit_resample(X_early[tr], y_tr)

            n_classes = int(df_all["y"].nunique())

            # train by hp (no tuning)
            m_cp = train_torch_mlp(
                Xtr_cp, ytr_cp, n_classes, device,
                hidden=hp_cp["hidden"], dropout=hp_cp["dropout"], act=hp_cp["act"],
                lr=args.lr, weight_decay=hp_cp["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 11,
            )
            m_ge = train_torch_mlp(
                Xtr_ge, ytr_ge, n_classes, device,
                hidden=hp_ge["hidden"], dropout=hp_ge["dropout"], act=hp_ge["act"],
                lr=args.lr, weight_decay=hp_ge["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 12,
            )
            m_ef = train_torch_mlp(
                Xtr_ef, ytr_ef, n_classes, device,
                hidden=hp_ef["hidden"], dropout=hp_ef["dropout"], act=hp_ef["act"],
                lr=args.lr, weight_decay=hp_ef["wd"],
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + fold * 10 + 13,
            )

            print(f"[HP fold={fold} top_n={topn_tag}] "
                  f"CP={hp_cp} | GE={hp_ge} | EF={hp_ef}")

            # probs
            prob_cp = predict_proba(m_cp, X_cp[te], device)
            prob_ge = predict_proba(m_ge, X_ge[te], device)
            prob_ef = predict_proba(m_ef, X_early[te], device)

            pred_cp = prob_cp.argmax(1)
            pred_ge = prob_ge.argmax(1)
            pred_ef = prob_ef.argmax(1)

            # late fusion
            prob_lf = (prob_cp + prob_ge) / 2.0
            pred_lf = prob_lf.argmax(1)

            f1_cp = f1_score(y_te, pred_cp, average=avg)
            f1_ge = f1_score(y_te, pred_ge, average=avg)
            f1_ef = f1_score(y_te, pred_ef, average=avg)
            f1_lf = f1_score(y_te, pred_lf, average=avg)

            rows += [
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="CP", f1=f1_cp,
                     gene_mode=args.gene_mode, random_rep=args.random_rep),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="GE", f1=f1_ge,
                     gene_mode=args.gene_mode, random_rep=args.random_rep),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="Early Fusion", f1=f1_ef,
                     gene_mode=args.gene_mode, random_rep=args.random_rep),
                dict(dataset=args.dataset, top_n=topn_tag, fold=fold, modality="Late Fusion", f1=f1_lf,
                     gene_mode=args.gene_mode, random_rep=args.random_rep),
            ]

            dumps.append(pd.DataFrame({
                "dataset": args.dataset,
                "top_n": topn_tag,
                "fold": fold,
                "gene_mode": args.gene_mode,
                "random_rep": args.random_rep,
                "PERT": df_all.loc[te, "PERT"].values,
                "Compounds": df_all.loc[te, "Compounds"].values,
                "y_true": y_te,
                "CP": pred_cp,
                "GE": pred_ge,
                "Early Fusion": pred_ef,
                "Late Fusion": pred_lf,
            }))

        print(f"[DONE] TOP_N={topn_tag} (genes used={len(available)})")

    res = pd.DataFrame(rows)
    pred = pd.concat(dumps, ignore_index=True) if len(dumps) else pd.DataFrame()

    tag = f"{args.dataset}_torch_mlp_topn_{args.gene_mode}"
    if args.gene_mode == "random":
        tag += f"_rep{args.random_rep}_seed{args.random_seed}"

    res_path = out_dir / f"summary_{tag}.csv"
    xlsx_path = out_dir / f"pred_{tag}.xlsx"

    res.to_csv(res_path, index=False)
    if len(pred):
        pred.to_excel(xlsx_path, index=False)

    print("[SAVED]", res_path)
    if len(pred):
        print("[SAVED]", xlsx_path)


if __name__ == "__main__":
    main()
