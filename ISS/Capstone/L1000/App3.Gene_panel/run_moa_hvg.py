#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

import sys
sys.path.insert(0, "../utils/")
from readProfiles import read_paired_treatment_level_profiles

# ----------------
# helpers
# ----------------
pertColName = "PERT"
moa_col = "Metadata_MoA"

def load_ranked_genes(gene_dir: Path, dataset: str):
    gene_file = gene_dir / f"genes_ranked_{dataset}.npy"
    return np.load(gene_file, allow_pickle=True)

def harmonize_moa_and_compounds(merg: pd.DataFrame, dataset: str) -> pd.DataFrame:
    merg = merg.copy()
    if dataset == "LINCS":
        merg[moa_col] = merg["Metadata_moa"]
        merg.loc[merg["Metadata_moa"].isnull(), moa_col] = (
            merg.loc[merg["Metadata_moa"].isnull(), "moa"].astype(str).str.lower()
        )
        merg["Compounds"] = merg[pertColName].astype(str).str[0:13]
    elif dataset == "CDRP-bio":
        merg[moa_col] = merg["Metadata_moa"].astype(str).str.lower()
        merg["Compounds"] = merg[pertColName].astype(str).str[0:13]
    else:
        raise ValueError(dataset)
    return merg

def scale_modality(df: pd.DataFrame, feat_cols):
    out = df.copy()
    ss = preprocessing.StandardScaler()
    mm = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = out[feat_cols].values.astype("float64")
    X = ss.fit_transform(X)
    X = mm.fit_transform(X)
    out.loc[:, feat_cols] = X
    return out

def select_moas_by_compound_count(merg: pd.DataFrame, nSamplesMOA: int, seed: int):
    tmp = (
        merg.groupby(["Compounds"]).sample(1, random_state=seed)
            .groupby([moa_col]).size()
            .reset_index(name="size")
            .sort_values("size", ascending=False)
    )
    selected = tmp[tmp["size"] > nSamplesMOA][moa_col].tolist()
    multi = [m for m in selected if isinstance(m, str) and "|" in m]
    selected = [m for m in selected if m not in multi]

    df_all = merg[merg[moa_col].isin(selected)].reset_index(drop=True).copy()
    le = preprocessing.LabelEncoder()
    le.fit(selected)
    df_all["Metadata_moa_num"] = le.transform(df_all[moa_col].tolist())

    print(f"[INFO] MoA kept: {len(selected)} (removed multilabel: {len(multi)})")
    print(f"[INFO] rows={df_all.shape[0]} | compounds={df_all['Compounds'].nunique()} | classes={df_all['Metadata_moa_num'].nunique()}")
    return df_all, le

def make_mlp_grid(n_jobs: int, inner_cv: int, max_iter: int, seed: int):
    param_space = {
        "hidden_layer_sizes": [(100,), (200,), (400,)],
        "activation": ["tanh", "relu"],
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
    }
    base = MLPClassifier(random_state=seed, max_iter=max_iter)
    return GridSearchCV(base, param_space, n_jobs=n_jobs, cv=inner_cv)

def proba_align(prob_src, classes_src, classes_ref):
    src_map = {c: i for i, c in enumerate(classes_src)}
    out = np.zeros((prob_src.shape[0], len(classes_ref)), dtype=float)
    for j, c in enumerate(classes_ref):
        if c in src_map:
            out[:, j] = prob_src[:, src_map[c]]
    return out

# ----------------
# main
# ----------------
def run_dataset(
    dataset: str,
    procProf_dir: str,
    profileType: str,
    filter_perts: str,
    repCorrFilePath: str,
    per_plate_normalized_flag: int,
    gene_dir: Path,
    results_dir: Path,
    topn_list,
    nSamplesMOA: int,
    n_splits: int,
    seed: int,
    mlp_seed: int,
    n_jobs: int,
    inner_cv: int,
    max_iter: int,
):
    print(f"\n========== RUN {dataset} ==========")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    filter_repCorr_params = [filter_perts, repCorrFilePath]

    merg, cp_features, l1k_features_full = read_paired_treatment_level_profiles(
        procProf_dir, dataset, profileType, filter_repCorr_params, per_plate_normalized_flag
    )
    merg = harmonize_moa_and_compounds(merg, dataset)

    cp = merg[[pertColName, "Compounds", moa_col] + list(cp_features)].copy()
    ge = merg[[pertColName, "Compounds", moa_col] + list(l1k_features_full)].copy()

    cp_s = scale_modality(cp, list(cp_features))
    ge_s = scale_modality(ge, list(l1k_features_full))

    merged_scaled = pd.concat([cp_s, ge_s], axis=1)
    merged_scaled = merged_scaled.loc[:, ~merged_scaled.columns.duplicated()].copy()
    merged_scaled["Compounds"] = merged_scaled[pertColName].astype(str).str[0:13]

    df_all, _ = select_moas_by_compound_count(merged_scaled, nSamplesMOA=nSamplesMOA, seed=seed)
    y_all = df_all["Metadata_moa_num"].values
    groups = df_all["Compounds"].values

    X_cp = df_all[list(cp_features)].values.astype("float64")

    genes_ranked = load_ranked_genes(gene_dir, dataset)

    rows = []
    dumps = []

    ros = RandomOverSampler(sampling_strategy="not majority", random_state=seed)

    for TOP_N in topn_list:
        if TOP_N == "all":
            wanted = genes_ranked.tolist()
            topn_tag = "all"
        else:
            wanted = genes_ranked[:int(TOP_N)].tolist()
            topn_tag = str(TOP_N)

        available = [g for g in wanted if g in df_all.columns]
        if len(available) == 0:
            print(f"[WARN] TOP_N={TOP_N}: no genes matched columns. skip.")
            continue

        X_ge = df_all[available].values.astype("float64")
        X_early = np.concatenate([X_cp, X_ge], axis=1)

        print(f"[TOP_N={TOP_N}] genes_used={len(available)} | folds={n_splits}")

        for fold, (tr, te) in enumerate(sgkf.split(X_early, y_all, groups=groups), start=1):
            y_tr, y_te = y_all[tr], y_all[te]

            # CP
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y_tr)
            m_cp = make_mlp_grid(n_jobs=n_jobs, inner_cv=inner_cv, max_iter=max_iter, seed=mlp_seed)
            m_cp.fit(Xtr_cp, ytr_cp)
            pred_cp = m_cp.predict(X_cp[te])
            prob_cp = m_cp.predict_proba(X_cp[te])
            cls_cp = m_cp.best_estimator_.classes_

            # GE
            Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y_tr)
            m_ge = make_mlp_grid(n_jobs=n_jobs, inner_cv=inner_cv, max_iter=max_iter, seed=mlp_seed)
            m_ge.fit(Xtr_ge, ytr_ge)
            pred_ge = m_ge.predict(X_ge[te])
            prob_ge = m_ge.predict_proba(X_ge[te])
            cls_ge = m_ge.best_estimator_.classes_

            # Early Fusion
            Xtr_ef, ytr_ef = ros.fit_resample(X_early[tr], y_tr)
            m_ef = make_mlp_grid(n_jobs=n_jobs, inner_cv=inner_cv, max_iter=max_iter, seed=mlp_seed)
            m_ef.fit(Xtr_ef, ytr_ef)
            pred_ef = m_ef.predict(X_early[te])
            prob_ef = m_ef.predict_proba(X_early[te])
            cls_ef = m_ef.best_estimator_.classes_

            # Late Fusion
            prob_ge_aligned = proba_align(prob_ge, cls_ge, cls_cp)
            prob_lf = (prob_cp + prob_ge_aligned) / 2.0
            pred_lf = cls_cp[np.argmax(prob_lf, axis=1)]

            avg = "weighted" if dataset == "LINCS" else "macro"
            f1_cp = f1_score(y_te, pred_cp, average=avg)
            f1_ge = f1_score(y_te, pred_ge, average=avg)
            f1_ef = f1_score(y_te, pred_ef, average=avg)
            f1_lf = f1_score(y_te, pred_lf, average=avg)

            rows += [
                dict(dataset=dataset, top_n=topn_tag, fold=fold, modality="CP", f1=f1_cp),
                dict(dataset=dataset, top_n=topn_tag, fold=fold, modality="GE", f1=f1_ge),
                dict(dataset=dataset, top_n=topn_tag, fold=fold, modality="Early Fusion", f1=f1_ef),
                dict(dataset=dataset, top_n=topn_tag, fold=fold, modality="Late Fusion", f1=f1_lf),
            ]

            dumps.append(pd.DataFrame({
                "dataset": dataset, "top_n": topn_tag, "fold": fold,
                "PERT": df_all.loc[te, pertColName].values,
                "Compounds": df_all.loc[te, "Compounds"].values,
                "Metadata_moa_num": y_te,
                "CP": pred_cp, "GE": pred_ge, "Early Fusion": pred_ef, "Late Fusion": pred_lf,
            }))

    res = pd.DataFrame(rows)
    pred_df = pd.concat(dumps, ignore_index=True) if dumps else pd.DataFrame()

    results_dir.mkdir(parents=True, exist_ok=True)
    res_csv = results_dir / f"summary_{dataset}_MLP_TOPN.csv"
    pred_xlsx = results_dir / f"pred_{dataset}_MLP_TOPN.xlsx"
    res.to_csv(res_csv, index=False)
    if len(pred_df):
        pred_df.to_excel(pred_xlsx, index=False)

    print(f"[SAVED] {res_csv}")
    if len(pred_df):
        print(f"[SAVED] {pred_xlsx}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["LINCS", "CDRP-bio"])
    ap.add_argument("--procProf_dir", default="../")
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="../results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--per_plate_normalized_flag", type=int, default=1)
    ap.add_argument("--gene_dir", default="./gene_sets")
    ap.add_argument("--results_dir", default="../results/Jiwoo/Genepanel")
    ap.add_argument("--nSamplesMOA", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mlp_seed", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--inner_cv", type=int, default=3)
    ap.add_argument("--max_iter", type=int, default=600)
    ap.add_argument("--topn", nargs="+", default=["5","10","15","20","25","30","50","75","100","150","200","300","400","all"])
    args = ap.parse_args()

    gene_dir = Path(args.gene_dir)
    results_dir = Path(args.results_dir)

    topn_list = []
    for x in args.topn:
        if x == "all":
            topn_list.append("all")
        else:
            topn_list.append(int(x))

    for ds in args.datasets:
        n_splits = 5 if ds == "LINCS" else 10
        run_dataset(
            dataset=ds,
            procProf_dir=args.procProf_dir,
            profileType=args.profileType,
            filter_perts=args.filter_perts,
            repCorrFilePath=args.repCorrFilePath,
            per_plate_normalized_flag=args.per_plate_normalized_flag,
            gene_dir=gene_dir,
            results_dir=results_dir,
            topn_list=topn_list,
            nSamplesMOA=args.nSamplesMOA,
            n_splits=n_splits,
            seed=args.seed,
            mlp_seed=args.mlp_seed,
            n_jobs=args.n_jobs,
            inner_cv=args.inner_cv,
            max_iter=args.max_iter,
        )

if __name__ == "__main__":
    main()
