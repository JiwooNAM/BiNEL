#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("DEBUG: starting script")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

import sys
sys.path.insert(0, "./utils")
from readProfiles import read_paired_treatment_level_profiles



# ============================================================
# Utility functions
# ============================================================

def compound_table(proba, df_meta):
    df = df_meta.copy()
    n_classes = proba.shape[1]
    for c in range(n_classes):
        df[f"p_{c}"] = proba[:, c]

    agg = df.groupby("Compounds", as_index=False).agg(
        y_true=("y_true", "first"),
        **{f"p_{c}": (f"p_{c}", "max") for c in range(n_classes)}
    )
    return agg


def enrichment_curve_macro(ct, ps=(0.5,1,2,5,10,20), min_pos=5):
    C = len([c for c in ct.columns if c.startswith("p_")])
    N = ct.shape[0]
    out = []

    for p in ps:
        m = int(np.ceil(N * (p/100)))
        enr_list = []

        for c in range(C):
            y = (ct["y_true"].values == c).astype(int)
            npos = y.sum()
            if npos < min_pos:
                continue

            base = npos / N
            score = ct[f"p_{c}"].values
            top_idx = np.argsort(-score)[:m]
            prec = y[top_idx].mean()
            enr = prec / base if base > 0 else np.nan
            enr_list.append(enr)

        out.append({
            "top_pct": p,
            "macro_enrichment": float(np.nanmean(enr_list)) if enr_list else np.nan,
            "median_enrichment": float(np.nanmedian(enr_list)) if enr_list else np.nan,
            "n_moas_used": len(enr_list)
        })

    return pd.DataFrame(out)


# ============================================================
# Main
# ============================================================

def main(args):

    # --------------------------
    # Load profiles
    # --------------------------
    filter_repCorr_params = [args.filter_perts, args.repCorrFilePath]

    merg, cp_features, l1k_features = read_paired_treatment_level_profiles(
        args.procProf_dir,
        args.dataset,
        args.profileType,
        filter_repCorr_params,
        1
    )
    print("DEBUG: merg shape =", merg.shape)
    moa_col = "Metadata_MoA"

    if args.dataset == "LINCS":
        merg[moa_col] = merg["Metadata_moa"]
        merg.loc[merg["Metadata_moa"].isnull(), moa_col] = \
            merg.loc[merg["Metadata_moa"].isnull(), "moa"].astype(str).str.lower()
        merg["Compounds"] = merg["PERT"].astype(str).str[0:13]
    else:
        merg[moa_col] = merg["Metadata_moa"].astype(str).str.lower()
        merg["Compounds"] = merg["PERT"].astype(str).str[0:13]

    merg = merg.dropna(subset=[moa_col]).copy()

    # --------------------------
    # Scaling
    # --------------------------
    scaler_cp = StandardScaler()
    scaler_ge = StandardScaler()

    cp = merg[["PERT","Compounds",moa_col] + list(cp_features)].copy()
    ge = merg[["PERT","Compounds",moa_col] + list(l1k_features)].copy()

    cp[cp_features] = MinMaxScaler().fit_transform(
        scaler_cp.fit_transform(cp[cp_features])
    )
    ge[l1k_features] = MinMaxScaler().fit_transform(
        scaler_ge.fit_transform(ge[l1k_features])
    )

    merged = pd.concat([cp, ge], axis=1)
    merged = merged.loc[:,~merged.columns.duplicated()].copy()

    # --------------------------
    # Filter MoAs
    # --------------------------
    nSamplesforEachMOAclass = merg.groupby(["Compounds"]).sample(1)\
        .groupby([moa_col]).size().reset_index(name="size")

    selected = nSamplesforEachMOAclass[
        nSamplesforEachMOAclass["size"] > args.nSamplesMOA
    ][moa_col].tolist()

    selected = [m for m in selected if "|" not in m]

    df = merged[merged[moa_col].isin(selected)].reset_index(drop=True).copy()

    le = LabelEncoder()
    le.fit(selected)
    df["y"] = le.transform(df[moa_col].values)

    print("Samples:", df.shape[0])
    print("Compounds:", df["Compounds"].nunique())
    print("MoAs:", len(selected))

    # --------------------------
    # Prepare arrays
    # --------------------------
    X_cp = df[cp_features].values
    X_ge = df[l1k_features].values
    X_ef = np.concatenate([X_cp, X_ge], axis=1)
    y = df["y"].values
    groups = df["Compounds"].values

    n_classes = len(selected)
    N = df.shape[0]

    proba_CP = np.zeros((N, n_classes))
    proba_GE = np.zeros((N, n_classes))
    proba_EF = np.zeros((N, n_classes))
    proba_LF = np.zeros((N, n_classes))

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits,
                                shuffle=True,
                                random_state=42)

    ros = RandomOverSampler(sampling_strategy='not majority', random_state=5)

    # --------------------------
    # CV Loop
    # --------------------------
    for fold, (tr, te) in enumerate(sgkf.split(X_cp, y, groups), start=1):

        print("Fold", fold)

        for X_mod, store in zip(
            [X_cp, X_ge, X_ef],
            [proba_CP, proba_GE, proba_EF]
        ):
            Xtr, ytr = ros.fit_resample(X_mod[tr], y[tr])

            if args.model == "mlp":
                model = MLPClassifier(hidden_layer_sizes=(200,),
                                      max_iter=500)
            else:
                model = LogisticRegression(max_iter=1000,
                                           multi_class="multinomial")

            model.fit(Xtr, ytr)
            p = model.predict_proba(X_mod[te])
            store[te] = p

        proba_LF[te] = (proba_CP[te] + proba_GE[te]) / 2.0

    # --------------------------
    # Compound-level aggregation
    # --------------------------
    df_meta = df[["Compounds","y"]].copy()
    df_meta["y_true"] = df_meta["y"]

    ct_CP = compound_table(proba_CP, df_meta)
    ct_GE = compound_table(proba_GE, df_meta)
    ct_EF = compound_table(proba_EF, df_meta)
    ct_LF = compound_table(proba_LF, df_meta)

    ps = [0.5,1,2,5,10,20]

    curve_CP = enrichment_curve_macro(ct_CP, ps)
    curve_GE = enrichment_curve_macro(ct_GE, ps)
    curve_EF = enrichment_curve_macro(ct_EF, ps)
    curve_LF = enrichment_curve_macro(ct_LF, ps)

    # --------------------------
    # Save results
    # --------------------------
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    curve_CP.to_csv(outdir/"curve_CP.csv", index=False)
    curve_GE.to_csv(outdir/"curve_GE.csv", index=False)
    curve_EF.to_csv(outdir/"curve_EF.csv", index=False)
    curve_LF.to_csv(outdir/"curve_LF.csv", index=False)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(curve_CP.top_pct, curve_CP.macro_enrichment, marker="o", label="CP")
    plt.plot(curve_GE.top_pct, curve_GE.macro_enrichment, marker="o", label="GE")
    plt.plot(curve_EF.top_pct, curve_EF.macro_enrichment, marker="o", label="Early Fusion")
    plt.plot(curve_LF.top_pct, curve_LF.macro_enrichment, marker="o", label="Late Fusion")
    plt.xscale("log")
    plt.xlabel("Top % compounds")
    plt.ylabel("Macro Enrichment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"enrichment_curve.png", dpi=300)
    print("Saved results to:", outdir)


# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="LINCS")
    parser.add_argument("--procProf_dir", required=True)
    parser.add_argument("--profileType", default="normalized_variable_selected")
    parser.add_argument("--filter_perts", default="highRepUnion")
    parser.add_argument("--repCorrFilePath", required=True)
    parser.add_argument("--out_dir", default="./enrichment_results")
    parser.add_argument("--model", choices=["mlp","lr"], default="mlp")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--nSamplesMOA", type=int, default=4)

    args = parser.parse_args()
    main(args)
