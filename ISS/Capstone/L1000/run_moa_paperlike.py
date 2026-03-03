#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper-like MoA prediction runner (CDRP-bio / LINCS) from Rosetta CP+GE.
- Uses sklearn MLPClassifier / LogisticRegression + GridSearchCV
- Oversampling with RandomOverSampler (not majority)
- Group split by Compounds (PERT[:13])
- StratifiedGroupKFold (CDRP-bio: 10 folds, LINCS: 5 folds by default)

Outputs:
1) results/MoAprediction/pred_moa_{DATASET}.xlsx  (one sheet per model)
2) results/MoAprediction/f1_{DATASET}.csv         (fold-wise F1 summary)

NOTE:
This intentionally mirrors the notebook logic (including fitting scalers on the full set),
to reproduce the paper's pipeline as closely as possible.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

import sys
sys.path.insert(0, "./utils")
from readProfiles import read_paired_treatment_level_profiles  # noqa

# -------------------------
# utils: save xlsx sheet
# -------------------------
def save_sheet_xlsx(xlsx_path: Path, df: pd.DataFrame, sheet_name: str):
    """
    Save df to xlsx sheet. If file exists, append/replace sheet.
    """
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    if xlsx_path.exists():
        # append / replace
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)


def scale_like_notebook(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Notebook behavior:
    StandardScaler -> MinMaxScaler(0,1) fitted on the whole set (not fold-wise).
    """
    out = df.copy()
    ss = preprocessing.StandardScaler()
    mm = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = out[feat_cols].values.astype("float64")
    X = ss.fit_transform(X)
    X = mm.fit_transform(X)
    out.loc[:, feat_cols] = X
    return out


def prepare_dataset(procProf_dir: str, dataset: str, profileType: str, filter_perts: str, repCorrFilePath: str, nRep: int):
    """
    Load merged treatment-level profiles using readProfiles.py and apply notebook harmonization.
    Returns:
      filteredMOAs (merged_scaled filtered to selected MoAs, with y numeric)
      cp_scaled subset, ge_scaled subset (each aligned to filteredMOAs rows by concat)
      cp_features, ge_features
      label encoder (for inverse)
      groups (Compounds)
    """
    pertColName = "PERT"
    moa_col = "Metadata_MoA"

    filter_repCorr_params = [filter_perts, repCorrFilePath]
    merg, cp_features, l1k_features = read_paired_treatment_level_profiles(
        procProf_dir, dataset, profileType, filter_repCorr_params, nRep
    )

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
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Build CP/GE tables (treatment-level)
    cp = merg[[pertColName, "Compounds", moa_col] + list(cp_features)].copy()
    ge = merg[[pertColName, "Compounds", moa_col] + list(l1k_features)].copy()

    # Scale each modality like notebook (fit on full dataset)
    cp_scaled = scale_like_notebook(cp, list(cp_features))
    ge_scaled = scale_like_notebook(ge, list(l1k_features))

    # concat for early fusion
    merged_scaled = pd.concat([cp_scaled, ge_scaled], axis=1)
    merged_scaled = merged_scaled.loc[:, ~merged_scaled.columns.duplicated()].copy()
    merged_scaled["Compounds"] = merged_scaled[pertColName].astype(str).str[0:13]

    return merged_scaled, cp_scaled, ge_scaled, list(cp_features), list(l1k_features), moa_col, pertColName


def select_moas_like_notebook(merged_scaled: pd.DataFrame, moa_col: str, nSamplesMOA: int):
    """
    Notebook behavior: count unique compounds per MoA (by sampling one row per compound),
    keep MoAs with > nSamplesMOA compounds, remove multilabel MoAs containing '|'.
    """
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


def fit_predict_one_fold(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    model_name: str,
    param_space: dict,
    avg: str,
    random_state: int,
    cv_inner: int,
    n_jobs: int,
    fixed_params: dict | None = None,   # ✅ 추가
):
    """
    Train GridSearchCV for given model on oversampled training, return (pred, proba, best_params).
    If fixed_params is provided, skip gridsearch and fit a single model with those params.
    """
    if model_name == "mlp":
        base = MLPClassifier(
            random_state=random_state,
            max_iter=600,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            tol=1e-4,
        )
    elif model_name == "lr":
        base = LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=2000,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(model_name)

    if fixed_params is not None:
        est = base.set_params(**fixed_params)
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        proba = est.predict_proba(Xte)
        return pred, proba, fixed_params

    gs = GridSearchCV(base, param_space, n_jobs=n_jobs, cv=cv_inner)
    gs.fit(Xtr, ytr)
    pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)
    return pred, proba, gs.best_params_




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["CDRP-bio", "LINCS"], required=True)

    # keep defaults inside (as you wanted); still override-able
    ap.add_argument("--procProf_dir", default="./")
    ap.add_argument("--results_dir", default="./results/Jiwoo")
    ap.add_argument("--profileType", default="normalized_variable_selected")
    ap.add_argument("--filter_perts", default="highRepUnion")
    ap.add_argument("--repCorrFilePath", default="./results/RepCor/RepCorrDF.xlsx")
    ap.add_argument("--nRep", type=int, default=1)

    ap.add_argument("--nSamplesMOA", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)

    # folds (paper-like: CDRP-bio 10, LINCS 5)
    ap.add_argument("--n_folds_cdrp", type=int, default=10)
    ap.add_argument("--n_folds_lincs", type=int, default=5)

    # inner CV for gridsearch
    ap.add_argument("--cv_inner", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=4)

    # options
    ap.add_argument("--run_models", default="mlp,lr")  # "mlp", "lr", or "mlp,lr"
    ap.add_argument("--include_rgcca", action="store_true", help="If you later want RGCCA factors; default off.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    results_dir = Path(args.results_dir)
    out_xlsx = results_dir / "paper" / f"pred_moa_{args.dataset}.xlsx"
    out_f1csv = results_dir / "paper" / f"f1_{args.dataset}.csv"

    # -------------------------
    # Load + preprocess
    # -------------------------
    merged_scaled, cp_scaled, ge_scaled, cp_features, ge_features, moa_col, pertColName = prepare_dataset(
        procProf_dir=args.procProf_dir,
        dataset=args.dataset,
        profileType=args.profileType,
        filter_perts=args.filter_perts,
        repCorrFilePath=args.repCorrFilePath,
        nRep=args.nRep,
    )

    selected_moas, multi_removed, moa_sizes = select_moas_like_notebook(merged_scaled, moa_col, args.nSamplesMOA)

    filteredMOAs = merged_scaled[merged_scaled[moa_col].isin(selected_moas)].reset_index(drop=True).copy()

    le = preprocessing.LabelEncoder()
    le.fit(selected_moas)
    filteredMOAs["y"] = le.transform(filteredMOAs[moa_col].tolist())

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] kept MoAs={len(selected_moas)} (multilabel removed={len(multi_removed)})")
    print(f"[INFO] rows={len(filteredMOAs)} compounds={filteredMOAs['Compounds'].nunique()} classes={filteredMOAs['y'].nunique()}")

    # modalities (aligned by concat behavior)
    X_cp = filteredMOAs[cp_features].values.astype("float64")
    X_ge = filteredMOAs[ge_features].values.astype("float64")
    X_ef = filteredMOAs[cp_features + ge_features].values.astype("float64")

    y = filteredMOAs["y"].values.astype(int)
    groups = preprocessing.LabelEncoder().fit_transform(filteredMOAs["Compounds"].values)

    # paper-like averaging: CDRP-bio macro, LINCS weighted
    avg = "macro" if args.dataset == "CDRP-bio" else "weighted"

    # folds
    n_splits = args.n_folds_cdrp if args.dataset == "CDRP-bio" else args.n_folds_lincs
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    # oversampler
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=5)

    # param spaces (paper-ish)
    # NOTE: notebook had slightly different grids in different blocks; this is the "main" one you pasted
    param_space_mlp = {
        "hidden_layer_sizes": [(100,), (200,), (400,)],
        "activation": ["tanh", "relu"],
        # notebook uses alpha [0.0001, 0.05] and sometimes learning_rate too
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
    }
    param_space_lr = {"C": [1, 10, 1000]} if args.dataset == "CDRP-bio" else {"C": [500, 1000, 1500]}

    run_models = [m.strip() for m in args.run_models.split(",") if m.strip()]

    # -------------------------
    # Run CV
    # -------------------------
    f1_rows = []

    for model_name in run_models:
        best_cache = {}  # (model_name, modality) -> best_params

        # store per-sample predictions like notebook-style table
        pred_df = pd.DataFrame({
            "PERT": filteredMOAs[pertColName].values,
            "Compounds": filteredMOAs["Compounds"].values,
            "y_true": y,
            "Fold": -1,
            "CP": -1,
            "GE": -1,
            "Early Fusion": -1,
            "Late Fusion": -1,
        })

        for fold, (tr, te) in enumerate(sgkf.split(X_ef, y, groups=groups), start=1):
            pred_df.loc[te, "Fold"] = fold

            # oversample independently per modality (as notebook)
            Xtr_cp, ytr_cp = ros.fit_resample(X_cp[tr], y[tr])
            Xtr_ge, ytr_ge = ros.fit_resample(X_ge[tr], y[tr])
            Xtr_ef, ytr_ef = ros.fit_resample(X_ef[tr], y[tr])

            # choose param grid
            if model_name == "mlp":
                ps = param_space_mlp
            else:
                ps = param_space_lr
            
            best_cache = {}  # (model, modality) -> best_params

            # inside fold loop, before calling fit_predict_one_fold
            # 예: CP
            fixed = best_cache.get((model_name, "CP"), None)
            pred_cp, prob_cp, best_cp = fit_predict_one_fold(
                Xtr_cp, ytr_cp,
                X_cp[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
                fixed_params=fixed,
            )
            if fixed is None:
                best_cache[(model_name, "CP")] = best_cp


            # GE
            fixed = best_cache.get((model_name, "GE"), None)
            pred_ge, prob_ge, best_ge = fit_predict_one_fold(
                Xtr_ge, ytr_ge,
                X_ge[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
                fixed_params=fixed,
            )
            if fixed is None:
                best_cache[(model_name, "GE")] = best_ge


            # EF
            fixed = best_cache.get((model_name, "EF"), None)
            pred_ef, prob_ef, best_ef = fit_predict_one_fold(
                Xtr_ef, ytr_ef,
                X_ef[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
                fixed_params=fixed,
            )
            if fixed is None:
                best_cache[(model_name, "EF")] = best_ef



            # fit/predict each modality
            pred_cp, prob_cp, best_cp = fit_predict_one_fold(
                Xtr_cp, ytr_cp, X_cp[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
            )
            pred_ge, prob_ge, best_ge = fit_predict_one_fold(
                Xtr_ge, ytr_ge, X_ge[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
            )
            pred_ef, prob_ef, best_ef = fit_predict_one_fold(
                Xtr_ef, ytr_ef, X_ef[te], y[te],
                model_name=model_name,
                param_space=ps,
                avg=avg,
                random_state=5,
                cv_inner=args.cv_inner,
                n_jobs=args.n_jobs,
            )

            # late fusion = avg of CP/GE probs
            prob_lf = (prob_cp + prob_ge) / 2.0
            pred_lf = np.argmax(prob_lf, axis=1)

            # write predictions
            pred_df.loc[te, "CP"] = pred_cp
            pred_df.loc[te, "GE"] = pred_ge
            pred_df.loc[te, "Early Fusion"] = pred_ef
            pred_df.loc[te, "Late Fusion"] = pred_lf

            # fold f1
            f1_cp = f1_score(y[te], pred_cp, average=avg)
            f1_ge = f1_score(y[te], pred_ge, average=avg)
            f1_ef = f1_score(y[te], pred_ef, average=avg)
            f1_lf = f1_score(y[te], pred_lf, average=avg)

            f1_rows += [
                {"dataset": args.dataset, "model": model_name, "fold": fold, "modality": "CP", "f1": f1_cp},
                {"dataset": args.dataset, "model": model_name, "fold": fold, "modality": "GE", "f1": f1_ge},
                {"dataset": args.dataset, "model": model_name, "fold": fold, "modality": "Early Fusion", "f1": f1_ef},
                {"dataset": args.dataset, "model": model_name, "fold": fold, "modality": "Late Fusion", "f1": f1_lf},
            ]

            # keep stdout light (no giant hp logs)
            print(f"[{args.dataset}][{model_name}] fold {fold}/{n_splits} "
                  f"F1: CP={f1_cp:.3f} GE={f1_ge:.3f} EF={f1_ef:.3f} LF={f1_lf:.3f}")

        # save sheet
        sheet_name = f"{model_name}-sgkf{n_splits}"
        save_sheet_xlsx(out_xlsx, pred_df, sheet_name)
        print(f"[SAVED] {out_xlsx} (sheet={sheet_name})")

    # save f1 summary
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv(out_f1csv, index=False)
    print(f"[SAVED] {out_f1csv}")

    # quick aggregate print
    agg = f1_df.groupby(["model", "modality"])["f1"].mean().reset_index()
    print("\n[MEAN F1]")
    print(agg.sort_values(["model", "modality"]).to_string(index=False))


if __name__ == "__main__":
    main()
