#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances

def to_dense(X):
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)

def get_matrix(adata: ad.AnnData, use_obsm: str | None):
    if use_obsm is None or str(use_obsm).lower() in {"none", "null", ""}:
        return to_dense(adata.X)
    return to_dense(adata.obsm[use_obsm])

def norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def norm_rep(x):
    s = str(x).strip().lower()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s

def make_gid(obs: pd.DataFrame, keys: list[str]) -> pd.Series:
    tmp = obs[keys].copy()
    for k in keys:
        if k.lower() == "replicate":
            tmp[k] = tmp[k].apply(norm_rep)
        else:
            tmp[k] = norm_str_series(tmp[k])
    return tmp.astype(str).agg("|".join, axis=1)

def group_indices(gid: pd.Series):
    d = {}
    arr = gid.to_numpy()
    for i, g in enumerate(arr):
        d.setdefault(g, []).append(i)
    return {g: np.asarray(ix, dtype=int) for g, ix in d.items()}

def cosine_dist_to_centroid(X, c):
    # returns 1 - cosine_similarity
    return pairwise_distances(X, c[None, :], metric="cosine").reshape(-1)

def quantile_rank(x: np.ndarray) -> np.ndarray:
    # map to [0,1] by rank within vector
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    return ranks / (x.size - 1)

def build_dmso_centroids(
    X: np.ndarray,
    obs: pd.DataFrame,
    treat_key: str,
    control_name: str,
    ctx_keys: list[str],   # e.g., ["batch","Replicate"]
):
    # centroids per context (batch|rep). fallback global DMSO centroid.
    is_dmso = norm_str_series(obs[treat_key]) == control_name.lower()
    if is_dmso.sum() == 0:
        raise RuntimeError(f"No control '{control_name}' found in {treat_key}.")
    dmso_obs = obs.loc[is_dmso, ctx_keys].copy()
    # normalize ctx keys
    for k in ctx_keys:
        if k.lower() == "replicate":
            dmso_obs[k] = dmso_obs[k].apply(norm_rep)
        else:
            dmso_obs[k] = norm_str_series(dmso_obs[k])
    ctx = dmso_obs.astype(str).agg("|".join, axis=1)
    idx = np.where(is_dmso.to_numpy())[0]

    centroids = {}
    for cval, ix in group_indices(ctx).items():
        centroids[cval] = X[idx[ix]].mean(axis=0)

    global_c = X[idx].mean(axis=0)
    return centroids, global_c

def ctx_id(obs: pd.DataFrame, ctx_keys: list[str]) -> pd.Series:
    tmp = obs[ctx_keys].copy()
    for k in ctx_keys:
        if k.lower() == "replicate":
            tmp[k] = tmp[k].apply(norm_rep)
        else:
            tmp[k] = norm_str_series(tmp[k])
    return tmp.astype(str).agg("|".join, axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_h5ad", required=True)
    ap.add_argument("--cp_h5ad", required=True)
    ap.add_argument("--rna_embed_key", default="scvi_n200")
    ap.add_argument("--cp_use_obsm", default=None)
    ap.add_argument("--treat_key", default="Treatment")
    ap.add_argument("--batch_key", default="batch")
    ap.add_argument("--rep_key", default="Replicate")
    ap.add_argument("--control_name", default="DMSO")
    ap.add_argument("--max_cells_per_group", type=int, default=1500)
    ap.add_argument("--out_h5ad", required=True)
    ap.add_argument("--out_h5ad_obsm", default=None)  # optional
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[INFO] Loading RNA:", args.rna_h5ad)
    rna = ad.read_h5ad(args.rna_h5ad)
    print("[INFO] Loading CP:", args.cp_h5ad)
    cp  = ad.read_h5ad(args.cp_h5ad)

    assert args.rna_embed_key in rna.obsm_keys(), f"RNA obsm missing {args.rna_embed_key}"
    if args.cp_use_obsm not in (None, "None", "none", ""):
        assert args.cp_use_obsm in cp.obsm_keys(), f"CP obsm missing {args.cp_use_obsm}"

    # matrices
    Xr = get_matrix(rna, args.rna_embed_key).astype(np.float32)
    Xc = get_matrix(cp, None if str(args.cp_use_obsm).lower() in {"none","null",""} else args.cp_use_obsm).astype(np.float32)

    # keys exist?
    for k in [args.treat_key, args.batch_key, args.rep_key]:
        if k not in rna.obs.columns or k not in cp.obs.columns:
            raise RuntimeError(f"Missing key '{k}' in either RNA or CP obs.")

    # gid = Treatment|batch|Replicate  (source 제외)
    anchor_keys = [args.treat_key, args.batch_key, args.rep_key]
    rna.obs["_gid"] = make_gid(rna.obs, anchor_keys)
    cp.obs["_gid"]  = make_gid(cp.obs,  anchor_keys)

    # context id for DMSO reference: batch|Replicate (treatment 제외)
    ctx_keys = [args.batch_key, args.rep_key]
    rna.obs["_ctx"] = ctx_id(rna.obs, ctx_keys)
    cp.obs["_ctx"]  = ctx_id(cp.obs,  ctx_keys)

    common_gids = sorted(set(rna.obs["_gid"]) & set(cp.obs["_gid"]))
    print("[INFO] Common gids:", len(common_gids))
    if len(common_gids) == 0:
        raise RuntimeError("No common gids. Check normalization / keys.")

    # build DMSO centroids per context for each modality
    r_centroids, r_global = build_dmso_centroids(Xr, rna.obs, args.treat_key, args.control_name, ctx_keys)
    c_centroids, c_global = build_dmso_centroids(Xc, cp.obs,  args.treat_key, args.control_name, ctx_keys)

    # groups
    r_groups = group_indices(rna.obs["_gid"])
    c_groups = group_indices(cp.obs["_gid"])

    pairs = []
    for gi, g in enumerate(common_gids):
        ir = r_groups[g]
        ic = c_groups[g]
        if ir.size == 0 or ic.size == 0:
            continue

        # cap for runtime
        kr = min(ir.size, args.max_cells_per_group)
        kc = min(ic.size, args.max_cells_per_group)
        if ir.size > kr:
            ir = rng.choice(ir, size=kr, replace=False)
        if ic.size > kc:
            ic = rng.choice(ic, size=kc, replace=False)

        # choose ctx centroid (batch|rep) for this gid
        # ctx is constant within gid by construction; take first
        ctx_r = rna.obs["_ctx"].to_numpy()[ir[0]]
        ctx_c = cp.obs["_ctx"].to_numpy()[ic[0]]

        cr = r_centroids.get(ctx_r, r_global)
        cc = c_centroids.get(ctx_c, c_global)

        dr = cosine_dist_to_centroid(Xr[ir], cr)
        dc = cosine_dist_to_centroid(Xc[ic], cc)

        qr = quantile_rank(dr)
        qc = quantile_rank(dc)

        # rank matching: sort by quantile and pair in order
        ord_r = np.argsort(qr)
        ord_c = np.argsort(qc)
        k = min(ord_r.size, ord_c.size)

        for a, b in zip(ic[ord_c[:k]].tolist(), ir[ord_r[:k]].tolist()):
            pairs.append((a, b))

        if (gi+1) % 50 == 0:
            print(f"  processed gids: {gi+1}/{len(common_gids)}")

    print("[INFO] dmso-rank matched pairs:", len(pairs))

    # build paired AnnData: concat X
    ic = np.array([p[0] for p in pairs], dtype=int)
    ir = np.array([p[1] for p in pairs], dtype=int)

    Xpair = np.concatenate([Xc[ic], Xr[ir]], axis=1)
    obs = pd.DataFrame({
        "cp_index": ic,
        "rna_index": ir,
        "Treatment": cp.obs[args.treat_key].astype(str).to_numpy()[ic],
        "batch": cp.obs[args.batch_key].astype(str).to_numpy()[ic],
        "Replicate": cp.obs[args.rep_key].astype(str).to_numpy()[ic],
    })
    var = pd.DataFrame(index=[f"cp_{i}" for i in range(Xc.shape[1])] + [f"rna_{j}" for j in range(Xr.shape[1])])
    out = ad.AnnData(Xpair, obs=obs, var=var)

    # NOTE: uns에 None 넣지 말기 (backed read 안전)
    out.uns["pairing"] = {
        "name": "dmso_rank_matched",
        "anchor_keys": [str(k) for k in anchor_keys],
        "context_keys": [str(k) for k in ctx_keys],
        "control_name": str(args.control_name),
        "rna_embed_key": str(args.rna_embed_key),
        "cp_use_obsm": str(args.cp_use_obsm) if args.cp_use_obsm not in (None, "None", "none", "") else "X",
    }

    out.write_h5ad(args.out_h5ad)
    print("[OK] wrote", args.out_h5ad)

    # optional: obsm-separated output
    if args.out_h5ad_obsm:
        out2 = ad.AnnData(np.zeros((ic.size, 1), dtype=np.float32),
                          obs=obs.copy(),
                          var=pd.DataFrame(index=["dummy"]))
        out2.obsm["cp"] = Xc[ic].astype(np.float32)
        out2.obsm["rna"] = Xr[ir].astype(np.float32)
        out2.uns["pairing"] = out.uns["pairing"].copy()
        out2.write_h5ad(args.out_h5ad_obsm)
        print("[OK] wrote", args.out_h5ad_obsm)

if __name__ == "__main__":
    main()
