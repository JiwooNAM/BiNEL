from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

# =========================
# USER SETTINGS
# =========================
date="2026-02-11"
time = "16-38-22"
epochs = 25
steps = 8112
RUN_DIR = Path(f"logs/train/runs/{date}_{time}").resolve()
CFG_DIR = (RUN_DIR / ".hydra").resolve()
CKPT    = (RUN_DIR / "checkpoints" / f"epoch={epochs}-step={steps}.ckpt").resolve()
CFG_YAML = (RUN_DIR / ".hydra" / "config.yaml").resolve()

# =========================
# Resolvers (offline)
# =========================
def _ensure_resolvers():
    if not OmegaConf.has_resolver("multiply"):
        def multiply(*xs):
            out = 1.0
            for x in xs:
                out *= float(x)
            return int(out) if abs(out - int(out)) < 1e-9 else out
        OmegaConf.register_new_resolver("multiply", multiply)

    # ${hydra:...} 같은게 cfg에 박혀있어서, HydraConfig 없이도 최소한 돌아가게 stub
    if not OmegaConf.has_resolver("hydra"):
        hydra_stub = {
            "runtime": {"output_dir": str(RUN_DIR), "cwd": os.getcwd()},
            "run": {"dir": str(RUN_DIR)},
            "sweep": {"dir": str(RUN_DIR)},
            "job": {"name": "offline_eval"},
        }
        def hydra_resolver(path: str):
            cur = hydra_stub
            for part in str(path).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    raise KeyError(f"hydra resolver missing key: {path} (stuck at {part})")
            return cur
        OmegaConf.register_new_resolver("hydra", hydra_resolver)

_ensure_resolvers()

# =========================
# category list (ground truth label mapping)
# =========================
def load_categories_from_cfg(cfg):
    """
    cfg.data.transform.example_transform.condition.encode.path_to_categories 를 instantiate해서
    실제 categories 파일 경로를 얻고, 거기서 class_names 로드.
    """
    enc = cfg.data.transform.example_transform.condition.encode
    p = hydra.utils.instantiate(enc.path_to_categories)  # -> actual path
    p = Path(p)

    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p, allow_pickle=True)
        return [str(x) for x in arr.tolist()], p
    if suf == ".json":
        import json
        return [str(x) for x in json.loads(p.read_text())], p
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) == 1 and ("," in lines[0]):
        lines = [x.strip() for x in lines[0].split(",") if x.strip()]
    return lines, p

# =========================
# Pull treatment_select lists from cfg (train/val/test, rnaseq/imaging)
# =========================
def get_ts(cfg, split, modality):
    """
    split in {"train","val","test"}
    modality in {"rnaseq","imaging"}
    returns list[str] or None
    """
    try:
        ts = cfg.data.data_iter_factory[split].factory[modality].samples_factory.file_path.treatment_select
    except Exception:
        return None
    if ts is None:
        return None
    return [str(x) for x in ts]

# =========================
# Batch parsing (confirmed keys)
# =========================
def pick_xy_multimodal(batch):
    # batch keys: ['condition', 'rnaseq', 'imaging']
    return batch["imaging"], batch["rnaseq"], batch["condition"], batch

def forward_logits_multimodal(model, batch_dict, x_img, x_rna):
    # 모델 구현에 따라 model(batch_dict) 또는 model(x_img, x_rna)일 수 있음
    try:
        return model(batch_dict)
    except Exception:
        return model(x_img, x_rna)

# =========================
# Main
# =========================
def main():
    assert CFG_YAML.exists(), f"Missing: {CFG_YAML}"
    assert CKPT.exists(), f"Missing: {CKPT}"

    cfg = OmegaConf.load(CFG_YAML)

    # ---- categories (LabelEncode 기준: 모델 output index -> treatment name) ----
    class_names, cat_path = load_categories_from_cfg(cfg)
    print(f"[CATEGORIES] loaded {len(class_names)} classes from: {cat_path}")
    # print(class_names)

    # ---- cfg에 박힌 treatment_select (각 split/modality) ----
    for sp in ["train", "val", "test"]:
        for mod in ["rnaseq", "imaging"]:
            ts = get_ts(cfg, sp, mod)
            if ts is None:
                print(f"[CFG] {sp}/{mod} treatment_select: None")
            else:
                print(f"[CFG] {sp}/{mod} treatment_select: {len(ts)}")
    # test에서 rnaseq vs imaging set 비교 (join에서 drop되는지 의심 포인트)
    ts_r = get_ts(cfg, "test", "rnaseq") or []
    ts_i = get_ts(cfg, "test", "imaging") or []
    inter = sorted(set(ts_r) & set(ts_i))
    only_r = sorted(set(ts_r) - set(ts_i))
    only_i = sorted(set(ts_i) - set(ts_r))
    print(f"[CFG] test intersection(rnaseq∩imaging): {len(inter)}")
    if "BAY 11-7082" in only_r:
        print("[CFG][IMPORTANT] BAY 11-7082 is in rnaseq-only (would be DROPPED by join).")
    if "BAY 11-7082" in only_i:
        print("[CFG][IMPORTANT] BAY 11-7082 is in imaging-only (would be DROPPED by join).")
    if "BAY 11-7082" in inter:
        print("[CFG][OK] BAY 11-7082 is in intersection(rnaseq∩imaging).")

    # ---- instantiate + load ckpt ----
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    ckpt_obj = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    # ---- test inference ----
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true_list, y_pred_list = [], []
    first_batch_debugged = False

    with torch.no_grad():
        for batch in loader:
            x_img, x_rna, y, batch_dict = pick_xy_multimodal(batch)

            # debug: 첫 배치에서 label 분포/unique 찍기
            if not first_batch_debugged:
                y_cpu = y.detach().cpu()
                if y_cpu.ndim > 1:
                    y_cpu = y_cpu.view(-1)
                u = torch.unique(y_cpu).numpy()
                print("[DEBUG] first batch y unique:", u[:50], " ...")
                first_batch_debugged = True

            x_img = x_img.to(device, non_blocking=True)
            x_rna = x_rna.to(device, non_blocking=True)

            batch_dict = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                          for k, v in batch_dict.items()}

            logits = forward_logits_multimodal(model, batch_dict, x_img, x_rna)
            pred = torch.argmax(logits, dim=-1).detach().cpu().view(-1)

            y_cpu = y.detach().cpu()
            if y_cpu.ndim > 1:
                y_cpu = y_cpu.view(-1)

            y_true_list.append(y_cpu)
            y_pred_list.append(pred)

    y_true = torch.cat(y_true_list).numpy().astype(int)
    y_pred = torch.cat(y_pred_list).numpy().astype(int)

    # ---- 핵심 디버그 1: y_true에 어떤 클래스가 실제로 존재? ----
    uniq = np.unique(y_true)
    print(f"[DEBUG] y_true unique count = {len(uniq)}")
    print(f"[DEBUG] y_true min={uniq.min()} max={uniq.max()}")
    # BAY index가 뭔지 categories에서 찾아보기
    if "BAY 11-7082" in class_names:
        bay_idx = class_names.index("BAY 11-7082")
        print(f"[DEBUG] BAY 11-7082 index in categories = {bay_idx}")
        print(f"[DEBUG] Does y_true contain BAY index? {bay_idx in set(uniq)}")
    else:
        print("[WARN] 'BAY 11-7082' not found in categories list!")

    # ---- 모델 output dimension 확인 ----
    C = int(model(x_img[:1], x_rna[:1]).shape[-1]) if False else None  # (안전상 미사용)
    # 대신 마지막 logits shape로 확인 (이미 loop에서 구했으니)
    # logits은 loop 마지막 값이 scope에 남아있음
    K = int(logits.shape[-1])
    print(f"logits C (num classes) = {K}")

    # ---- Header: categories (LabelEncode) 기준이 정답 ----
    if len(class_names) != K:
        raise RuntimeError(
            f"Header mismatch: categories has {len(class_names)} names but model outputs K={K}.\n"
            f"Refusing to write CSV with wrong headers."
        )

    labels = list(range(K))

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.int64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    # =========================
    # SAVE 3 FILES
    # =========================
    out_norm_npy = RUN_DIR / f"confusion_row_norm_{K}x{K}.npy"
    np.save(out_norm_npy, cm_norm)
    print("Saved:", out_norm_npy)

    out_counts_npy = RUN_DIR / f"confusion_counts_{K}x{K}.npy"
    np.save(out_counts_npy, cm)
    print("Saved:", out_counts_npy)

    # ===== per-treatment precision/recall/f1 from COUNT confusion matrix =====
    # cm: shape (K, K), rows=true, cols=pred
    tp = np.diag(cm).astype(float)
    row_sum = cm.sum(axis=1).astype(float)   # support per true class
    col_sum = cm.sum(axis=0).astype(float)   # total predicted per class

    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum != 0)
    recall    = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum != 0)
    f1        = np.divide(2 * precision * recall, precision + recall,
                        out=np.zeros_like(tp), where=(precision + recall) != 0)

    df_prf = pd.DataFrame(
        {
            "Treatment": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    out_prf_csv = RUN_DIR / f"per_treatment_prf_{K}x4_from_counts_with_header.csv"
    df_prf.to_csv(out_prf_csv, index=False, float_format="%.6f")
    print("Saved:", out_prf_csv)


    df_counts = pd.DataFrame(cm, index=class_names, columns=class_names)
    out_counts_csv = RUN_DIR / f"confusion_counts_{K}x{K}_with_header.csv"
    df_counts.to_csv(out_counts_csv, index=True)
    print("Saved:", out_counts_csv)

    df_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    out_norm_csv = RUN_DIR / f"confusion_row_norm_{K}x{K}_with_header.csv"
    df_norm.to_csv(out_norm_csv, index=True, float_format="%.6f")
    print("Saved:", out_norm_csv)

if __name__ == "__main__":
    main()
