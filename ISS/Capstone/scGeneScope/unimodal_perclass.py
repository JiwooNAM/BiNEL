from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

date="2026-02-10"
time = "07-06-52"
epochs = 2
steps = 108
RUN_DIR = Path(f"logs/train/runs/{date}_{time}").resolve()
CFG_DIR = (RUN_DIR / ".hydra").resolve()
CKPT    = (RUN_DIR / "checkpoints" / f"epoch={epochs}-step={steps}.ckpt").resolve()

def _load_categories_from_path(p: Path):
    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    if suf == ".json":
        return [str(x) for x in json.loads(p.read_text())]
    # txt/csv/tsv 등: 기본은 "한 줄 = 한 class"
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # 혹시 한 줄에 comma-separated면 split
    if len(lines) == 1 and ("," in lines[0]):
        lines = [x.strip() for x in lines[0].split(",") if x.strip()]
    return [str(x) for x in lines]


def _get_categories_path_from_cfg(cfg) -> Path:
    """
    cfg에서 LabelEncode의 path_to_categories 위치가
    - cfg.data.transform.condition.encode.path_to_categories
    - cfg.data.transform.example_transform.condition.encode.path_to_categories
    둘 중 어디에 있든 찾아서 instantiate 후 Path로 반환.
    """
    candidates = [
        ("cfg.data.transform.condition.encode.path_to_categories",
         lambda c: c.data.transform.condition.encode.path_to_categories),
        ("cfg.data.transform.example_transform.condition.encode.path_to_categories",
         lambda c: c.data.transform.example_transform.condition.encode.path_to_categories),
    ]

    last_err = None
    for name, getter in candidates:
        try:
            node = getter(cfg)
            p = hydra.utils.instantiate(node)  # resource resolver 등을 실제 경로로
            return Path(p)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not find/instantiate path_to_categories from cfg.\n"
        "Tried:\n"
        "  - data.transform.condition.encode.path_to_categories\n"
        "  - data.transform.example_transform.condition.encode.path_to_categories\n"
        f"Last error: {last_err}"
    )


def _pick_xy_unimodal(batch):
    """
    다양한 batch 형태를 robust하게 처리:
    - dict batch: 흔히 condition / rnaseq / imaging / embeddings 같은 키가 있음
    - tuple/list batch: (x, y, ...)
    """
    if isinstance(batch, dict):
        # x 후보들: embeddings / rnaseq / imaging / x / inputs
        x = None
        for k in ["embeddings", "rnaseq", "imaging", "x", "inputs"]:
            if k in batch:
                x = batch[k]
                break

        y = None
        for k in ["condition", "y", "labels", "target"]:
            if k in batch:
                y = batch[k]
                break

        return x, y
    else:
        # tuple/list
        x, y = batch[0], batch[1]
        return x, y


@hydra.main(version_base=None, config_path=str(CFG_DIR), config_name="config.yaml")
def main(cfg):
    assert CKPT.exists(), f"Missing: {CKPT}"

    # 1) instantiate datamodule/model exactly like training
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # 2) load checkpoint weights
    ckpt_obj = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    # 3) class names from LabelEncode categories file (this is the GT mapping: index -> treatment)
    cat_path = _get_categories_path_from_cfg(cfg)
    class_names = _load_categories_from_path(cat_path)
    K = len(class_names)
    print(f"[CATEGORIES] loaded {K} classes from: {cat_path}")

    # 4) run test inference
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true_chunks = []
    y_pred_chunks = []

    last_logits = None

    with torch.no_grad():
        for batch in loader:
            x, y = _pick_xy_unimodal(batch)
            if x is None or y is None:
                raise RuntimeError(f"Could not extract x/y from batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")

            x = x.to(device, non_blocking=True)
            logits = model(x)
            last_logits = logits

            pred = torch.argmax(logits, dim=-1).detach().cpu().view(-1)

            y_cpu = y.detach().cpu()
            if y_cpu.ndim > 1:
                y_cpu = y_cpu.view(-1)

            y_true_chunks.append(y_cpu)
            y_pred_chunks.append(pred)

    y_true = torch.cat(y_true_chunks).numpy().astype(int)
    y_pred = torch.cat(y_pred_chunks).numpy().astype(int)

    # sanity: model output dim vs categories length
    C = int(last_logits.shape[-1])
    print(f"logits C (num classes) = {C}")
    if C != K:
        raise RuntimeError(
            f"Header mismatch: categories has K={K} names but model outputs C={C} classes.\n"
            f"Refusing to write CSV with potentially wrong treatment-name headers."
        )

    # 5) per-class precision/recall/f1/support
    rep = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rep_df = (pd.DataFrame(rep).T
              .rename(columns={"f1-score": "f1"}))

    out_rep_csv = RUN_DIR / "per_class_report.csv"
    rep_df.to_csv(out_rep_csv, index=True)
    print("\nSaved:", out_rep_csv)

    # 6) confusion matrix: counts + row-normalized
    labels = list(range(K))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.int64)

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    # =========================
    # SAVE (multimodal과 동일 포맷)
    # =========================
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


    out_norm_npy = RUN_DIR / f"confusion_row_norm_{K}x{K}.npy"
    np.save(out_norm_npy, cm_norm)
    print("Saved:", out_norm_npy)

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
