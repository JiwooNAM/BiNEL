# compute_auroc_from_ckpt.py
from pathlib import Path
import argparse
import torch
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return ap.parse_args()



def load_cfg_from_run(run_dir: Path):
    cfg_path = (run_dir / ".hydra" / "config.yaml").resolve()
    cfg = OmegaConf.load(str(cfg_path))

    # ✅ ${hydra:run.dir} 같은 interpolation을 우리가 직접 풀어주도록 resolver를 등록
    #    ${hydra:xxx}에서 xxx는 "run.dir" / "runtime.output_dir" 같은 key로 들어옴.
    hydra_vals = {
        "run.dir": str(run_dir),
        "runtime.output_dir": str(run_dir),
        # 필요하면 추가:
        # "job.name": "compute_auroc_from_ckpt",
    }

    # replace=True 로 기존 hydra resolver가 있더라도 덮어씀
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: hydra_vals.get(key, ""),
        replace=True
    )

    return cfg


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt = sorted(ckpt_dir.glob("*.ckpt"))[-1]
    print(f"[INFO] ckpt: {ckpt}")

    cfg = load_cfg_from_run(run_dir)


    # (중요) config에 ${hydra:...} 같은 게 남아있으면 여기서 터질 수 있음.
    # 그럴 땐 아래 한 줄을 추가해서 안전하게 처리:
    # cfg.hydra.run.dir = str(run_dir)   # 필요한 경우에만

    # ✅ instantiate
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # ✅ load weights
    ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj
    model.load_state_dict(state, strict=False)
    model.eval()

    # ✅ dataloader
    datamodule.setup(stage=None)
    if args.split == "train":
        loader = datamodule.train_dataloader()
    elif args.split == "val":
        loader = datamodule.val_dataloader()
    else:
        loader = datamodule.test_dataloader()

    # TODO: 여기서 logits/probs 모아서 AUROC 계산 (네 기존 코드 이어붙이면 됨)


if __name__ == "__main__":
    main()
