import os, json, time, argparse, copy
from datetime import datetime

import baseline_train

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def make_run_dir(base_out_dir: str, cfg_path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_dir = os.path.join(base_out_dir, f"{cfg_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

from models_opt import CNN1D_MS
from baseline_train import CNN1D, TCN

def build_model(model_name: str, n_classes: int, use_meta: bool, meta_dim: int = 33):
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNN1D(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)
    elif model_name == "tcn":
        return TCN(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)
    elif model_name == "cnn_ms":
        return CNN1D_MS(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="path to json config")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    if "out_root" in cfg and cfg["out_root"]:
        run_dir = make_run_dir(cfg["out_root"], args.cfg)
        cfg = copy.deepcopy(cfg)
        cfg["out_dir"] = run_dir
    else:
        base_out = cfg.get("out_dir", "runs/exp")
        run_dir = make_run_dir(base_out, args.cfg)
        cfg = copy.deepcopy(cfg)
        cfg["out_dir"] = run_dir

    with open(os.path.join(run_dir, "cfg_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    t0 = time.perf_counter()
    history, test_metrics = baseline_train.run_experiment(cfg, model_builder=build_model)
    total_sec = time.perf_counter() - t0

    # 追加总耗时信息
    with open(os.path.join(run_dir, "time.json"), "w", encoding="utf-8") as f:
        json.dump({"total_time_sec": total_sec}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] run_dir={run_dir} total_time_sec={total_sec:.2f}")

if __name__ == "__main__":
    main()