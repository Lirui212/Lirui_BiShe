import os
import json
import time
import copy
from datetime import datetime
from types import SimpleNamespace
import argparse

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from baseline_train import(
    load_whole_data,
    onehot_to_index,
    make_val_split,
    WholeDataDataset,
    evaluate,
    save_confusion_matrix,
    compute_class_weights,
    set_seed,
    CNN1D,
    TCN,
    _series,
    plot_training_curves,
    parse_train_class_counts,
    subsample_train_indices_by_class,
    save_class_count_records,
    build_per_class_recall_records,
    build_confmat_class_names,
)

from models_opt import(
    CNN1D_MS,
    CNN1D_MS_1Stage,
)

from losses_opt import(
    FocalLoss,
    compute_cb_weights_from_counts,
    StageAdaptiveCompositeLoss,
)

LABEL_INFO = {
    12: {"label": "C0", "label_name": "无故障",               "data_name": "L12 正常运行"},
    13: {"label": "C1", "label_name": "驱动端轴承保持架故障", "data_name": "L13 轴承支架破坏"},
    10: {"label": "C2", "label_name": "非驱动端轴承外圈故障", "data_name": "L10 西边轴承滚珠外圈磨损"},
    9:  {"label": "C3", "label_name": "双侧轴承保持架故障",   "data_name": "L9 双轴承支架破坏"},
    11: {"label": "C4", "label_name": "双侧轴承外圈故障",     "data_name": "L11 双轴承滚珠外圈破坏性磨损"},
    7:  {"label": "C5", "label_name": "电机偏心",             "data_name": "L7 电机偏心"},
}  # 类别字典，对应原始数据标签，同baseline_train中内容

# 0) cfg读取，run_dir设置
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_run_dir(base_out_dir: str, cfg_path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_dir = os.path.join(base_out_dir, f"{cfg_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# 1) build_model, 同train_opt文件中
def build_model(model_name: str, n_classes: int, use_meta: bool, meta_dim: int = 33):
    model_name = model_name.lower().strip()
    # baseline
    if model_name == "cnn":
        return CNN1D(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)
    if model_name == "tcn":
        return TCN(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)

    # multi-scale
    if model_name.startswith("cnn_ms"):
        # stage1和2
        if model_name.startswith("cnn_ms2"):
            ModelCls = CNN1D_MS
        elif model_name.startswith("cnn_ms1"):
            ModelCls = CNN1D_MS_1Stage
        else:
            raise ValueError(
                f"Unknown ms stage in model name: {model_name}. "
                f"Use cnn_ms1_* or cnn_ms2_*"
            )

        # kernels
        if model_name.endswith("3715"):
            ms_kernels = (3, 7, 15)
        elif model_name.endswith("359"):
            ms_kernels = (3, 5, 9)
        else:
            raise ValueError(
                f"Unknown kernels suffix in model name: {model_name}. "
                f"Use *_3715 or *_359"
            )

        return ModelCls(
            n_classes=n_classes,
            use_meta=use_meta,
            meta_dim=meta_dim,
            ms_kernels=ms_kernels
        )

    raise ValueError(f"Unknown model: {model_name}")

# 2) sampler 和 criterion
def build_sampler(y_train_subset: np.ndarray, sampler_type="none"):
    """
    y_train_subset: 已经 remap 后的训练集标签 (只包含 train 子集)
    """
    sampler_type = sampler_type.lower().strip()

    if sampler_type == "none":
        return None

    if sampler_type == "weighted":
        counts = np.bincount(y_train_subset)
        class_weights = 1.0 / (counts + 1e-12)
        sample_weights = class_weights[y_train_subset]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    raise ValueError(f"Unknown sampler_type: {sampler_type}")


def build_criterion(
    loss_type,
    counts,
    device,
    total_epochs=None,
    focal_gamma=2.0,
    cb_beta=0.9999,
    stage_beta=0.999,
    stage_gamma_max=2.5,
):
    """
    counts: 训练子集各类别样本数
    """
    loss_type = loss_type.lower().strip()
    counts = np.asarray(counts, dtype=np.float64)

    if loss_type == "ce":
        # 与 baseline 一致：逆频率加权 CE
        weights = (counts.sum() / (counts + 1e-12))
        weights = weights / weights.mean()
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)

    elif loss_type == "raw":
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        weights = (counts.sum() / (counts + 1e-12))
        weights = weights / weights.mean()
        alpha_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return FocalLoss(alpha=alpha_tensor, gamma=focal_gamma)

    elif loss_type == "cb_ce":
        cb_weights = compute_cb_weights_from_counts(counts, beta=cb_beta)
        cb_tensor = torch.tensor(cb_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=cb_tensor)
    
    elif loss_type == "stage_comp":
        return StageAdaptiveCompositeLoss(
            class_counts=counts,
            total_epochs=total_epochs,
            beta=stage_beta,
            gamma_max=stage_gamma_max
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# 3) run_experiment
def run_experiment_trainlayer(cfg: dict):
    defaults = dict(
        model="cnn",
        use_meta=False,
        meta_only=False,
        norm="per_sample",
        epochs=20,
        batch_size=256,
        lr=1e-3,
        val_ratio=0.1,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        classes="",
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,

        sampler_type="none",   # none / weighted
        loss_type="ce",        # ce / focal / cb_ce / stage_comp
        stage_beta=0.999,
        stage_gamma_max=2.5,
        focal_gamma=2.0,
        cb_beta=0.9999,

        
        # 数据情景设置
        use_custom_train_counts=False,
        train_class_counts={},
        train_count_cfg_uses_old_ids=True,
        train_count_seed=None,
        train_count_strict=False,
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    args = SimpleNamespace(**cfg)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # 数据读取与筛选
    # -------------------------
    (wave_tr, meta_tr, y_tr_oh), (wave_te, meta_te, y_te_oh) = load_whole_data(args.data_path)
    y_tr = onehot_to_index(y_tr_oh)
    y_te = onehot_to_index(y_te_oh)

    if args.classes.strip():
        motor_ids = [int(x.strip()) for x in args.classes.split(",") if x.strip() != ""]
        motor_ids = sorted(set(motor_ids))
    else:
        motor_ids = list(range(16))

    n_classes = len(motor_ids)
    assert n_classes >= 2, f"Need at least 2 classes, got {motor_ids}"

    mask_tr = np.isin(y_tr, motor_ids)
    mask_te = np.isin(y_te, motor_ids)

    assert mask_tr.sum() > 0, f"No train samples for classes {motor_ids}"
    assert mask_te.sum() > 0, f"No test samples for classes {motor_ids}"

    wave_tr, meta_tr, y_tr = wave_tr[mask_tr], meta_tr[mask_tr], y_tr[mask_tr]
    wave_te, meta_te, y_te = wave_te[mask_te], meta_te[mask_te], y_te[mask_te]

    # remap old ids -> 0..K-1
    id_map = {old: new for new, old in enumerate(motor_ids)}
    y_tr = np.array([id_map[c] for c in y_tr], dtype=np.int64)
    y_te = np.array([id_map[c] for c in y_te], dtype=np.int64)

    # 保存标签映射，便于后续读 recall
    label_map = {str(v): f"L{old}" for old, v in id_map.items()}
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "motor_ids_sorted": motor_ids,
                "id_map_old_to_new": id_map,
                "label_map_new_to_old": label_map,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # split train -> val
    tr_idx, va_idx = make_val_split(y_tr, val_ratio=args.val_ratio, seed=args.seed)
    if args.use_custom_train_counts:
        count_seed = args.train_count_seed if args.train_count_seed is not None else args.seed
        target_counts = parse_train_class_counts(
            args.train_class_counts,
            id_map=id_map,
            use_old_ids=args.train_count_cfg_uses_old_ids
        )
        tr_idx = subsample_train_indices_by_class(
            y=y_tr,
            train_idx=tr_idx,
            target_counts=target_counts,
            seed=count_seed,
            strict=args.train_count_strict
        )

    # global norm
    global_mean, global_std = None, None
    if args.norm == "global_train":
        global_mean = wave_tr[tr_idx].mean()
        global_std = wave_tr[tr_idx].std() + 1e-8

    # datasets
    ds_train = WholeDataDataset(
        wave_tr, meta_tr, y_tr, tr_idx,
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std
    )
    ds_val = WholeDataDataset(
        wave_tr, meta_tr, y_tr, va_idx,
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std
    )
    ds_test = WholeDataDataset(
        wave_te, meta_te, y_te, np.arange(len(y_te)),
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std
    )

    # DataLoader + sampler
    num_workers = getattr(args, "num_workers", 0)
    pin_memory = getattr(args, "pin_memory", False)
    prefetch_factor = getattr(args, "prefetch_factor", 1)
    drop_last = getattr(args, "drop_last", False)

    y_train_subset = y_tr[tr_idx]
    sampler = build_sampler(y_train_subset, sampler_type=args.sampler_type)

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
        test_loader_kwargs["prefetch_factor"] = prefetch_factor

    if sampler is not None:
        dl_train = DataLoader(
            ds_train,
            sampler=sampler,
            shuffle=False,
            **train_loader_kwargs
        )
    else:
        dl_train = DataLoader(
            ds_train,
            shuffle=True,
            **train_loader_kwargs
        )

    dl_val = DataLoader(ds_val, shuffle=False, **val_loader_kwargs)
    dl_test = DataLoader(ds_test, shuffle=False, **test_loader_kwargs)

    # 类别统计 / loss / model
    counts = np.bincount(y_train_subset, minlength=n_classes).astype(np.float64)
    print("[INFO] Train class counts:", counts.tolist())

    save_class_count_records(
        out_path=os.path.join(args.out_dir, "class_counts_detailed.json"),
        motor_ids=motor_ids,
        id_map=id_map,
        counts=counts,
        split_name="train"
    )

    device = torch.device(args.device)

    model = build_model(
        model_name=args.model,
        n_classes=n_classes,
        use_meta=args.use_meta,
        meta_dim=33
    )
    model.to(device)

    # 自检
    with torch.no_grad():
        B = 2
        xw = torch.randn(B, 1, 2048, device=device)
        if args.use_meta:
            xm = torch.randn(B, 33, device=device)
            out = model(xw, xm)
        else:
            out = model(xw, None)
        assert out.shape == (B, n_classes), f"Bad logits shape: {out.shape}"

    print(f"[INFO] Model={type(model).__name__}, device={next(model.parameters()).device}")
    print(f"[INFO] sampler_type={args.sampler_type}, loss_type={args.loss_type}")

    criterion = build_criterion(
    loss_type=args.loss_type,
    counts=counts,
    device=device,
    total_epochs=args.epochs,
    focal_gamma=args.focal_gamma,
    cb_beta=args.cb_beta,
    stage_beta=args.stage_beta,
    stage_gamma_max=args.stage_gamma_max,
)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    history = []

    # 训练
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        t0 = time.perf_counter()

        for batch in dl_train:
            optimizer.zero_grad()

            if args.use_meta:
                xw, xm, y = batch
                xw, xm, y = xw.to(device), xm.to(device), y.to(device)
                logits = model(xw, xm)
            else:
                xw, y = batch
                xw, y = xw.to(device), y.to(device)
                logits = model(xw, None)

            if args.loss_type == "stage_comp":
                loss, loss_info = criterion(logits, y, ep - 1)
            else:
                loss = criterion(logits, y)
                loss_info = None
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = loss_sum / max(1, n)

        val_metrics = evaluate(model, dl_val, device, n_classes=n_classes, use_meta=args.use_meta)
        score = val_metrics["macro_f1"]

        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": vars(args),
                    "motor_ids": motor_ids,
                    "id_map": id_map,
                },
                best_path
            )

        epoch_time = time.perf_counter() - t0

        log = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_topk": val_metrics["topk_acc"],
            "val_avg_conf": val_metrics["avg_confidence"],
            "best_val_macro_f1": float(best_val),
            "epoch_time": epoch_time,
        }
        if loss_info is not None:
            log.update({
                "train_loss_ce": loss_info["loss_ce"],
                "train_loss_cb": loss_info["loss_cb"],
                "train_loss_fl": loss_info["loss_fl"],
                "lambda_ce": loss_info["lambda_ce"],
                "lambda_cb": loss_info["lambda_cb"],
                "lambda_fl": loss_info["lambda_fl"],
                "gamma_stage": loss_info["gamma"],
            })
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    plot_training_curves(history, args.out_dir, prefix="curve")

    test_metrics = evaluate(model, dl_test, device, n_classes=n_classes, use_meta=args.use_meta)

    test_metrics_save = {k: v for k, v in test_metrics.items() if k not in ["y_true", "y_pred"]}
    test_metrics_save["per_class_recall_detail"] = build_per_class_recall_records(
        per_class_recall=test_metrics["per_class_recall"],
        motor_ids=motor_ids,
        id_map=id_map
    )

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics_save, f, ensure_ascii=False, indent=2)

    class_names = build_confmat_class_names(motor_ids, id_map)

    save_confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        os.path.join(args.out_dir, "confusion_matrix.png"),
        n_classes=n_classes,
        class_names=class_names,
        normalize="true"
    )

    print(
        "\n[TEST] acc=", test_metrics["acc"],
        "macro_f1=", test_metrics["macro_f1"],
        "balanced_acc=", test_metrics["balanced_acc"],
        "topk=", test_metrics["topk_acc"]
    )

    return history, {k: v for k, v in test_metrics.items() if k not in ["y_true", "y_pred"]}


# -------------------------
# 4) main
# -------------------------
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
    history, test_metrics = run_experiment_trainlayer(cfg)
    total_sec = time.perf_counter() - t0

    with open(os.path.join(run_dir, "time.json"), "w", encoding="utf-8") as f:
        json.dump({"total_time_sec": total_sec}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] run_dir={run_dir} total_time_sec={total_sec:.2f}")


if __name__ == "__main__":
    main()