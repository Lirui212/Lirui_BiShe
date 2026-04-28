import os
import json
import time
import copy
from types import SimpleNamespace
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from baseline_train import (
    load_whole_data,
    onehot_to_index,
    make_val_split,
    WholeDataDataset,
    evaluate,
    save_confusion_matrix,
    set_seed,
    plot_training_curves,
    parse_train_class_counts,
    subsample_train_indices_by_class,
    save_class_count_records,
    build_per_class_recall_records,
    build_confmat_class_names,
)

from losses_opt import (
    CenterLoss,
    BalancedSupConLoss,
    MultiCenterLoss,
)

from train_trainlayer import (
    load_cfg,
    make_run_dir,
    build_model,
    build_criterion,
    build_sampler,
)

LABEL_INFO = {
    12: {"label": "C0", "label_name": "无故障",               "data_name": "L12 正常运行"},
    13: {"label": "C1", "label_name": "驱动端轴承保持架故障", "data_name": "L13 轴承支架破坏"},
    10: {"label": "C2", "label_name": "非驱动端轴承外圈故障", "data_name": "L10 西边轴承滚珠外圈磨损"},
    9:  {"label": "C3", "label_name": "双侧轴承保持架故障",   "data_name": "L9 双轴承支架破坏"},
    11: {"label": "C4", "label_name": "双侧轴承外圈故障",     "data_name": "L11 双轴承滚珠外圈破坏性磨损"},
    7:  {"label": "C5", "label_name": "电机偏心",             "data_name": "L7 电机偏心"},
}


def get_embed_lambda(epoch, total_epochs, max_lambda=1e-3, start_ratio=0.3, ramp_ratio=0.3):
    r = epoch / max(total_epochs - 1, 1)
    if r < start_ratio:
        return 0.0
    elif r < start_ratio + ramp_ratio:
        u = (r - start_ratio) / max(ramp_ratio, 1e-12)
        return max_lambda * u
    else:
        return max_lambda


# 兼容旧代码名
get_center_lambda = get_embed_lambda


def normalize_embed_cfg(cfg: dict) -> dict:
    """
    兼容旧版 center-loss 配置，同时支持更通用的嵌入约束配置。
    """
    cfg = copy.deepcopy(cfg)

    if "embed_constraint_type" not in cfg:
        if cfg.get("use_centerloss", False):
            cfg["embed_constraint_type"] = "center"
        else:
            cfg["embed_constraint_type"] = cfg.get("embed_constraint", "none")

    if "use_embed_constraint" not in cfg:
        if "use_centerloss" in cfg:
            cfg["use_embed_constraint"] = bool(cfg.get("use_centerloss", False))
        else:
            cfg["use_embed_constraint"] = cfg.get("embed_constraint_type", "none").lower() != "none"

    if "embed_lambda" not in cfg:
        cfg["embed_lambda"] = cfg.get("center_lambda", 1e-3)
    if "embed_start_ratio" not in cfg:
        cfg["embed_start_ratio"] = cfg.get("center_start_ratio", 0.3)
    if "embed_ramp_ratio" not in cfg:
        cfg["embed_ramp_ratio"] = cfg.get("center_ramp_ratio", 0.3)

    return cfg


def build_embed_constraint(args, feat_dim, n_classes, device):
    embed_type = str(getattr(args, "embed_constraint_type", "none")).lower().strip()
    use_flag = bool(getattr(args, "use_embed_constraint", False))

    if (not use_flag) or embed_type == "none":
        return None, None, "none"

    if embed_type == "center":
        criterion = CenterLoss(
            num_classes=n_classes,
            feat_dim=feat_dim,
            device=device,
            normalize=getattr(args, "center_normalize", True),
        )
        optimizer = torch.optim.SGD(criterion.parameters(), lr=args.center_lr)
        return criterion, optimizer, embed_type

    if embed_type in {"supcon", "supcon_bal", "balanced_supcon"}:
        balance_mode = "class" if embed_type in {"supcon_bal", "balanced_supcon"} else "sample"
        criterion = BalancedSupConLoss(
            temperature=args.supcon_temperature,
            base_temperature=args.supcon_base_temperature,
            normalize=args.supcon_normalize,
            balance_mode=balance_mode,
        )
        return criterion, None, embed_type

    if embed_type in {"multicenter", "multi_center", "subcenter"}:
        criterion = MultiCenterLoss(
            num_classes=n_classes,
            feat_dim=feat_dim,
            device=device,
            num_centers=args.multicenter_num_centers,
            normalize=args.center_normalize,
            balance_mode=args.multicenter_balance_mode,
            separation_weight=args.multicenter_separation_weight,
            separation_margin=args.multicenter_separation_margin,
        )
        optimizer = torch.optim.SGD(criterion.parameters(), lr=args.center_lr)
        return criterion, optimizer, "multicenter"

    raise ValueError(f"Unknown embed_constraint_type: {embed_type}")


def extract_embeddings(model, dataloader, device, use_meta=False):
    model.eval()
    all_embed = []
    all_y = []

    with torch.no_grad():
        for batch in dataloader:
            if use_meta:
                xw, xm, y = batch
                xw = xw.to(device)
                xm = xm.to(device)
                _, embed = model(xw, xm, return_embed=True)
            else:
                xw, y = batch
                xw = xw.to(device)
                _, embed = model(xw, None, return_embed=True)

            all_embed.append(embed.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())

    X = np.concatenate(all_embed, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def get_class_name_map(motor_ids):
    name_map = {}
    for new_id, old_id in enumerate(motor_ids):
        if old_id in LABEL_INFO:
            info = LABEL_INFO[old_id]
            name_map[new_id] = f"{info['label']}:{info['label_name']}"
        else:
            name_map[new_id] = f"class_{new_id}"
    return name_map


def downsample_per_class(X, y, max_points_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    keep_indices = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        keep_indices.append(idx)

    keep_indices = np.concatenate(keep_indices)
    keep_indices = np.sort(keep_indices)
    return X[keep_indices], y[keep_indices]


def run_tsne(X, pca_dim=50, perplexity=30, random_state=42):
    if X.shape[0] < 3:
        raise ValueError("样本太少，无法做 t-SNE")

    pca_dim = min(pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)

    perplexity = min(perplexity, max(2, X.shape[0] // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X)
    return X_2d


def plot_tsne_overall(X_2d, y, class_name_map, save_path, title="t-SNE of test embeddings"):
    plt.figure(figsize=(10, 8))
    classes = np.unique(y)

    for c in classes:
        mask = (y == c)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=18,
            alpha=0.75,
            label=class_name_map.get(int(c), f"class_{c}"),
        )

    plt.title(title)
    plt.legend(fontsize=8, markerscale=1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_tail_highlight(X_2d, y, class_name_map, tail_classes, save_path,
                             title="t-SNE tail classes highlighted"):
    plt.figure(figsize=(10, 8))

    tail_classes = set(int(c) for c in tail_classes)
    for c in np.unique(y):
        mask = (y == c)
        is_tail = int(c) in tail_classes

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=26 if is_tail else 12,
            alpha=0.9 if is_tail else 0.18,
            label=class_name_map.get(int(c), f"class_{c}") if is_tail else None,
        )

    plt.title(title)
    plt.legend(fontsize=9, markerscale=1.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def export_tsne_figures(
    model,
    dataloader,
    device,
    out_dir,
    motor_ids,
    use_meta=False,
    class_counts=None,
    random_state=42,
    perplexity=30,
    pca_dim=50,
    max_points_per_class=300,
    tail_k=3,
):
    X, y = extract_embeddings(model, dataloader, device, use_meta=use_meta)

    np.save(os.path.join(out_dir, "test_embed.npy"), X)
    np.save(os.path.join(out_dir, "test_labels.npy"), y)

    X_vis, y_vis = downsample_per_class(
        X, y,
        max_points_per_class=max_points_per_class,
        seed=random_state,
    )

    X_2d = run_tsne(
        X_vis,
        pca_dim=pca_dim,
        perplexity=perplexity,
        random_state=random_state,
    )

    class_name_map = get_class_name_map(motor_ids)

    plot_tsne_overall(
        X_2d, y_vis, class_name_map,
        save_path=os.path.join(out_dir, "tsne_test_overall.png"),
        title="t-SNE of test embeddings (overall)",
    )

    tail_classes = []
    if class_counts is not None:
        class_counts = np.asarray(class_counts)
        tail_classes = np.argsort(class_counts)[:tail_k].tolist()

        with open(os.path.join(out_dir, "tail_classes.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tail_k": int(tail_k),
                    "tail_classes_new_id": [int(x) for x in tail_classes],
                    "tail_class_names": [class_name_map[int(x)] for x in tail_classes],
                    "class_counts": class_counts.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        plot_tsne_tail_highlight(
            X_2d, y_vis, class_name_map, tail_classes,
            save_path=os.path.join(out_dir, "tsne_test_tail_highlight.png"),
            title="t-SNE of test embeddings (tail classes highlighted)",
        )
    np.save(os.path.join(out_dir, "tsne_test_2d.npy"), X_2d)


def run_experiment_centerloss(cfg: dict):
    cfg = normalize_embed_cfg(cfg)

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

        sampler_type="none",          # none / weighted
        base_loss_type="ce",          # ce / focal / cb_ce / stage_comp
        stage_beta=0.999,
        stage_gamma_max=2.5,
        focal_gamma=2.0,
        cb_beta=0.9999,

        # 通用嵌入约束配置
        use_embed_constraint=False,
        embed_constraint_type="none", # none / center / supcon / supcon_bal / multicenter
        embed_lambda=0.001,
        embed_start_ratio=0.3,
        embed_ramp_ratio=0.3,

        # Center Loss 特定参数
        center_lr=0.05,
        center_normalize=True,

        # SupCon 特定参数
        supcon_temperature=0.07,
        supcon_base_temperature=0.07,
        supcon_normalize=True,

        # MultiCenter 特定参数
        multicenter_num_centers=2,
        multicenter_balance_mode="class",     # class / sample
        multicenter_separation_weight=0.1,
        multicenter_separation_margin=0.4,

        # t-SNE 设置
        export_tsne=True,
        tsne_random_state=42,
        tsne_perplexity=30,
        tsne_pca_dim=50,
        tsne_max_points_per_class=300,
        tsne_tail_k=3,

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

    id_map = {old: new for new, old in enumerate(motor_ids)}
    y_tr = np.array([id_map[c] for c in y_tr], dtype=np.int64)
    y_te = np.array([id_map[c] for c in y_te], dtype=np.int64)

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
            indent=2,
        )

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

    global_mean, global_std = None, None
    if args.norm == "global_train":
        global_mean = wave_tr[tr_idx].mean()
        global_std = wave_tr[tr_idx].std() + 1e-8

    ds_train = WholeDataDataset(
        wave_tr, meta_tr, y_tr, tr_idx,
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std,
    )
    ds_val = WholeDataDataset(
        wave_tr, meta_tr, y_tr, va_idx,
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std,
    )
    ds_test = WholeDataDataset(
        wave_te, meta_te, y_te, np.arange(len(y_te)),
        use_meta=args.use_meta,
        meta_only=args.meta_only,
        norm_mode=args.norm,
        global_mean=global_mean,
        global_std=global_std,
    )

    num_workers = getattr(args, "num_workers", 4)
    pin_memory = getattr(args, "pin_memory", True)
    prefetch_factor = getattr(args, "prefetch_factor", 2)
    drop_last = getattr(args, "drop_last", True)

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
        dl_train = DataLoader(ds_train, sampler=sampler, shuffle=False, **train_loader_kwargs)
    else:
        dl_train = DataLoader(ds_train, shuffle=True, **train_loader_kwargs)

    dl_val = DataLoader(ds_val, shuffle=False, **val_loader_kwargs)
    dl_test = DataLoader(ds_test, shuffle=False, **test_loader_kwargs)

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
        meta_dim=33,
    )
    model.to(device)

    with torch.no_grad():
        B = 2
        xw = torch.randn(B, 1, 2048, device=device)
        if args.use_meta:
            xm = torch.randn(B, 33, device=device)
            logits, embed = model(xw, xm, return_embed=True)
        else:
            logits, embed = model(xw, None, return_embed=True)
        assert logits.shape == (B, n_classes), f"Bad logits shape: {logits.shape}"
        feat_dim = embed.shape[1]

    base_criterion = build_criterion(
        loss_type=args.base_loss_type,
        counts=counts,
        device=device,
        total_epochs=args.epochs,
        focal_gamma=args.focal_gamma,
        cb_beta=args.cb_beta,
        stage_beta=args.stage_beta,
        stage_gamma_max=args.stage_gamma_max,
    )
    base_criterion = base_criterion.to(device)

    embed_criterion, optimizer_embed, embed_type = build_embed_constraint(
        args=args,
        feat_dim=feat_dim,
        n_classes=n_classes,
        device=device,
    )

    print(f"[INFO] Model={type(model).__name__}, feat_dim={feat_dim}, device={next(model.parameters()).device}")
    print(
        f"[INFO] sampler_type={args.sampler_type}, base_loss_type={args.base_loss_type}, "
        f"embed_constraint_type={embed_type}, use_embed_constraint={args.use_embed_constraint}"
    )

    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        cls_loss_sum = 0.0
        embed_loss_sum = 0.0
        n = 0
        t0 = time.perf_counter()

        for batch in dl_train:
            optimizer_model.zero_grad()
            if optimizer_embed is not None:
                optimizer_embed.zero_grad()

            if args.use_meta:
                xw, xm, y = batch
                xw, xm, y = xw.to(device), xm.to(device), y.to(device)
                logits, embed = model(xw, xm, return_embed=True)
            else:
                xw, y = batch
                xw, y = xw.to(device), y.to(device)
                logits, embed = model(xw, None, return_embed=True)

            if args.base_loss_type == "stage_comp":
                cls_loss, loss_info = base_criterion(logits, y, ep - 1)
            else:
                cls_loss = base_criterion(logits, y)
                loss_info = None

            current_embed_lambda = 0.0
            if embed_criterion is not None:
                current_embed_lambda = get_embed_lambda(
                    ep - 1,
                    args.epochs,
                    max_lambda=args.embed_lambda,
                    start_ratio=args.embed_start_ratio,
                    ramp_ratio=args.embed_ramp_ratio,
                )
                embed_loss = embed_criterion(embed, y)
                loss = cls_loss + current_embed_lambda * embed_loss
            else:
                embed_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

            loss.backward()
            optimizer_model.step()

            if optimizer_embed is not None:
                # 对 center 参数做反缩放，保持其更新量不被 lambda 压得过小
                if current_embed_lambda > 0:
                    for param in embed_criterion.parameters():
                        if param.grad is not None:
                            param.grad.data *= (1.0 / max(current_embed_lambda, 1e-12))
                    optimizer_embed.step()

            loss_sum += loss.item() * y.size(0)
            cls_loss_sum += cls_loss.item() * y.size(0)
            embed_loss_sum += embed_loss.item() * y.size(0)
            n += y.size(0)

        train_loss = loss_sum / max(1, n)
        train_cls_loss = cls_loss_sum / max(1, n)
        train_embed_loss = embed_loss_sum / max(1, n)

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
                    "feat_dim": feat_dim,
                    "embed_constraint_type": embed_type,
                },
                best_path,
            )

        epoch_time = time.perf_counter() - t0

        log = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "train_cls_loss": float(train_cls_loss),
            "train_embed_loss": float(train_embed_loss),
            "embed_constraint_type": embed_type,
            "embed_lambda": float(current_embed_lambda),
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

    if args.export_tsne:
        export_tsne_figures(
            model=model,
            dataloader=dl_test,
            device=device,
            out_dir=args.out_dir,
            motor_ids=motor_ids,
            use_meta=args.use_meta,
            class_counts=counts,
            random_state=args.tsne_random_state,
            perplexity=args.tsne_perplexity,
            pca_dim=args.tsne_pca_dim,
            max_points_per_class=args.tsne_max_points_per_class,
            tail_k=args.tsne_tail_k,
        )

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
        "topk=", test_metrics["topk_acc"],
    )

    return history, {k: v for k, v in test_metrics.items() if k not in ["y_true", "y_pred"]}


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
    history, test_metrics = run_experiment_centerloss(cfg)
    total_sec = time.perf_counter() - t0

    with open(os.path.join(run_dir, "time.json"), "w", encoding="utf-8") as f:
        json.dump({"total_time_sec": total_sec}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] run_dir={run_dir} total_time_sec={total_sec:.2f}")


if __name__ == "__main__":
    main()
