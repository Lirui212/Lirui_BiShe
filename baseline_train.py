import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time

LABEL_INFO = {
    12: {"label": "C0", "label_name": "无故障",               "data_name": "L12 正常运行"},
    13: {"label": "C1", "label_name": "驱动端轴承保持架故障", "data_name": "L13 轴承支架破坏"},
    10: {"label": "C2", "label_name": "非驱动端轴承外圈故障", "data_name": "L10 西边轴承滚珠外圈磨损"},
    9:  {"label": "C3", "label_name": "双侧轴承保持架故障",   "data_name": "L9 双轴承支架破坏"},
    11: {"label": "C4", "label_name": "双侧轴承外圈故障",     "data_name": "L11 双轴承滚珠外圈破坏性磨损"},
    7:  {"label": "C5", "label_name": "电机偏心",             "data_name": "L7 电机偏心"},
}  # 类别字典，对应原始数据标签


# 0 复现性
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1 数据wholedata.py 已经预处理好并保存为 npy，加载并拆分
def load_whole_data(npy_path: str):
    data = np.load(npy_path, allow_pickle=True).item()

    x_train = np.array(data["train_sample"]).reshape(-1, 2081)
    y_train = np.array(data["train_label"])
    x_test  = np.array(data["test_sample"]).reshape(-1, 2081)
    y_test  = np.array(data["test_label"])

    # [wave 2048, sensor 15, condition 18]
    def split_x(x):
        wave = x[:, :2048]
        sensor = x[:, 2048:2063]
        cond = x[:, 2063:2081]
        meta = np.concatenate([sensor, cond], axis=1)  # 合并取用
        return wave, meta

    wave_tr, meta_tr = split_x(x_train)
    wave_te, meta_te = split_x(x_test)

    # y (N,16) one-hot
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (wave_tr, meta_tr, y_train), (wave_te, meta_te, y_test)

def onehot_to_index(y_onehot: np.ndarray):
    sums = y_onehot.sum(axis=1)
    bad = np.where(sums != 1)[0]
    if len(bad) > 0:
        print(f"[WARN] Found {len(bad)} samples with one-hot sum != 1. Example idx: {bad[:10].tolist()}")
    y_idx = np.argmax(y_onehot, axis=1).astype(np.int64)
    return y_idx

# 2 Train/Val split 针对类别不平衡
def make_val_split(y: np.ndarray, val_ratio=0.1, seed=42):
    """
    按类别拆分: max(1, int(val_ratio*n_c))
    避免验证集中缺失稀有类别
    """
    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(y))
    val_idx = []
    train_idx = []

    for c in np.unique(y):
        idx_c = idx_all[y == c]
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        if n_c <= 1:
            # 极稀少类放训练集
            train_idx.extend(idx_c.tolist())
            continue
        n_val = max(1, int(round(val_ratio * n_c)))
        n_val = min(n_val, n_c - 1)
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)

# 3 数据集类，归一化
class WholeDataDataset(Dataset):
    def __init__(self, wave, meta, y_idx, indices,
                 use_meta=True,
                 meta_only = False,
                 norm_mode="per_sample",
                 global_mean=None,
                 global_std=None):
        self.wave = wave[indices].astype(np.float32)
        self.meta = meta[indices].astype(np.float32)
        self.y = y_idx[indices].astype(np.int64)
        self.use_meta = use_meta
        self.meta_only = meta_only
        self.norm_mode = norm_mode
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.y)

    def _norm_wave(self, x):
        # x shape: (2048,)
        if self.norm_mode == "per_sample":
            m = x.mean()
            s = x.std() + 1e-8
            return (x - m) / s
        elif self.norm_mode == "global_train":
            return (x - self.global_mean) / (self.global_std + 1e-8)
        else:
            return x

    def __getitem__(self, i):
        xw = self._norm_wave(self.wave[i])
        xw = torch.from_numpy(xw).unsqueeze(0)  # (1, 2048)
        y = torch.tensor(self.y[i], dtype=torch.long)
        if self.use_meta:
            xm = torch.from_numpy(self.meta[i])  # (33,)
            if self.meta_only:  # 如果设置只用meta，跑meta_only实验
                xw = torch.zeros_like(xw)
                return xw, xm, y
            return xw, xm, y
            
        else:
            return xw, y

# 4 CNN&TCN模型，简单baseline-
class CNN1D(nn.Module):
    def __init__(self, n_classes=16, use_meta=True, meta_dim=33):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = 256
        if use_meta:
            self.head = nn.Sequential(
                nn.Linear(feat_dim + meta_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes)
            )

    def forward(self, x_wave, x_meta=None, return_embed=False):
        z = self.backbone(x_wave)         # (B,256,T)
        z = self.pool(z).squeeze(-1)      # (B,256)
        embed = z

        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)

        logits = self.head(z)
        if return_embed:
            return logits, embed

        return logits

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, dropout=0.15):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_embed=False):
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        y = self.drop(F.relu(self.bn2(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TCN(nn.Module):
    def __init__(self, n_classes=16, use_meta=True, meta_dim=33, channels=(16,32,64,128)):
        super().__init__()
        self.use_meta = use_meta
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, k=5, dilation=2**i, dropout=0.15))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = channels[-1]
        if use_meta:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim + meta_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes)
            )

    def forward(self, x_wave, x_meta=None, return_embed=False):
        z = self.tcn(x_wave)
        z = self.pool(z).squeeze(-1)
        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)
        return self.fc(z)

# 5 评估函数，计算多种指标，保存混淆矩阵
@torch.no_grad()
def evaluate(model, loader, device, n_classes=16, use_meta=True, topk=(1,3)):
    model.eval()
    y_true, y_pred = [], []
    probs_max = []
    topk_hits = {k:0 for k in topk}
    n_total = 0

    for batch in loader:
        if use_meta:
            xw, xm, y = batch
            xw, xm, y = xw.to(device), xm.to(device), y.to(device)
            logits = model(xw, xm)
        else:
            xw, y = batch
            xw, y = xw.to(device), y.to(device)
            logits = model(xw, None)

        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs_max.extend(torch.max(prob, dim=1).values.cpu().numpy().tolist())

        # top-k
        for k in topk:
            tk = torch.topk(prob, k=k, dim=1).indices
            hit = (tk == y.unsqueeze(1)).any(dim=1).sum().item()
            topk_hits[k] += hit
        n_total += y.size(0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = (np.array(y_true) == np.array(y_pred)).mean()

    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_recall = [per_class.get(str(i), {}).get("recall", 0.0) for i in range(n_classes)]

    topk_acc = {k: topk_hits[k]/max(1,n_total) for k in topk}

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "topk_acc": {str(k): float(v) for k,v in topk_acc.items()},
        "per_class_recall": per_class_recall,
        "avg_confidence": float(np.mean(probs_max)) if len(probs_max)>0 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def save_confusion_matrix(
    y_true,
    y_pred,
    out_path,
    n_classes=16,
    class_names=None,
    normalize="true",   # "true" 表示按真实类别行归一化
    figsize=(11, 9),
    dpi=220
):
    """
    更美观的混淆矩阵：
    - 颜色深浅按百分比（行归一化）显示
    - 方格内同时显示 count 和 percent
    - 支持中文类别名称显示
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm = cm.astype(np.int64)

    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.clip(row_sums, 1, None)
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_norm = cm / np.clip(col_sums, 1, None)
    else:
        total = cm.sum()
        cm_norm = cm / max(total, 1)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row Percentage", rotation=90, va="bottom")

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix (%)"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # 网格线更清晰
    ax.set_xticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 在格子中写 count + percent
    threshold = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            text = f"{count}\n{pct:.1f}%"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > threshold else "black",
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import os

def _series(history, key, default=np.nan):
    """从history(list[dict])里抽取某个key的序列，缺失则填nan"""
    out = []
    for h in history:
        v = h.get(key, default)
        out.append(v if v is not None else default)
    return np.array(out, dtype=float)

def plot_training_curves(history, out_dir, prefix="curve", show_best=True):
    """
    history: 训练过程中append的log列表
    prefix: 输出文件名前缀
    """
    if not history:
        print("[WARN] history is empty, skip plotting.")
        return

    os.makedirs(out_dir, exist_ok=True)

    epochs = _series(history, "epoch")
    # 如果epoch缺失，用索引代替
    if np.isnan(epochs).all():
        epochs = np.arange(1, len(history) + 1)

    train_loss = _series(history, "train_loss")

    val_acc = _series(history, "val_acc")
    val_macro_f1 = _series(history, "val_macro_f1")
    val_bal_acc = _series(history, "val_balanced_acc")
    best_macro_f1 = _series(history, "best_val_macro_f1")
    epoch_time = _series(history, "epoch_time")

    #  Loss curve
    if not np.isnan(train_loss).all():
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, marker="o", linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200)
        plt.close()

    #  Metrics curve 
    metrics_exist = not (np.isnan(val_acc).all() and np.isnan(val_macro_f1).all() and np.isnan(val_bal_acc).all())
    if metrics_exist:
        plt.figure(figsize=(9, 5))
        if not np.isnan(val_acc).all():
            plt.plot(epochs, val_acc, marker="o", linewidth=1, label="Val Acc")
        if not np.isnan(val_macro_f1).all():
            plt.plot(epochs, val_macro_f1, marker="o", linewidth=1, label="Val Macro-F1")
        if not np.isnan(val_bal_acc).all():
            plt.plot(epochs, val_bal_acc, marker="o", linewidth=1, label="Val Balanced-Acc")
        if show_best and (not np.isnan(best_macro_f1).all()):
            plt.plot(epochs, best_macro_f1, linestyle="--", linewidth=2, label="Best Val Macro-F1")

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_metrics.png"), dpi=200)
        plt.close()

    #  Time per epoch
    if not np.isnan(epoch_time).all():
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, epoch_time, marker="o", linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Seconds")
        plt.title("Epoch Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_time.png"), dpi=200)
        plt.close()

    print(f"[INFO] Saved curves to: {out_dir} ({prefix}_*.png)")

# 6 训练
def compute_class_weights(y_idx, n_classes=16):
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    # inverse freq (simple baseline)
    weights = (counts.sum() / (counts + 1e-9))
    weights = weights / weights.mean()
    return weights, counts

from types import SimpleNamespace
def run_experiment(cfg: dict, model_builder=None):
    # 防止json cfg漏字段
    defaults = dict(
        model="cnn",
        use_meta=False,
        meta_only = False,
        norm="per_sample",
        epochs=30,
        batch_size=256,
        lr=1e-3,
        val_ratio=0.1,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        classes="",
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2
    )

    for k,v in defaults.items():
        cfg.setdefault(k, v)

    args = SimpleNamespace(**cfg)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    (wave_tr, meta_tr, y_tr_oh), (wave_te, meta_te, y_te_oh) = load_whole_data(args.data_path)
    y_tr = onehot_to_index(y_tr_oh)
    y_te = onehot_to_index(y_te_oh)

    # 先确定要保留的旧类 id，由终端命令确定
    if args.classes.strip():
        motor_ids = [int(x.strip()) for x in args.classes.split(",") if x.strip() != ""]
        motor_ids = sorted(set(motor_ids))
    else:
        motor_ids = list(range(16))

    n_classes = len(motor_ids)
    assert n_classes >= 2, f"Need at least 2 classes, got {motor_ids}"

    # mask：只保留这些类
    mask_tr = np.isin(y_tr, motor_ids)
    mask_te = np.isin(y_te, motor_ids)

    assert mask_tr.sum() > 0, f"No train samples for classes {motor_ids}"
    assert mask_te.sum() > 0, f"No test samples for classes {motor_ids}"

    wave_tr, meta_tr, y_tr = wave_tr[mask_tr], meta_tr[mask_tr], y_tr[mask_tr]
    wave_te, meta_te, y_te = wave_te[mask_te], meta_te[mask_te], y_te[mask_te]

    # remap：旧类 id -> 0..K-1
    id_map = {old: new for new, old in enumerate(motor_ids)}
    y_tr = np.array([id_map[c] for c in y_tr], dtype=np.int64)
    y_te = np.array([id_map[c] for c in y_te], dtype=np.int64)

    # split train -> train/val
    tr_idx, va_idx = make_val_split(y_tr, val_ratio=args.val_ratio, seed=args.seed)

    # global norm stats
    global_mean, global_std = None, None
    if args.norm == "global_train":
        global_mean = wave_tr[tr_idx].mean()
        global_std = wave_tr[tr_idx].std() + 1e-8

    ds_train = WholeDataDataset(wave_tr, meta_tr, y_tr, tr_idx, use_meta=args.use_meta, meta_only=args.meta_only,  # 几个数据集都添加了meta_only指令
                               norm_mode=args.norm, global_mean=global_mean, global_std=global_std)
    ds_val = WholeDataDataset(wave_tr, meta_tr, y_tr, va_idx, use_meta=args.use_meta, meta_only=args.meta_only,
                             norm_mode=args.norm, global_mean=global_mean, global_std=global_std)
    ds_test = WholeDataDataset(wave_te, meta_te, y_te, np.arange(len(y_te)), use_meta=args.use_meta, meta_only=args.meta_only,
                               norm_mode=args.norm, global_mean=global_mean, global_std=global_std)

    # dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    # dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_workers = getattr(args, "num_workers", 0)
    pin_memory = getattr(args, "pin_memory", False)
    prefetch_factor = getattr(args, "prefetch_factor", 1)
    drop_last = getattr(args, "drop_last", False)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    class_w, counts = compute_class_weights(y_tr[tr_idx], n_classes=n_classes)
    print("\n[INFO] Class mapping & counts (train split):")
    print("new_id | old_id | ppt | ppt_name | data_name | train_count")
    print("-"*90)
    rows = []
    for old_id in motor_ids:
        new_id = id_map[old_id]
        info = LABEL_INFO.get(old_id, {"label":"?", "label_name":"UNKNOWN", "data_name": f"L{old_id}"})
        row = {
            "new_id": int(new_id),
            "old_id": int(old_id),
            "label": info["label"],
            "label_name": info["label_name"],
            "data_name": info["data_name"],
            "train_count": int(counts[new_id]),
        }
        rows.append(row)
        print(f"{row['new_id']:>5} | {row['old_id']:>6} | {row['label']:<3} | {row['label_name']:<16} | {row['data_name']:<22} | {row['train_count']}")

    # 保存在输出文件中，各类统计
    with open(os.path.join(args.out_dir, "class_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


    device = torch.device(args.device)

    model = model_builder(
        cfg.get("model", "cnn"),
        n_classes=n_classes,
        use_meta=cfg.get("use_meta", False),
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

    print(f"[INFO] Model device: {next(model.parameters()).device}")
    # 验张量是否会被移到GPU
    test_tensor = torch.tensor([1.0])
    test_tensor = test_tensor.to(device)
    print(f"[INFO] Test tensor device: {test_tensor.device}")

    # weighted CE baseline
    w_tensor = torch.tensor(class_w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        t0 = time.perf_counter()

        for batch in dl_train:
            opt.zero_grad()
            if args.use_meta:
                xw, xm, y = batch
                xw, xm, y = xw.to(device), xm.to(device), y.to(device)
                logits = model(xw, xm)
            else:
                xw, y = batch
                xw, y = xw.to(device), y.to(device)
                logits = model(xw, None)

            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = loss_sum / max(1, n)

        val_metrics = evaluate(model, dl_val, device, n_classes=n_classes, use_meta=args.use_meta)
        # 以 macro_f1 作为选择基准
        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

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
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_training_curves(history, args.out_dir, prefix="curve")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

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

    print("\n[TEST] acc=", test_metrics["acc"],
          "macro_f1=", test_metrics["macro_f1"],
          "balanced_acc=", test_metrics["balanced_acc"],
          "topk=", test_metrics["topk_acc"])
    return history, {k:v for k,v in test_metrics.items() if k not in ["y_true","y_pred"]}

def parse_train_class_counts(train_class_counts, id_map=None, use_old_ids=True):
    """
    train_class_counts: dict, 例如 {"12": 400, "13": 120, "9": 25, "11": 20}
    返回 remap 后的新类 id -> count
    """
    if not train_class_counts:
        return {}

    parsed = {}
    for k, v in train_class_counts.items():
        cls_id = int(k)
        cnt = int(v)
        if cnt <= 0:
            continue

        if use_old_ids:
            if id_map is None or cls_id not in id_map:
                continue
            cls_id = id_map[cls_id]

        parsed[cls_id] = cnt
    return parsed


def subsample_train_indices_by_class(y, train_idx, target_counts, seed=42, strict=False):
    """
    y: remap 后标签
    train_idx: 原始训练子集索引
    target_counts: {new_class_id: target_count}
    strict=False 时，若目标数量超过可用数量，则自动截断并告警
    """
    if not target_counts:
        return train_idx

    rng = np.random.RandomState(seed)
    train_idx = np.asarray(train_idx)
    final_idx = []

    classes_in_train = np.unique(y[train_idx])
    for c in classes_in_train:
        idx_c = train_idx[y[train_idx] == c]
        rng.shuffle(idx_c)

        target = target_counts.get(int(c), len(idx_c))

        if target > len(idx_c):
            if strict:
                raise ValueError(
                    f"class {c}: requested {target}, but only {len(idx_c)} available in train split"
                )
            print(f"[WARN] class {c}: requested {target}, clipped to {len(idx_c)}")
            target = len(idx_c)

        final_idx.extend(idx_c[:target].tolist())

    rng.shuffle(final_idx)
    return np.array(final_idx, dtype=np.int64)

def build_class_count_records(motor_ids, id_map, counts, split_name="train"):
    """
    motor_ids: 保留的 old class id 列表（已排序）
    id_map: old_id -> new_id
    counts: remap 后的新类计数数组，如 [1164, 715, ...]
    """
    counts = np.asarray(counts).astype(int)
    records = []

    for old_id in motor_ids:
        new_id = id_map[old_id]
        info = LABEL_INFO.get(old_id, {
            "label": f"C{new_id}",
            "label_name": "UNKNOWN",
            "data_name": f"L{old_id}"
        })

        records.append({
            "new_id": int(new_id),
            "old_id": int(old_id),
            "raw_code": f"L{old_id}",
            "ppt_label": info["label"],
            "fault_name": info["label_name"],
            "data_name": info["data_name"],
            f"{split_name}_count": int(counts[new_id])
        })
    return records

def build_confmat_class_names(motor_ids, id_map):
    """
    返回按 new_id 顺序排列的类别显示名
    例如:
    ['L7\n电机偏心', 'L9\n双侧轴承保持架故障', ...]
    """
    class_names = [None] * len(motor_ids)
    for old_id in motor_ids:
        new_id = id_map[old_id]
        info = LABEL_INFO.get(old_id, {
            "label_name": f"class_{new_id}",
            "data_name": f"L{old_id}"
        })
        class_names[new_id] = f"L{old_id}\n{info['label_name']}"
    return class_names

def save_class_count_records(out_path, motor_ids, id_map, counts, split_name="train"):
    records = build_class_count_records(
        motor_ids=motor_ids,
        id_map=id_map,
        counts=counts,
        split_name=split_name
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def build_per_class_recall_records(per_class_recall, motor_ids, id_map):
    records = []
    for old_id in motor_ids:
        new_id = id_map[old_id]
        info = LABEL_INFO.get(old_id, {
            "label": f"C{new_id}",
            "label_name": "UNKNOWN",
            "data_name": f"L{old_id}"
        })
        records.append({
            "new_id": int(new_id),
            "old_id": int(old_id),
            "raw_code": f"L{old_id}",
            "ppt_label": info["label"],
            "fault_name": info["label_name"],
            "data_name": info["data_name"],
            "recall": float(per_class_recall[new_id])
        })
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="whole_data.npy")
    ap.add_argument("--out_dir", type=str, default="runs/baseline")
    ap.add_argument("--model", type=str, choices=["cnn","tcn"], default="cnn")
    ap.add_argument("--use_meta", action="store_true")  # 是否拼接sensor+condition(33维)
    ap.add_argument("--norm", type=str, choices=["per_sample","global_train","none"], default="per_sample")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--classes", type=str, default="", help="comma-separated old class ids to keep, e.g. '3,7,12'")
    args = ap.parse_args()
    run_experiment(vars(args))
    

if __name__ == "__main__":
    main()