import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time

# -------------------------
# 0) 复现性
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# 1) 数据--wholedata.py 已经预处理好并保存为 npy，加载并拆分
# -------------------------
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

# -------------------------
# 2) Train/Val split 针对类别不平衡
# -------------------------
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

# -------------------------
# 3) 数据集类，归一化
# -------------------------
class WholeDataDataset(Dataset):
    def __init__(self, wave, meta, y_idx, indices,
                 use_meta=True,
                 norm_mode="per_sample",
                 global_mean=None,
                 global_std=None):
        self.wave = wave[indices].astype(np.float32)
        self.meta = meta[indices].astype(np.float32)
        self.y = y_idx[indices].astype(np.int64)
        self.use_meta = use_meta
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
            return xw, xm, y
        else:
            return xw, y

# -------------------------
# 4) CNN&TCN模型，简单baseline
# -------------------------
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

    def forward(self, x_wave, x_meta=None):
        z = self.backbone(x_wave)         # (B,256,T)
        z = self.pool(z).squeeze(-1)      # (B,256)
        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)
        logits = self.head(z)
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

    def forward(self, x):
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

    def forward(self, x_wave, x_meta=None):
        z = self.tcn(x_wave)
        z = self.pool(z).squeeze(-1)
        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)
        return self.fc(z)

# -------------------------
# 5) 评估函数，计算多种指标，保存混淆矩阵
# -------------------------
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

def save_confusion_matrix(y_true, y_pred, out_path, n_classes=16):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------
# 6) 训练
# -------------------------
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
        prefetch_factor=2,
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

    ds_train = WholeDataDataset(wave_tr, meta_tr, y_tr, tr_idx, use_meta=args.use_meta,
                               norm_mode=args.norm, global_mean=global_mean, global_std=global_std)
    ds_val = WholeDataDataset(wave_tr, meta_tr, y_tr, va_idx, use_meta=args.use_meta,
                             norm_mode=args.norm, global_mean=global_mean, global_std=global_std)
    ds_test = WholeDataDataset(wave_te, meta_te, y_te, np.arange(len(y_te)), use_meta=args.use_meta,
                               norm_mode=args.norm, global_mean=global_mean, global_std=global_std)

    # dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    # dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_workers = getattr(args, "num_workers", 4)
    pin_memory = getattr(args, "pin_memory", True)
    prefetch_factor = getattr(args, "prefetch_factor", 2)
    drop_last = getattr(args, "drop_last", True)

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
    print("[INFO] Train class counts:", counts.tolist())

    with open(os.path.join(args.out_dir, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts.tolist(), "weights": class_w.tolist()}, f, ensure_ascii=False, indent=2)

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

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, dl_test, device, n_classes=n_classes, use_meta=args.use_meta)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in test_metrics.items() if k not in ["y_true","y_pred"]},
                  f, ensure_ascii=False, indent=2)

    save_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"],
                          os.path.join(args.out_dir, "confusion_matrix.png"),
                          n_classes=n_classes)

    print("\n[TEST] acc=", test_metrics["acc"],
          "macro_f1=", test_metrics["macro_f1"],
          "balanced_acc=", test_metrics["balanced_acc"],
          "topk=", test_metrics["topk_acc"])
    return history, {k:v for k,v in test_metrics.items() if k not in ["y_true","y_pred"]}

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