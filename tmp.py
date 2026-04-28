import argparse
import json
import hashlib
import numpy as np


def onehot_to_index(y_onehot):
    sums = y_onehot.sum(axis=1)
    bad = np.where(sums != 1)[0]
    y_idx = np.argmax(y_onehot, axis=1).astype(np.int64)
    return y_idx, bad


def make_val_split(y, val_ratio=0.1, seed=42):
    """
    与你当前训练代码一致：
    每类取 max(1, round(val_ratio * n_c)) 个到 val，
    但保证 train 至少留 1 个；若该类只有 1 个样本，则全留 train。
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
            train_idx.extend(idx_c.tolist())
            continue
        n_val = max(1, int(round(val_ratio * n_c)))
        n_val = min(n_val, n_c - 1)
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


def describe_array(name, arr):
    print(f"\n[{name}]")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        print(f"  min/max: {arr.min()} / {arr.max()}")


def split_x(x):
    wave = x[:, :2048]
    sensor = x[:, 2048:2063]
    condition = x[:, 2063:2081]
    return wave, sensor, condition


def count_by_class(y_idx, n_classes=None):
    if n_classes is None:
        n_classes = int(y_idx.max()) + 1
    return np.bincount(y_idx, minlength=n_classes)


def hash_rows(arr):
    """
    对每一行做哈希，用于检查 train/test 是否有完全重复样本
    """
    hashes = []
    arr_c = np.ascontiguousarray(arr)
    for row in arr_c:
        h = hashlib.md5(row.tobytes()).hexdigest()
        hashes.append(h)
    return hashes


def inspect_whole_data(npy_path, val_ratio=0.1, seed=42):
    print(f"Loading: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()

    print("\n===== 1) 文件顶层结构 =====")
    print("keys:", list(data.keys()))

    required_keys = ["train_sample", "train_label", "test_sample", "test_label"]
    for k in required_keys:
        print(f"  {k}: {'FOUND' if k in data else 'MISSING'}")

    x_train = np.array(data["train_sample"]).reshape(-1, 2081)
    y_train = np.array(data["train_label"])
    x_test = np.array(data["test_sample"]).reshape(-1, 2081)
    y_test = np.array(data["test_label"])

    print("\n===== 2) train/test 基本信息 =====")
    describe_array("x_train", x_train)
    describe_array("y_train", y_train)
    describe_array("x_test", x_test)
    describe_array("y_test", y_test)

    print("\n===== 3) 样本结构检查 =====")
    wave_tr, sensor_tr, cond_tr = split_x(x_train)
    wave_te, sensor_te, cond_te = split_x(x_test)

    print("train:")
    print("  wave shape     :", wave_tr.shape)
    print("  sensor shape   :", sensor_tr.shape)
    print("  condition shape:", cond_tr.shape)

    print("test:")
    print("  wave shape     :", wave_te.shape)
    print("  sensor shape   :", sensor_te.shape)
    print("  condition shape:", cond_te.shape)

    print("\n===== 4) 标签 one-hot 检查 =====")
    y_train_idx, bad_train = onehot_to_index(y_train)
    y_test_idx, bad_test = onehot_to_index(y_test)

    print(f"train bad one-hot count: {len(bad_train)}")
    if len(bad_train) > 0:
        print("  example bad indices:", bad_train[:10].tolist())

    print(f"test bad one-hot count: {len(bad_test)}")
    if len(bad_test) > 0:
        print("  example bad indices:", bad_test[:10].tolist())

    n_classes = max(y_train.shape[1], y_test.shape[1])
    print(f"num_classes inferred from label dim: {n_classes}")

    print("\n===== 5) 类别分布 =====")
    train_counts = count_by_class(y_train_idx, n_classes=n_classes)
    test_counts = count_by_class(y_test_idx, n_classes=n_classes)

    print("train class counts:")
    for i, c in enumerate(train_counts):
        print(f"  class {i:2d}: {int(c)}")

    print("test class counts:")
    for i, c in enumerate(test_counts):
        print(f"  class {i:2d}: {int(c)}")

    print("\n===== 6) 传感器/工况编码检查 =====")
    unique_sensor_train = np.unique(sensor_tr, axis=0)
    unique_sensor_test = np.unique(sensor_te, axis=0)
    unique_sensor_all = np.unique(np.vstack([sensor_tr, sensor_te]), axis=0)

    unique_cond_train = np.unique(cond_tr, axis=0)
    unique_cond_test = np.unique(cond_te, axis=0)
    unique_cond_all = np.unique(np.vstack([cond_tr, cond_te]), axis=0)

    print(f"unique sensors  - train/test/all: {len(unique_sensor_train)} / {len(unique_sensor_test)} / {len(unique_sensor_all)}")
    print(f"unique conditions - train/test/all: {len(unique_cond_train)} / {len(unique_cond_test)} / {len(unique_cond_all)}")

    sensor_onehot_ok_tr = np.all(sensor_tr.sum(axis=1) == 1)
    sensor_onehot_ok_te = np.all(sensor_te.sum(axis=1) == 1)
    cond_onehot_ok_tr = np.all(cond_tr.sum(axis=1) == 2)  # 压力13维 one-hot + 频率5维 one-hot
    cond_onehot_ok_te = np.all(cond_te.sum(axis=1) == 2)

    print(f"sensor one-hot valid - train/test: {sensor_onehot_ok_tr} / {sensor_onehot_ok_te}")
    print(f"condition code valid(sum==2) - train/test: {cond_onehot_ok_tr} / {cond_onehot_ok_te}")

    print("\n===== 7) train/test 是否有完全重复样本 =====")
    # 用完整 x + y 检查更严格
    train_xy = np.concatenate([x_train, y_train], axis=1)
    test_xy = np.concatenate([x_test, y_test], axis=1)

    train_hash = set(hash_rows(train_xy))
    test_hash = set(hash_rows(test_xy))
    overlap = train_hash.intersection(test_hash)

    print(f"exact duplicate rows between train and test: {len(overlap)}")

    print("\n===== 8) 模拟当前训练脚本中的 val 划分 =====")
    tr_idx, va_idx = make_val_split(y_train_idx, val_ratio=val_ratio, seed=seed)
    train_after_split = count_by_class(y_train_idx[tr_idx], n_classes=n_classes)
    val_after_split = count_by_class(y_train_idx[va_idx], n_classes=n_classes)

    print(f"val_ratio={val_ratio}, seed={seed}")
    print(f"train subset size after split: {len(tr_idx)}")
    print(f"val subset size after split  : {len(va_idx)}")

    print("\nclass distribution after simulated train/val split:")
    for i in range(n_classes):
        print(
            f"  class {i:2d}: "
            f"train={int(train_after_split[i])}, "
            f"val={int(val_after_split[i])}, "
            f"orig_train={int(train_counts[i])}"
        )

    print("\n===== 9) 保存简要报告 =====")
    report = {
        "keys": list(data.keys()),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
        "bad_train_onehot_count": int(len(bad_train)),
        "bad_test_onehot_count": int(len(bad_test)),
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": test_counts.tolist(),
        "unique_sensors_train": int(len(unique_sensor_train)),
        "unique_sensors_test": int(len(unique_sensor_test)),
        "unique_sensors_all": int(len(unique_sensor_all)),
        "unique_conditions_train": int(len(unique_cond_train)),
        "unique_conditions_test": int(len(unique_cond_test)),
        "unique_conditions_all": int(len(unique_cond_all)),
        "sensor_onehot_ok_train": bool(sensor_onehot_ok_tr),
        "sensor_onehot_ok_test": bool(sensor_onehot_ok_te),
        "condition_code_ok_train": bool(cond_onehot_ok_tr),
        "condition_code_ok_test": bool(cond_onehot_ok_te),
        "exact_duplicate_rows_between_train_test": int(len(overlap)),
        "simulated_val_ratio": float(val_ratio),
        "simulated_train_counts_after_split": train_after_split.tolist(),
        "simulated_val_counts_after_split": val_after_split.tolist(),
    }

    out_json = "whole_data_inspect_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"saved report to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="whole_data.npy")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inspect_whole_data(
        npy_path=args.data_path,
        val_ratio=args.val_ratio,
        seed=args.seed
    )