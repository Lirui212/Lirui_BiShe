# analyze_meta_bias.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def onehot_to_index(y_onehot: np.ndarray):
    return np.argmax(y_onehot, axis=1).astype(np.int64)

def entropy(p: np.ndarray, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def main():
    npy_path = "whole_data.npy"
    out_dir = "analysis_meta_bias"
    os.makedirs(out_dir, exist_ok=True)

    motor_ids = [7, 9, 10, 11, 12, 13]
    # 你PPT的标签顺序（可选，用于表格行名）
    class_names = ["C0 Normal(L12)", "C1 L13", "C2 L10", "C3 L9", "C4 L11", "C5 L7"]

    data = np.load(npy_path, allow_pickle=True).item()

    x_train = np.array(data["train_sample"]).reshape(-1, 2081)
    y_train = np.array(data["train_label"])
    x_test  = np.array(data["test_sample"]).reshape(-1, 2081)
    y_test  = np.array(data["test_label"])

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    y_idx = onehot_to_index(y)

    # split meta
    sensor_oh = x[:, 2048:2063]     # (N,15)
    cond_oh   = x[:, 2063:2081]     # (N,18)

    sensor_id = np.argmax(sensor_oh, axis=1)  # 0..14
    cond_id   = np.argmax(cond_oh, axis=1)    # 0..17

    # filter motor classes
    mask = np.isin(y_idx, motor_ids)
    y_f = y_idx[mask]
    sensor_f = sensor_id[mask]
    cond_f = cond_id[mask]

    # remap to 0..5
    id_map = {old:i for i,old in enumerate(motor_ids)}
    y_r = np.array([id_map[c] for c in y_f], dtype=np.int64)

    # crosstab: class x sensor
    df = pd.DataFrame({"cls": y_r, "sensor": sensor_f, "cond": cond_f})

    ct_sensor = pd.crosstab(df["cls"], df["sensor"])
    prob_sensor = ct_sensor.div(ct_sensor.sum(axis=1), axis=0)

    ct_cond = pd.crosstab(df["cls"], df["cond"])
    prob_cond = ct_cond.div(ct_cond.sum(axis=1), axis=0)

    # add row names
    ct_sensor.index = class_names
    prob_sensor.index = class_names
    ct_cond.index = class_names
    prob_cond.index = class_names

    # save tables
    ct_sensor.to_csv(os.path.join(out_dir, "count_table_sensor.csv"), encoding="utf-8-sig")
    prob_sensor.to_csv(os.path.join(out_dir, "prob_table_sensor.csv"), encoding="utf-8-sig")

    ct_cond.to_csv(os.path.join(out_dir, "count_table_condition.csv"), encoding="utf-8-sig")
    prob_cond.to_csv(os.path.join(out_dir, "prob_table_condition.csv"), encoding="utf-8-sig")

    # concentration summary
    summary = []
    for i, name in enumerate(class_names):
        ps = prob_sensor.loc[name].fillna(0).values.astype(float)
        pc = prob_cond.loc[name].fillna(0).values.astype(float)
        summary.append({
            "class": name,
            "sensor_max_prob": float(ps.max()),
            "sensor_entropy": entropy(ps),
            "cond_max_prob": float(pc.max()),
            "cond_entropy": entropy(pc),
            "n_samples": int(ct_sensor.loc[name].sum())
        })
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "summary_bias.csv"), index=False, encoding="utf-8-sig")

    # heatmaps (simple matplotlib)
    def plot_heatmap(mat: pd.DataFrame, title: str, out_png: str):
        plt.figure(figsize=(12, 4))
        plt.imshow(mat.values, aspect="auto")
        plt.title(title)
        plt.yticks(range(mat.shape[0]), mat.index)
        plt.xticks(range(mat.shape[1]), mat.columns, rotation=90)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, out_png), dpi=200)
        plt.close()

    plot_heatmap(prob_sensor, "P(sensor | class) heatmap", "heatmap_sensor.png")
    plot_heatmap(prob_cond, "P(condition | class) heatmap", "heatmap_condition.png")

    print("[DONE]", out_dir)

if __name__ == "__main__":
    main()