import numpy as np
import matplotlib.pyplot as plt
import os
import random

# =========================
# 1. 基本设置
# =========================
NPY_PATH = "whole_data.npy"      # 改成你的文件路径
SAVE_PATH = "motor_waveforms_raw.png"
RANDOM_SEED = 42

# 是否从 train+test 全部样本里选
USE_ALL_DATA = True

# 是否每类随机抽1条；False时取该类第一条
RANDOM_PICK = True

# 是否对每条波形做标准化后再画
# True: 更适合比较波形形态
# False: 保留原始幅值
NORMALIZE_EACH_WAVE = True

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# 2. 电机端故障标签映射
# 原始标签 -> 中文名称
# =========================
motor_label_map = {
    7:  "电机偏心",
    9:  "双侧轴承保持架故障",
    10: "非驱动端轴承外圈故障",
    11: "双侧轴承外圈故障",
    12: "无故障",
    13: "驱动端轴承保持架故障",
}

target_labels = list(motor_label_map.keys())


# =========================
# 3. 读取数据
# =========================
def load_whole_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()

    x_train = np.array(data["train_sample"]).reshape(-1, 2081)
    y_train = np.array(data["train_label"])

    x_test = np.array(data["test_sample"]).reshape(-1, 2081)
    y_test = np.array(data["test_label"])

    if USE_ALL_DATA:
        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
    else:
        x = x_train
        y = y_train

    # 前2048维是振动波形
    wave = x[:, :2048]

    # one-hot -> 类别索引
    y_idx = np.argmax(y, axis=1).astype(int)

    return wave, y_idx


# =========================
# 4. 每类挑选一个样本
# =========================
def pick_one_sample_per_class(wave, y_idx, target_labels):
    picked = {}

    for cls in target_labels:
        idxs = np.where(y_idx == cls)[0]
        if len(idxs) == 0:
            print(f"[WARN] 类别 {cls} 没有样本")
            continue

        pick_idx = random.choice(idxs) if RANDOM_PICK else idxs[0]
        picked[cls] = {
            "sample_index": int(pick_idx),
            "wave": wave[pick_idx].copy()
        }

    return picked


# =========================
# 5. 波形预处理（可选）
# =========================
def process_wave(x):
    if NORMALIZE_EACH_WAVE:
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std
    return x


# =========================
# 6. 绘图
# =========================
def plot_motor_waveforms(picked_dict, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for ax, cls in zip(axes, target_labels):
        if cls not in picked_dict:
            ax.set_visible(False)
            continue

        x = process_wave(picked_dict[cls]["wave"])
        idx = picked_dict[cls]["sample_index"]
        title = f"L{cls} - {motor_label_map[cls]}\n样本索引: {idx}"

        ax.plot(x, linewidth=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("采样点")
        ax.set_ylabel("幅值")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "电机端不同故障类别原始振动波形示意"
        + ("（单样本标准化后）" if NORMALIZE_EACH_WAVE else "（原始幅值）"),
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[INFO] 图片已保存到: {os.path.abspath(save_path)}")


# =========================
# 7. 主程序
# =========================
if __name__ == "__main__":
    wave, y_idx = load_whole_data(NPY_PATH)
    picked = pick_one_sample_per_class(wave, y_idx, target_labels)
    plot_motor_waveforms(picked, SAVE_PATH)