import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# utils
# -------------------------
def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 有些文件可能是 utf-8-sig 或写入不完整
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read json: {p} ({e})")
            return None


def flatten_metrics(test_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """把 test_metrics.json 展平成一行可放表格的字段"""
    out = {}
    if not test_metrics:
        return out

    for k in ["acc", "macro_f1", "balanced_acc", "avg_confidence"]:
        if k in test_metrics:
            out[k] = test_metrics[k]

    # topk_acc: {"1":..., "3":...}
    topk = test_metrics.get("topk_acc", {})
    if isinstance(topk, dict):
        for kk, vv in topk.items():
            out[f"topk@{kk}"] = vv

    # per_class_recall: list
    recalls = test_metrics.get("per_class_recall", None)
    if isinstance(recalls, list):
        for i, r in enumerate(recalls):
            out[f"recall_{i}"] = r
        out["recall_min"] = float(np.min(recalls)) if len(recalls) else np.nan
        out["recall_mean"] = float(np.mean(recalls)) if len(recalls) else np.nan

    return out


def parse_run_dir(run_dir: Path) -> Optional[Dict[str, Any]]:
    """从一个 run 目录读取 cfg/test/time/history 并拼成一行"""
    cfg = safe_read_json(run_dir / "cfg_used.json")
    test = safe_read_json(run_dir / "test_metrics.json")
    timej = safe_read_json(run_dir / "time.json")
    hist = safe_read_json(run_dir / "history.json")

    # 必须至少有 test_metrics 才算有效实验
    if test is None:
        return None

    row = {}
    row["run_dir"] = str(run_dir)

    # cfg
    if cfg:
        # 常用字段直接放出来（你可以按需增减）
        for k in [
            "model", "use_meta", "meta_only", "norm", "epochs", "batch_size",
            "lr", "val_ratio", "seed", "device", "classes",
            "num_workers", "pin_memory", "prefetch_factor", "drop_last",
            "data_path"
        ]:
            if k in cfg:
                row[k] = cfg[k]

    # metrics
    row.update(flatten_metrics(test))

    # time
    if timej and "total_time_sec" in timej:
        row["total_time_sec"] = timej["total_time_sec"]
    else:
        row["total_time_sec"] = np.nan

    # history（可选：拿 best_val_macro_f1 / 最后 epoch_time）
    if isinstance(hist, list) and len(hist) > 0:
        # history 每条是 dict
        last = hist[-1]
        row["last_epoch_time"] = last.get("epoch_time", np.nan)

        # best_val_macro_f1 可能在每条 log 里
        bests = [h.get("best_val_macro_f1") for h in hist if isinstance(h, dict) and "best_val_macro_f1" in h]
        row["best_val_macro_f1"] = float(np.max(bests)) if len(bests) else np.nan
    else:
        row["last_epoch_time"] = np.nan
        row["best_val_macro_f1"] = np.nan

    return row


def scan_runs(runs_dir: Path) -> pd.DataFrame:
    """遍历 runs 下的子目录，收集所有包含 test_metrics.json 的 run"""
    rows: List[Dict[str, Any]] = []
    for sub in sorted(runs_dir.glob("*")):
        if not sub.is_dir():
            continue
        row = parse_run_dir(sub)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # 统一一些类型
    for col in ["use_meta", "meta_only", "pin_memory", "drop_last"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # 把 classes 统一成字符串（避免有的 cfg 是 list）
    if "classes" in df.columns:
        df["classes"] = df["classes"].astype(str)

    return df


def make_group_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str]
) -> pd.DataFrame:
    """按 group_cols 聚合出 mean/std/count"""
    keep_cols = [c for c in group_cols if c in df.columns]
    met_cols = [c for c in metric_cols if c in df.columns]

    if len(keep_cols) == 0:
        raise ValueError("No valid group_cols found in df.")
    if len(met_cols) == 0:
        raise ValueError("No valid metric_cols found in df.")

    agg_dict = {c: ["mean", "std"] for c in met_cols}
    g = df.groupby(keep_cols, dropna=False).agg(agg_dict)
    g.columns = [f"{m}_{stat}" for (m, stat) in g.columns]
    g = g.reset_index()
    g["n_runs"] = df.groupby(keep_cols, dropna=False).size().values
    return g


def plot_bar(df_group: pd.DataFrame, x_col: str, y_col: str, out_png: Path, title: str):
    """简单柱状图"""
    df_plot = df_group.sort_values(y_col, ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(df_plot)), df_plot[y_col].values)
    plt.xticks(range(len(df_plot)), df_plot[x_col].values, rotation=45, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str):
    """散点图：例如 macro_f1 vs time"""
    if x not in df.columns or y not in df.columns:
        return
    tmp = df[[x, y]].dropna()
    if len(tmp) == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(tmp[x].values, tmp[y].values, s=20)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="runs folder path")
    ap.add_argument("--out_dir", type=str, default="runs_summary", help="output folder")
    ap.add_argument("--save_excel", action="store_true", help="also save xlsx")
    ap.add_argument("--topn", type=int, default=20, help="top-N groups to plot (by macro_f1_mean)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_runs(runs_dir)
    if len(df) == 0:
        print("[ERROR] No runs found (no test_metrics.json).")
        return

    # 明细表输出
    df_detail = df.sort_values(["model", "classes", "use_meta", "meta_only", "seed"], na_position="last")
    df_detail.to_csv(out_dir / "runs_detail.csv", index=False, encoding="utf-8-sig")

    # 你通常关心的指标列（可自行加）
    metric_cols = [
        "acc", "macro_f1", "balanced_acc", "avg_confidence",
        "topk@1", "topk@3", "recall_min", "recall_mean",
        "total_time_sec"
    ]
    # 分组字段（用于“同一配置不同seed”求均值方差）
    group_cols = ["model", "classes", "use_meta", "meta_only", "epochs", "norm", "lr", "batch_size"]

    df_group = make_group_summary(df_detail, group_cols=group_cols, metric_cols=metric_cols)
    df_group = df_group.sort_values("macro_f1_mean", ascending=False)
    df_group.to_csv(out_dir / "runs_group_summary.csv", index=False, encoding="utf-8-sig")

    # 额外：挑出每个 group 的 best run（按 macro_f1）
    best_rows = []
    valid_group_cols = [c for c in group_cols if c in df_detail.columns]
    for _, g in df_detail.groupby(valid_group_cols, dropna=False):
        # 以 macro_f1 为主（没有就用 acc）
        key = "macro_f1" if "macro_f1" in g.columns else "acc"
        best = g.sort_values(key, ascending=False).iloc[0].to_dict()
        best_rows.append(best)
    df_best = pd.DataFrame(best_rows).sort_values("macro_f1", ascending=False)
    df_best.to_csv(out_dir / "runs_best_per_group.csv", index=False, encoding="utf-8-sig")

    # Excel 输出（可选）
    if args.save_excel:
        xlsx_path = out_dir / "runs_summary.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df_detail.to_excel(w, sheet_name="detail", index=False)
            df_group.to_excel(w, sheet_name="group_mean_std", index=False)
            df_best.to_excel(w, sheet_name="best_per_group", index=False)

    # 绘图：按 group 的 macro_f1_mean 画 topN
    topn = min(args.topn, len(df_group))
    df_plot = df_group.head(topn).copy()

    # 构造 x 轴标签（简洁展示）
    df_plot["label"] = (
        df_plot["model"].astype(str)
        + " | meta=" + df_plot.get("use_meta", False).astype(str)
        + " | classes=" + df_plot["classes"].astype(str)
    )

    plot_bar(
        df_plot,
        x_col="label",
        y_col="macro_f1_mean",
        out_png=out_dir / "bar_macro_f1_mean.png",
        title=f"Top{topn} groups by macro_f1_mean"
    )

    # 散点：macro_f1 vs total_time_sec（逐run）
    plot_scatter(
        df_detail,
        x="total_time_sec",
        y="macro_f1",
        out_png=out_dir / "scatter_time_vs_macro_f1.png",
        title="macro_f1 vs total_time_sec (per run)"
    )

    print(f"[DONE] Wrote to: {out_dir}")
    print(f" - {out_dir/'runs_detail.csv'}")
    print(f" - {out_dir/'runs_group_summary.csv'}")
    print(f" - {out_dir/'runs_best_per_group.csv'}")
    if args.save_excel:
        print(f" - {out_dir/'runs_summary.xlsx'}")
    print(f" - plots: {out_dir/'bar_macro_f1_mean.png'}, {out_dir/'scatter_time_vs_macro_f1.png'}")


if __name__ == "__main__":
    main()