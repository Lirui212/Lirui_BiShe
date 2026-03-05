由全data中筛选电机端故障标签

`7, 9, 10, 11, 12, 13`

| PPT标签 | PPT名称              | 数据标签                                                    |
| ------- | -------------------- | ----------------------------------------------------------- |
| C0      | 无故障               | L12 正常运行                                                |
| C1      | 驱动端轴承保持架故障 | L13 轴承支架破坏|
| C2      | 非驱动端轴承外圈故障 | L10 西边轴承滚珠外圈磨损                                    |
| C3      | 双侧轴承保持架故障   | L9 双轴承支架破坏                                           |
| C4      | 双侧轴承外圈故障     | L11 双轴承滚珠外圈破坏性磨损                                |
| C5      | 电机偏心             | L7 电机偏心                                                 |

## 实验

```
python baseline_train.py `
  --data_path whole_data.npy `
  --out_dir runs/motor6_cnn_meta_base `
  --model cnn `
  --use_meta `
  --classes 7,9,10,11,12,13 `
  --norm per_sample `
  --epochs 30 `
  --batch_size 256 `
  --lr 1e-3 `
  --val_ratio 0.1 `
  --seed 42
```

```
python baseline_train.py `
  --data_path whole_data.npy `
  --out_dir runs/motor6_cnn_nometa_base `
  --model cnn `
  --classes 7,9,10,11,12,13 `
  --norm per_sample `
  --epochs 30 `
  --batch_size 256 `
  --lr 1e-3 `
  --val_ratio 0.1 `
  --seed 42
```

```
python baseline_train.py `
  --data_path whole_data.npy `
  --out_dir runs/motor6_tcn_nometa_base `
  --model tcn `
  --classes 7,9,10,11,12,13 `
  --norm per_sample `
  --epochs 30 `
  --batch_size 256 `
  --lr 1e-3 `
  --val_ratio 0.1 `
  --seed 42
```



### 改进了下整体框架

变为baseline——train_opt——model_opt的调用形式

```
project/
├─ baseline_train.py          # Baseline训练入口：数据加载/训练评估
├─ train_opt.py               # cfg驱动实验入口：run_dir生成、计时、调用run_experiment
├─ models_opt.py              # 特征层优化模型：MultiScaleConv1D、CNN1D_MS等
├─ data.py                    # 数据说明/分层切片/统计特征与箱线图
├─ configs/                   # 实验配置（json）
│  ├─ motor6_cnn_baseline.json
│  ├─ motor6_tcn_baseline.json
│  └─ ...
└─ runs/                      # 实验输出
   └─ <exp_name_yyyymmdd_hhmmss>/
      ├─ cfg_used.json
      ├─ history.json
      ├─ test_metrics.json
      ├─ class_counts.json
      ├─ confusion_matrix.png
      ├─ best.pt
      └─ time.json
```





## CNN&TCN架构

### 1D-CNN（Baseline）

**Backbone**：4×Conv1D 下采样

- `Conv1d(1→32, k=9, s=2, p=4) + BN + ReLU`
- `Conv1d(32→64, k=9, s=2, p=4) + BN + ReLU`
- `Conv1d(64→128, k=9, s=2, p=4) + BN + ReLU`
- `Conv1d(128→256, k=9, s=2, p=4) + BN + ReLU`

**Pooling**：`AdaptiveAvgPool1d(1)` → `feat ∈ R^{B×256}`

**Head**：

- meta off：`Linear(256→256) + ReLU + Dropout(0.2) + Linear(256→K)`
- meta on：`Concat(feat, meta)` → `Linear(256+33→256) ... → K`

### TCN（Baseline）

**Backbone**：4×TCNBlock（残差膨胀卷积）

- Channels：`(16,32,64,128)`
- Kernel：`k=5`
- Dilation：`(1,2,4,8)`
- 每块：`[Conv+BN+ReLU+Dropout] ×2 + Residual`

**Pooling**：`AdaptiveAvgPool1d(1)` → `feat ∈ R^{B×128}`

**Head**：

- meta off：`Linear(128→128) + ReLU + Dropout(0.2) + Linear(128→K)`
- meta on：`Concat(feat, meta)` → `Linear(128+33→128) ... → K`

## 检验是否有传感器-工况 to 故障的识别捷径

设置meta_only训练

无波形，看是否直接从传感器加工况学习了故障导致准度虚高



## 指标

Accuracy

Marcro-F1

Balanced Accuracy

Per-class recall

Top-k

Avg-condition







1. **数据读取与样本拆分完成**

- `whole_data.npy` 读取成功，样本固定维度 `2081`
- 拆分为：`wave(2048)` + `sensor(15)` + `condition(18)`，可拼接为 `meta(33)`

1. **标签处理与数据集构建完成**

- 16维 one-hot 标签转为类别 id（0~15）
- 按类别做 train/val 分层切分（避免 val 缺类）
- 波形归一化支持 `per_sample / global_train / none`

1. **基线模型与训练/评测闭环完成**

- 支持 `CNN1D / TCN` 两类主干
- 使用类别权重（逆频率）CrossEntropy 作为不平衡 baseline
- 输出 `history.json / best.pt / test_metrics.json / confusion_matrix.png`
- 已跑通并取得强基线结果（Test Acc≈0.946, Macro-F1≈0.944, BalAcc≈0.952）







| `acc`：易受多数类影响 |                                                   |
| --------------------- | ------------------------------------------------- |
| `macro_f1`）          | 更关注小类表现，适合不均衡                        |
| `balanced_acc`        | 各类 recall 的平均                                |
| `per_class_recall`：  | 逐类召回，方便定位“漏检哪一类”                    |
| `topk_acc@3`：        | Top-3 命中率 |
| `avg_confidence`：    | 预测最大概率均值 |

