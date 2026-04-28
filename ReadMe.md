$$
f_{ms}(x) = \phi\left(\mathrm{Conv}_{1\times1}\Big(\big[f_1(x), f_2(x), f_3(x)\big]\Big)\right)
$$

whole_data.npy
    ↓
数据读取与标签筛选
（筛选电机端 6 类故障）
    ↓
样本拆分
wave(2048) + sensor(15) + condition(18)
    ↓
train / val / test 分层划分
    ↓
归一化处理
(per-sample / global-train)
    ↓
基线模型训练
(CNN / TCN)
    ↓
性能评估
Acc / Macro-F1 / Balanced Acc / Per-class Recall / Top-k
    ↓
────────────────────────────
    ↓
训练层优化                     特征层优化
│                              │
├─ Weighted CE                ├─ MultiScaleConv1D
├─ Focal Loss                 ├─ 1stage / 2stage
├─ CB-CE                      ├─ kernel 对比
├─ WeightedRandomSampler      └─ Center Loss
└─ Stage-Adaptive Loss
    ↓                              ↓
各模块消融对比实验  ←──────────────→  综合优化结果
    ↓
最优模型与阶段性结论



| 符号                                | 含义                                |
| ----------------------------------- | ----------------------------------- |
| $C$                                 | 类别总数                            |
| $n_c$                               | 第 $c$ 类样本数量                   |
| $N=\displaystyle\sum_{c=1}^{C} n_c$ | 训练集总样本数                      |
| $t$                                 | 当前样本的真实类别                  |
| $p_t$                               | 模型对真实类别 $t$ 的预测概率       |
| $w_c$                               | 第 $c$ 类的类别平衡权重             |
| $w_t$                               | 当前样本真实类别 $t$ 对应的损失权重 |
| $\alpha_t$                          | Focal Loss 中真实类别对应的平衡因子 |
| $s_i$                               | 第 $i$ 个样本的采样权重             |

## 故障标签

由全data中筛选电机端故障标签

`7, 9, 10, 11, 12, 13`

| 标签 | 名称                 | 数据标签                     | 数量    |
| ---- | -------------------- | ---------------------------- | ------- |
| C5   | 电机偏心             | L7 电机偏心                  | 1164    |
| C3   | 双侧轴承保持架故障   | L9 双轴承支架破坏            | 715     |
| C2   | 非驱动端轴承外圈故障 | L10 西边轴承滚珠外圈磨损     | **363** |
| C4   | 双侧轴承外圈故障     | L11 双轴承滚珠外圈破坏性磨损 | 1038    |
| C0   | 无故障               | L12 正常运行                 | *2394*  |
| C1   | 驱动端轴承保持架故障 | L13 轴承支架破坏             | **499** |



## 整体框架(最新版

变为baseline + model_opt，受到train_opt.py调用的结构

```
project/
├─ baseline_train.py          # 基线训练入口：数据读取、样本筛选、训练与评估
├─ train_opt.py               # 特征层优化实验入口：cfg驱动、run_dir生成、计时
├─ train_trainlayer.py        # 训练层优化实验入口：sampler / focal / cb-ce
├─ train_centerloss.py        # 嵌入空间约束实验入口：center loss
├─ models_opt.py              # 特征层优化模型：MultiScaleConv1D、CNN1D_MS等
├─ losses_opt.py              # FocalLoss、CB权重、CenterLoss等
├─ data.py                    # 数据说明、按工况/测点/状态分解、统计特征与箱线图
├─ summarize_runs.py          # runs结果汇总、表格导出与绘图
├─ configs/                   # 实验配置（json）
│  ├─ ...
└─ runs/                      # 实验输出
   └─ <exp_name_yyyymmdd_hhmmss>/
      ├─ cfg_used.json
      ├─ history.json
      ├─ test_metrics.json
      ├─ class_counts.json
      ├─ label_map.json
      ├─ confusion_matrix.png
      ├─ best.pt
      └─ time.json
```

到目前为止，项目完成了以下工作：

1. 电机端 6 类故障标签筛选与重映射使用 2081 33 
2. 建立 CNN / TCN 基线模型，并完成 wave-only / use_meta / meta-only 的对比，验证工况/测点信息对分类的提升（在此过程中选择CNN作为进一步深入的模型进行调试  
3. 实现了特征层多尺度卷积优化，包括 1stage / 2stage 及 359 / 3715 的 kernels 对比
4. 实现了训练层不平衡优化，包括 Weighted CE（一开始的baseline就用的这个，后面想起来才改框架补的最基础的nn.CrossEntropyLoss()）、Focal Loss、CB-CE、WeightedRandomSampler
5. 实现了嵌入空间约束 Center Loss，并形成独立训练脚本（复用了挺多函数，后续再整理的时候考虑汇总一个ultis.py出来）
6. 建立了统一的 cfg 驱动实验框架与 runs 结果保存规范



==TODO==

前面的一版结果整理的脚本在新的（更为混乱未经整理）的runs文件结果中直接跑的效果应该会比较丑陋，所以暂时没有成表的统计，会迅速重写

把实验的结果汇总为对比消融实验的文字内容

任务规划中的特征、训练层优化基本完成，在考虑数据层的计划（到中期前的主要任务）

FL和cb_ce的参数还可以调优



==设置场景，构造极少类检验==

==破坏性试验，时序性良好==



### 做meta_only检查工况/测点–故障的捷径

修改了训练模型和cfg设定的框架，添加meta_only进行工况/测点 —— 故障的训练检验，顺便添加了收敛曲线绘制，重跑前面几个baseline的实验。

分层优化的内容打算重跑后再整理检查下

根据meta_only的结果看需不需要加错配meta信息——结果显示使用了meta信息会有更高的监测准度



## 分层优化

### 特征层

#### 多尺度表征与融合的实验

前面实现了cnn_ms，是把前两层换成了MultiScaleConv1D，设定的kernerls=(3,7,15)

现在考虑在实现多尺度表征的基础上做一层与两层MultiScaleConv1D的对比，并设定区分kernerls=(3,5,9) or (3,7,15)，对比大小尺度的效果差异，从中选优后，再进行后续特征的优化(嵌入空间约束），

当前 baseline（CNN wave-only）在 6 类电机故障上整体已达到 macro-F1≈0.971，但仍存在“最弱类”召回率 ≈0.935 类，后续特征层优化重点观察该类 recall 是否提升以及混淆矩阵中主要误判对是否减少。



已实现 CNN_MS ，在原始 1D-CNN 的浅层卷积位置引入 `MultiScaleConv1D`，通过并行小核、中核、大核卷积同时提取不同时间尺度下的振动模式，再用 `1×1` 卷积完成跨尺度融合与通道压缩，从而增强模型对短时冲击、周期纹理和缓慢调制等多种故障特征的表达能力；在此基础上，设置1stage / 2stage两种结构，用于验证多尺度作用位置的贡献，并通过 kernel 组合对比（ `(3,7,15)` 与 `(3,5,9)`）分析不同尺度覆盖范围对故障识别性能、尾类召回率和混淆对变化的影响，这些实验共同构成了当前特征层优化的主要实现逻辑。

训练参数一开始没搞好效率很低。。。

最后得到的实验结果显示ms2_359的效果较好，可以进一步讨论其对尺度的适应性

but— 提升的效果都不是很明显，一开始的baseline就还挺好

但是一开始的baseline带了点ce，是否需要补最原始baseline？





#### 嵌入空间约束

即使加入多尺度卷积，某些故障类别在特征空间中仍可能与其他类别靠得较近，尤其是尾类样本由于数量少，更容易在嵌入空间中被多数类“挤压”。因此，在特征层优化基础上，进一步引入嵌入空间约束，使同类样本更紧凑、不同类更分离，是合理的下一步。

基本形式
Center Loss 通过为每个类别维护一个中心向量，并约束样本 embedding 向其所属类别中心靠近。其形式为：
$$
L = L_{cls} + \lambda L_{center},
$$
其中分类损失 \(L_{cls}\) 可取 CE / Focal / CB-CE，Center Loss 为：
$$
L_{center} = \frac{1}{2}\sum_i \|f_i - c_{y_i}\|_2^2,
$$
其中 \(f_i\) 为样本特征向量，\(c_{y_i}\) 为其对应类别中心。

==取消种子，10次取均值方差==

### 训练层 ———— ==各种loss的优缺点和融合调试==

baseline里直接做的类别重加权weighted CE，具体来说……见下

结合调整loss

#### Focal Loss 和 Class-Balanced CrossEntropy

**加权交叉熵（Weighted Cross-Entropy, CE）**

加权交叉熵是在标准交叉熵损失基础上引入类别权重，以缓解类别不平衡对模型训练的偏置。对于多分类任务，普通交叉熵可写为
$$
L_{CE}=-\log(p_t),
$$
其中 $p_t$ 表示模型对真实类别的预测概率。在长尾分布场景下，若直接使用普通交叉熵，多数类样本因数量占优会主导梯度更新，导致模型倾向于优先优化多数类而忽视尾类。为此，可为每个类别引入权重 $w_c$，构造加权交叉熵：
$$
L_{WCE}=-w_t\log(p_t),
$$
其中 $w_t$ 为真实类别对应的权重。在本文实现中，类别权重通常按样本数的逆频率构造，即样本数越少的类别权重越大，从而放大尾类样本在总损失中的贡献。该方法实现简单、可解释性强，是处理类别不平衡问题的基础策略，但其局限在于：若直接采用逆频率加权，极少数类的权重可能过大，容易引起训练不稳定或对个别样本过度敏感。

------

**Focal Loss**

Focal Loss 在交叉熵基础上进一步引入**难样本聚焦机制**，其核心思想是降低“已被模型正确分类的容易样本”的损失权重，同时提升“难分类样本”对参数更新的影响。其形式可写为
$$
L_{FL}=-\alpha_t(1-p_t)^\gamma \log(p_t),
$$
其中，$\alpha_t$ 为类别权重，用于补偿类别不平衡；$(1-p_t)^\gamma$ 为焦点调制因子，$\gamma$ 为聚焦参数。当某个样本已经被模型较好分类时，$p_t$ 较大，则 $(1-p_t)^\gamma$ 很小，对应损失被抑制；反之，当样本较难分类时，$p_t$ 较小，其损失项会被相对放大。由此，Focal Loss 同时兼顾了**类别不平衡补偿**与**难样本强化学习**两方面目标。对于电机故障诊断这类存在尾类故障、类间混淆及弱故障特征的任务，Focal Loss 往往能够提升难分类故障和尾类故障的识别能力，尤其适合用于改善尾类召回率与降低漏检风险。

------

**Class-Balanced Cross-Entropy**

Class-Balanced Cross-Entropy 的核心改进在于：它并不直接采用简单的类别逆频率作为权重，而是基于**有效样本数（effective number of samples）**来估计各类别的真实信息量，从而构造更加平滑、稳定的类别权重。其权重形式通常写为
$$
w_c=\frac{1-\beta}{1-\beta^{n_c}},
$$
其中 $n_c$ 表示类别 $c$ 的样本数，$\beta\in(0,1)$ 为接近 1 的常数（如 0.9999）。这样定义的动机在于：随着样本数增加，新增样本所带来的“有效信息增量”会逐渐减小，因此类别权重不应简单地与样本数严格成反比。基于该权重，CB-CE 可写为
$$
L_{CB-CE}=-w_t\log(p_t).
$$
与普通逆频率加权 CE 相比，CB-CE 在补偿尾类时更加平滑，避免了极少数类权重过大导致的训练震荡，更适合样本分布极不均衡的长尾分类任务。在你的故障诊断场景中，CB-CE 可以被看作是“比加权 CE 更稳健的类别平衡版本”，适合用来验证在长尾数据下是否能够更平衡地提升各类故障的识别性能。



WeightedRandomSampler 属于训练数据采样层面的类别平衡策略，其目标不是直接修改损失函数，而是在每个 epoch 的训练过程中通过调整样本被抽取的概率，改变模型实际“看到”的训练数据分布。在长尾分类任务中，如果仅采用普通的随机打乱（shuffle），训练批次中的类别比例通常仍接近原始数据分布，即多数类样本频繁出现、尾类样本出现较少，这会导致模型在训练初期和训练过程中持续受到多数类主导。WeightedRandomSampler 通过为每个样本赋予采样权重 $s_i$，并按该权重进行有放回抽样，使尾类样本在训练中被更频繁地采到。若样本 $x_i$ 属于类别 $y_i$，一个常见的构造方式是令
$$
s_i \propto \frac{1}{n_{y_i}},
$$
其中 $n_{y_i}$ 为该类别的样本数。这样，类别样本数越少，其对应样本的采样概率越高。与加权交叉熵不同，WeightedRandomSampler 的作用位置更靠前：它直接影响训练批次的组成，使模型在训练阶段就能接触更多尾类样本，从而缓解多数类对特征学习过程的压制。该方法实现简单、与现有网络结构解耦，常可与加权 CE、Focal Loss、CB-CE 等损失函数结合使用；



#### 代价/阈值策略

以减少漏检为目标允许部分误报



## 指标

Accuracy

Marcro-F1

Balanced Accuracy

Per-class recall

Top-k

Avg-condition



| `acc`：易受多数类影响 |                                |
| --------------------- | ------------------------------ |
| `macro_f1`）          | 更关注小类表现，适合不均衡     |
| `balanced_acc`        | 各类 recall 的平均             |
| `per_class_recall`：  | 逐类召回，方便定位“漏检哪一类” |
| `topk_acc@3`：        | Top-3 命中率                   |
| `avg_confidence`：    | 预测最大概率均值               |



## 最初的CNN&TCN架构

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

训练后发现tcn训练时间长效果不好

**Backbone**：4×TCNBlock（残差膨胀卷积）

- Channels：`(16,32,64,128)`
- Kernel：`k=5`
- Dilation：`(1,2,4,8)`
- 每块：`[Conv+BN+ReLU+Dropout] ×2 + Residual`

**Pooling**：`AdaptiveAvgPool1d(1)` → `feat ∈ R^{B×128}`

**Head**：

- meta off：`Linear(128→128) + ReLU + Dropout(0.2) + Linear(128→K)`
- meta on：`Concat(feat, meta)` → `Linear(128+33→128) ... → K`

