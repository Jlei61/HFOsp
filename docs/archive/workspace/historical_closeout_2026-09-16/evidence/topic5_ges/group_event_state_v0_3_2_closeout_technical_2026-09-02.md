# Group-Event State v0.3.2 阶段收口（技术版，复审更正）

**日期：** 2026-09-02

**状态：** `V0_3_2_PIPELINE_ACCEPTED_ASSAY_POWER_UNCALIBRATED_CLOSEOUT`

**分区：** development-only；`sealed_partition_opened=false`。

## 1. 正式验收状态

| 层级 | 状态 | 允许解释 |
|---|---|---|
| pipeline | accepted | 数据、状态训练、对照、冻结迁移和汇总可复现 |
| assay power | uncalibrated | positive recovery 尚无稳定功效曲线 |
| H1 | inconclusive, N=1 eligible | 只约束当前 count-trained representation |
| H2a | inconclusive, objective mismatch | count-view state 未稳定迁移到 grammar |
| H2b | not run | 不能写发作迁移阴性 |
| H3 | not run | 不能写 IED feedback 阴性 |

两条实现分支已合并到 `codex/topic5-group-event-state-v032-closeout`；结果生产提交为 `233f3ad1`。相关 v0.3.2 测试 65 项通过。leaky-bank 最大 PyTorch peak allocation 56.6 MB；H2a 每 GPU 两个 worker，无 OOM；收口时无相关训练进程。

## 2. 数据与分母

27 位患者完成 measurement/support/eligibility 与 history baseline。人体模型只运行三位固定 development 患者 × 三 seed。

30 分钟非重叠块数（base-fit / inner-val / dev-val / dev-test）：

- `epilepsiae_1146`：61 / 11 / 10 / 22；
- `yuquan_pengzihang`：20 / 2 / 3 / 7；
- `yuquan_zhangkexuan`：23 / 4 / 4 / 8，base-fit 比冻结阈值少 1。

因此 30 分钟 H1 primary 严格分母为 1；120 分钟严格分母为 0。H2a 三位均通过 prefix/contact-support 资格。

## 3. 当前模型实际学习了什么

### 3.1 输入不是 raw SEEG

v0.3.2 event token 来自群体事件的提取后特征：

- 逐触点 participation 与 leader；
- tied-group 数量和大小摘要；
- 精确 delay span/mean/std/median；
- 空间离散度；
- 逐频带 event-level energy、peak amplitude、peak time；
- cross-band lag；
- detector-confidence 与 coverage proxy。

代码明确排除 raw waveform、background SEEG、seizure label 和把 IEI 当普通 feature。真实时间只通过固定衰减进入状态。因此本轮不能回答“原始脑电编码器是否学会”。

### 3.2 状态与 readout

主状态是 12 维 fixed-timescale marked leaky bank：5/30/120 min 三个固定时间尺度，每个四通道。两层 MLP `D→32→4` 将 event token 映射成四维 write，并写入三个时间尺度。主 readout 只有一个 scalar：

```text
log μ(H+S) = log μ(H) + α wᵀS
```

这意味着 count objective 直接监督的有效方向最多先表现为一个 scalar subspace；12 维 nominal state 不等于 12 维都被识别。

### 3.3 唯一使用过的训练配方

| 参数 | v0.3.2 |
|---|---:|
| encoder hidden / write dim | 32 / 4 |
| encoder LR / adapter LR | 1e−3 / 3e−3 |
| weight decay | 1e−4 |
| alpha init | 0.03 |
| alpha freeze | 前 50 steps |
| max/min steps | 600 / 100 |
| validation interval / patience | 10 / 10 validations |
| gradient clip | 1.0 |

九个 learned run 的 selected step 均为 20–50；因此所有入选 checkpoint 的 `alpha` 都仍为初始化值 0.03。encoder 权重在 selected step 有非零梯度，训练 NLL 也下降，说明不是完全 dead path；但本轮没有做学习率、gate schedule、容量、正则、batching 或训练预算搜索，不能排除 training-recipe failure。

## 4. synthetic assay 的正确解读

### 4.1 null

H-only null 为 0/6 observed false positives，median gain −0.00881 nats/anchor。它只能称 sanity check；6 次零假阳性的单侧 95% 上界仍很宽，不能写 specificity established。

### 4.2 positive

预登记 β=0.35：0/3 满足完整 CI 规则；三个 continuous gain 为 +0.0193、+0.0866、+0.0020。

追加 ladder：

| β | median continuous gain | CI-rule recovered |
|---:|---:|---:|
| 0.35 | +0.0227 | 2/3 |
| 0.70 | +0.1931 | 3/3 |
| 1.40 | +0.2738 | 1/3 |

continuous median gain 随 β 单调增加。非单调的是三次 replicate 下的 pass/fail 计数，主要受 block variance 与 CI 阈值影响。β=1.40 还可能将 count 推出真实 support，不能把 β 数值直接解释为生理 small/medium/large。

### 4.3 未完成的 assay 定标

当前 `r2_hidden_vs_h_state_train=0.97–0.999` 来自 235 行、126 维的 in-sample regression，容易过拟合。下一版需要 purged blocked OOS `R²` 与 oracle held-out deviance gain。

**判定：** `ASSAY_POWER_UNCALIBRATED`，不是 `INSTRUMENT_UNSTABLE`。

## 5. canonical evaluator 尚未闭合

E1146 的同一科学比较在两条路径中方向相反：

- model internal dev-test：`H−correct=+0.1277`；
- unified H1 paired evaluator：`correct vs H=−0.3291`。

可能差异包括 anchor、dispersion、intercept/calibration、checkpoint、block weighting、seed aggregation 或 NLL reduction。v0.3.3 在任何新人体训练之前必须建立 canonical evaluator，并在同一 checkpoint/anchor 上达到数值容差。

## 6. H1 人体结果

30 分钟 primary：

| subject | correct vs H | correct vs shifted | dynamic vs mean | eligible |
|---|---:|---:|---:|---:|
| E1146 | −0.3291 | −0.4433 | −0.3291 | yes |
| Peng | +0.3796 | +0.9607 | +1.4055 | no |
| Zhang | −0.7877 | −0.3674 | −0.2955 | no |

严格结论只来自 E1146：当前 count-trained leaky representation 在该 development block 不提供 residual count gain。Peng 的有利读数保留为低支持诊断，不能进入主统计。120 分钟无合格患者。

`dynamic vs mean` 可能与 `correct vs H` 数学重合，因为 TRAIN-centred mean state 接近零且 H 已有 static calibration；mean-state 只保留为 sanity check，不视为独立科学证据。

**H1：** `H1_INCONCLUSIVE_N1`。

## 7. H2a 人体结果

冻结 count-state 后训练低容量 grammar adapter。主 subset endpoint 的患者均值：

| subject | vs H | vs shifted mean | vs mean | vs test-best-control |
|---|---:|---:|---:|---:|
| E1146 | −0.00242 | −0.00246 | −0.00244 | −0.00846 |
| Peng | −0.01384 | +0.01541 | +0.02115 | −0.01385 |
| Zhang | +0.03756 | +0.01356 | +0.03793 | +0.00475 |

test-best-control 是在 H、mean 与五个 shift 中选测试性能最好的对照，存在极值选择偏差，已降为 adversarial sensitivity。主比较为 `vs H` 与 shift-null 分布；当前只有五个 shift 的均值，尚不足以构成经验 null。

更根本的问题是 objective mismatch：count-trained state 的阴性迁移不能排除 grammar-trained state。continue、positive size、subset identity 与 later continuation 都没有形成稳定共享模式。

**H2a：** `H2A_INCONCLUSIVE_OBJECTIVE_MISMATCH`。

## 8. 图形收口修复

H1 图只显示事前 eligibility 合格患者；不再跨 horizon 连患者线或 median；n=1 不画 cohort median，n=0 留空。H2a 图分别显示 `vs H` 与 `vs shifted mean`，不再以 test-best-control 承担主视觉结论。右侧留白和 footer 已重新调整，PNG/PDF 与 metadata 同步重生成。

## 9. 未运行边界

- repaired RNN：未运行正式 triage；不能归为架构失败。
- H2b：未运行；下一版允许在 state 由 interictal objective 锁定后作 development-only frozen transfer，且不得反馈模型选择。
- H3：未运行；保持机制扩展，不是 H1/H2b gate。
- sealed partition：未打开。

## 10. 下一版必须分开的四个原因层

1. **Evaluator/assay：** 真效应能否被量到；
2. **Optimization/hyperparameters：** 网络是否在当前架构内学会；
3. **Objective alignment：** count 与 grammar 是否需要不同表示；
4. **Architecture：** 只有前三层通过后才比较 leaky 与 repaired gated state。

完整 spec 与 plan：

- `group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md`
- `group_event_state_v0_3_3_dual_view_state_plan_2026-09-02.md`
