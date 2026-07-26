# Persistent path-mode graph RNN：bounded-negative 收口

## 审阅结论

### 一句话判断

当前实现完成了原定的低成本证伪任务，但没有通过进入 34 人正式实验的科学门。模型
学到了局部 next-contact 规律，却没有形成可辨识、可稳定生成完整传播顺序的
event-persistent latent path。

### 完成度

> **完成度：100/100（按 bounded-negative stop rule）**

完成项包括：34 人 train80-only path prior、K=1–4、3 位患者 × 3 seeds、强对照、
结构消融、117/117 完整运行、资源与日志记录、节点分布、posterior/state/inhibition
分析、paper-ready 六块图、Methods/Results/caption 与 claim boundary。未运行的
34×3 和发作期任务不是欠项，而是硬门失败后合同明确禁止的步骤。

## 科学合同

- 输入：masked interictal contact-rank events。
- 切分：每位患者 chronological 80/20。
- prior：仅 train80；不使用 A/B、IEI 或发作期信息。
- pilot：`epilepsiae_1073`、`epilepsiae_1146`、
  `yuquan_chenziyang`；seeds `20260726/27/28`。
- 模型：每个事件固定一个 `(path mode, direction)` component，前缀因果更新
  posterior。
- 候选：K=1/2/3/4。
- 决策：只有 precedence、whole-path、seed stability 和结构 lesion 全部通过，
  才允许进入 34×3 和发作期 readout。

## 运行与工程验收

- 预期运行：117。
- 完成运行：117。
- 失败运行：0。
- 每次运行均有 checkpoint、metrics、event modes、rollouts、state 和日志。
- 峰值 GPU 显存：每进程不超过 298 MB。
- 单元测试：13/13 通过。
- 所有 prior 和运行均保持 `ictal_target_read=false`。

## 主要结果

### 1. Path bases 稳定，但未见事件重建有限

| K | split-half cosine | held-out reconstruction |
|---:|---:|---:|
| 1 | 0.982 | 0.408 |
| 2 | 0.943 | 0.425 |
| 3 | 0.884 | 0.463 |
| 4 | 0.888 | 0.468 |

这说明 train80 中存在稳定的平均路径结构，但该结构只能中等程度重建具体 heldout
事件。

### 2. 局部 transition 可学

相对 no-history，K=1–4 的 held-out event NLL 中位 benefit 分别为
`0.025/0.030/0.031/0.029`，全部为 9/9 patient-seed 改善。相对同密度
edge-weight shuffle，中位 benefit 为 `0.018/0.019/0.017/0.018`，同样全部
9/9 改善。

但是，相对 single aggregate path 的中位 benefit 只有
`0.003/0.002/0.001/−0.000`；K=2–4 相对 mode shuffle 基本为零。这说明局部
预测收益主要来自患者 transition scaffold，而不是来自“同一事件内保持 coherent
mode”的额外结构。

### 3. 节点分布只恢复了一部分

独立固定 K=2 作为最小多路径诊断：

- pooled contact participation：`r=0.49`；
- pooled contact mean rank：`r=0.23`；
- patient-seed median participation MAE：`0.142`；
- patient-seed median mean-rank MAE：`0.087`。

模型比完整顺序更容易恢复“哪个触点会参与”，却没有稳定恢复“该触点通常在事件的
哪个阶段出现”。

### 4. Latent path 不可辨识

| K | normalized posterior entropy | information fraction |
|---:|---:|---:|
| 1 | 0.955 | 4.5% |
| 2 | 0.978 | 2.2% |
| 3 | 0.983 | 1.7% |
| 4 | 0.987 | 1.3% |

K=2 中，median entropy 从事件开始的 `1.000` 只降到事件结束的 `0.978`。同时
recurrent state 绝对值从约 `0.003` 墠至 `0.190`，inhibition 从约 `0.0002`
增至 `0.031`。因此网络内部并非没有状态变化；失败在于这些状态变化没有把事件
归到可辨识的 path identity。

### 5. 硬门失败

- K=1：comparison、stability、lesion 均失败。
- K=2：comparison、stability、lesion 均失败。
- K=3：comparison、stability、lesion 均失败。
- K=4：dominant-mode lesion 单项通过，但 comparison 与 stability 失败。
- seed ranking stability：precedence `ρ=0.00`；whole-path `ρ=0.20`，均低于
  预注册门 `0.40`。

因此 `selected K = null`，下一步固定为
`bounded_negative_stop_no_ictal_read`。

## 科学解释

这是对一个具体假设的否定：**整次间期事件由一条离散 path mode 和一个方向持续驱动**。
它不否定患者间期传播 scaffold，也不否定间期与发作早期静态能量场之间的经验关系。
相反，结果把两层区分开来：

1. 患者级平均 transition scaffold 含有可学习信息；
2. 这些信息不足以支持单条、整事件持续的 latent path 生成机制。

因此该 RNN 不应进入主文正向机制链，也不应继续做发作期迁移。若保留，最安全位置是
补充材料中的预注册模型证伪。

## 复现与产物

- 执行合同：
  `docs/superpowers/specs/2026-07-26-topic5-persistent-path-mode-rnn-v0_9.md`
- 配置：`config/topic5_persistent_path_mode_rnn_v0_9.yaml`
- prior：`results/topic5_structured_axis_graph/path_mode_prior_v0_9/`
- pilot：
  `results/topic5_structured_axis_graph/screen_persistent_path_mode_v0_9/`
- gate：
  `results/topic5_structured_axis_graph/screen_persistent_path_mode_v0_9/analysis/pilot_gate_summary.json`
- bounded-negative 分析：
  `results/topic5_structured_axis_graph/screen_persistent_path_mode_v0_9/analysis/bounded_negative/`
- paper-ready 图：
  `results/paper-ready-figure/fig6_persistent_path_mode_bounded_negative/figures/`
- 论文文字：
  `docs/paper-draft/figure6_persistent_path_mode_rnn_bounded_negative.md`
