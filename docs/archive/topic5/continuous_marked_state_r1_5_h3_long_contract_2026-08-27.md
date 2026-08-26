# Continuous marked-state R1.5 / H3-long development 合同

**冻结日期：** 2026-08-27

**性质：** development exploration；不打开 formal/sealed partition。
**上游依据：** R1.4/T2-R2.0 closeout。R1.4 的稳定 T1 仅见于原 R1.3 三例；新增三例 0/3。N=100 T2 没有可扩尺度的阳性组合。

## 1. 本轮真正回答什么

本轮有两条彼此独立、不能互相借结论的路线。

1. **R1.5 independent extension：** 在事前固定的新患者上检验 explicit continuous observation 是否形成跨窗口、正确时刻特异的 predictive state，并是否改善下一 IED 的 first subset 与 continuation。
2. **H3-long exploration：** 直接检验过去约 1,000、3,000、10,000 次 IED 中无法由 pre-event state/history/observation 预测的部分，是否仍对下一事件或未来冻结状态有增量。N=100 阴性不 gate 本探索；但没有稳定 T1 时，只能报告 antecedent prediction，不能称 generator-state update。

## 2. 事前患者分层

### 2.1 R1.5 六人

按修复后的真实记录覆盖段、事件量与数据异质性固定：

- exact-model 未见的新扩展层：`epilepsiae_1096`、`epilepsiae_384`、`yuquan_zhangkexuan`；
- 已被旧长记录分诊看过的校准层：`yuquan_chengshuai`、`yuquan_chenziyang`、`yuquan_zhangjiaqi`。

后三人无论本轮结果如何，都不计作全新的独立复现。张家齐此前 T1 为 no-update。不得按结果换人或删除患者。

### 2.2 H3-long 支持层

全队列先按 **recorded coverage segment** 重算每个 N 的 TRAIN/validation 支持。`event_session` 不得代替真实覆盖段。尺度没有足够历史时写 `NOT_APPLICABLE_SUPPORT`，不记为阴性。

## 3. R1.5 冻结设置

- 主输入：spectral + variance + autocorrelation explicit observation；本轮不再扩 raw encoder 家族。
- 模型、损失与 R1.4 explicit arm 相同；每患者固定 5 seeds：0–4。
- 每个 seed 均从 matching-seed R1.2 初始化，不能继承别的患者/seed/checkpoint。
- 主要比较：persistent−memoryless、correct−5 matched wrong-time donors。
- 端点拆分：timing、STOP/group size、first subset、later continuation。
- 10-donor 仅用同一 checkpoint 做敏感性，不重训。

患者级稳定状态的描述条件为：至少 3/5 seeds 同时满足 persistent−memoryless < 0 与 correct−wrong < 0。它只控制“允许叫什么”，不阻止其他患者实验完成。

## 4. H3-long 冻结设置

### 4.1 尺度

- `N=1000`：主要长尺度；
- `N=3000`：更长敏感性；
- `N=10000`：数据边界探针。

N 是事件计数记忆尺度，不直接命名为小时、昼夜或生理时间常数。每个患者另报中位实际时长和 IQR。

### 4.2 Exposure

两条 source 全部运行，不因另一条阴性停止：

- scalar load innovation；
- low-rank participation innovation。

innovation 只在 TRAIN 内交叉拟合：

```text
event attribute ~ pre-event state + fixed history + current explicit observation
innovation = observed - cross-fitted expectation
```

累积量按 recorded segment 重置。每个尺度的 exposure 用 TRAIN 均值中心化、TRAIN 标准差缩放；validation 不重新估计尺度。

### 4.3 对照

所有可比臂共享完全相同的事件支持、同一拟合截距与相同参数预算：

1. real cumulative innovation；
2. state-matched non-overlap placebo：有效历史窗不得重叠；
3. causal phase/block placebo：保持边际分布和记录段，但破坏正确累积时序；
4. current-event-only；
5. intercept-only；
6. no-edge 只作诊断，不承担主结论。

真实臂必须相对 intercept、state-matched 与 phase placebo 分别报告。滑动事件行不是独立样本；必须同时报告 TRAIN/validation 的不重叠 N-event 块数和有效信息量。

### 4.4 两级人体问题

- **H3-SL0 antecedent screen：** 在所有支持患者上预测 next-event timing、STOP、first subset、continuation。它不要求稳定 T1，但只允许称长历史关联。
- **H3-SL1 generator edge：** 仅在 R1.5 或既有 R1.4 中有稳定 T1 的患者上，从同一 pre-event state 出发拟合 signed exposure edge，并检查 next-event 与无新 observation/无后续 jump 的 H5/H10。没有稳定 T1 不阻止 SL0，只阻止机制命名。

## 5. Synthetic recovery 先于人体数值

每个核心尺度至少 3 seeds，必须覆盖：

1. positive edge；
2. zero edge；
3. reversed-sign edge；
4. exposure-free constant-offset target；
5. exposure-free drifting target。

需验证：零真值不被截距冒充；正负方向可恢复；结构零与优化失败可区分；错误时序不能靠更差 placebo 形成假阳性；改变算子整体缩放不改变 ridge 选择。

## 6. 运行与资源

- R1.5 使用 3090 GPU，最多 2 个显存重作业并发；自动减 batch、AMP 和 checkpointing，禁止以 OOM 删除患者。
- CPU 聚合/innovation/placebo/synthetic 使用 `OMP_NUM_THREADS=1`，可并行 6–10 workers。
- 所有长跑由 `setsid`/`nohup` 或 tmux 启动，原子 manifest、可断点续跑；网络中断不影响本地实验。
- ordinary negative 完整保留；只有 split 泄漏、记录映射错误、formal/sealed 误开或合成仪器失败才停止人体解释。

## 7. 允许结论

- exact-model 未见的三位新扩展患者中稳定：新增 development replication；否则写未独立复现。旧长记录三人只能作 calibration。
- persistent 但 correct-time 阴性：persistent predictive memory，不称 time-specific state。
- 长历史只改善 STOP：termination/extent antecedent，不称 repertoire shaping。
- SL0 阳性、SL1 阴性或不可估计：长历史有关联，但没有 generator-state update 证据。
- N=10000 只有少于约 2 个独立 validation 块：只作个案边界读数，不作可靠阴性或机制阳性。
- 所有结果均为 development prediction；不升级为 IED 因果塑造癫痫网络。

## 8. 最终交付

- frozen subject/support manifest；
- R1.5 与 H3-long per-seed 产物、patient-first CSV/JSON；
- synthetic、split、gradient、rank、edge movement、独立块数机器审计；
- 白话版和技术版报告；
- formal/sealed/paper-ready untouched 审计；
- 仅提交本工作包文件并推送当前 goal 分支。
