# 可直接交给 Agent C 的 Prompt：Scientific State Experiments

你接手 `/home/honglab/leijiaxin/HFOsp` 的 Group-Event State v0.3.3 Workstream C。你的职责是定义并执行 R0/R1 dual-view state、sharedness、H1/H2a 和 frozen H2b；你不能自行临时改 optimizer/LR，训练请求必须交给 Agent B。

## 1. 开始前

1. 完整阅读 handoff 索引列出的共同文档。
2. 审计 worktree、base commit、dirty state、已有 registry、活动作业和旧结果。
3. 从 supervisor 指定的 clean release commit 建 `codex/topic5-ges-v033-scientific-experiments`。
4. release 文件缺失时，只实现目标、probe、shared/private、H2b risk、单元测试和 synthetic plumbing；不启动承重人体训练。
5. 不打开 sealed，不运行人体 H3，不碰 paper-ready Fig1–Fig4。

## 2. 核心科学问题

1. `H_rate→H_mark` 后，群体事件历史是否仍含 future-burden 或 conditional grammar 信息？
2. contact-resolved R1 是否比 summary R0 保留更多可泛化状态信息？
3. `S_N` 与 `S_G` 各自在自己的任务是否有效，是否单向/双向迁移，是否存在 shared/private subspace？
4. 给定相同 early prefix，pre-event state 是否改变 subset identity 和 later continuation？
5. 完全冻结的间期 state 是否跨到 seizure risk？

## 3. 必须实现

### C1. 显式历史阶梯

同一 anchor、同 evaluator 比较：

```text
H_rate
H_mark
H_mark + state
```

`H_rate` 包含 IEI、多尺度 counts、clock/session、coverage；`H_mark` 再加 extent/STOP EMA、contact/repertoire occupancy 和 multiband EMA。

### C2. R0/R1

- R0：summary token；
- R1：contact-resolved participation、exact delay、multiband peak/energy、shaft/coordinates、validity mask。

R0/R1 使用相同 search budget 和 inner-validation；训练由 Agent B 执行。

### C3. `S_N`

future-burden target：

```text
[N_0–5min, N_5–15min, N_15–30min]
```

同时报告 `N_0–30min`。write width 2/4/8，总 state 6/12/24；由 inner-validation near-best 最小容量规则锁定。

### C4. `S_G`

- `G-primary`：`subset identity | K,prefix`；
- `G-composite`：subset primary + low-weight continue/positive-size。

future block 内使用同一个 anchor state，后来事件不得更新待评分 state。仅对 `N_future>0` anchor 评分，同时报告 first-future-event 与 block-average grammar。later continuation 和 multiband 先作 frozen probes。

记录多任务 gradient norm/cosine；只有持续冲突被 Agent B 复核后，才请求 PCGrad。

### C5. Within-view、cross-transfer 与 shared/private

冻结 producers 后运行：

```text
S_N → count / grammar
S_G → grammar / count
```

只用 TRAIN 拟合 regularized CCA 或 reduced-rank regression，得到 `Z_shared/U_N/U_G`，在 later blocks 评价。双向迁移是强证据但非唯一条件；只有 within-view 均有效、双向不迁移且 shared subspace 无增量，才可称 separable predictive states。

拼接状态必须用 TRAIN-only 低秩投影或等自由度 probe 做容量匹配。

### C6. H1/H2a

H1 固定时间 anchor：

```text
H_rate → H_mark → H_mark+S_correct → H_mark+S_block-shifted
```

主 horizon 5/30 min。block circular shift 同 patient/session，偏移大于 target horizon，保留状态自相关。

`S_G` 训练时 contact decoder 主干固定、梯度经低容量 adapter 回到 producer。H2a transfer 时 producer 和 decoder 全冻结，只在 TRAIN 拟合容量受限 probe；subset identity 为主，continue/size/later continuation 分项。

### C7. 2 tuning + 4 untouched replication

两位 tuning patients 仅按 support 指标选定，用于锁定 R0/R1、state dimension、`G-primary/G-composite`、probe capacity、shift 和 checkpoint rule。随后四位 untouched development patients 单次运行，不重新调参、不因结果剔除。

每位先合并 seeds，再 patient-first 汇总；不可估 endpoint/horizon 明确退出对应分母。

### C8. Frozen H2b risk

使用 5 min fixed-grid 离散 survival：

```text
baseline
baseline + S_N
baseline + S_G
baseline + Z_shared
baseline + capacity-matched [S_N,S_G]
```

baseline 包含 time since last seizure、postictal/cluster 和可用临床背景。报告 Brier skill、log score、calibration，AUROC secondary。state、shared decomposition、输入和超参数必须在读取 seizure outcome 前冻结；H2b 不得反向选择 state。

### C9. 探索接口

R2 waveform、small shared producer、early ictal field/path、60/120 min 和 repaired gated 都不进入核心 DoD。R2 只能用 masked/partial prediction，不能靠重建输入中已有标签宣布学会。

## 4. 与 Agent A/B 协作

- 从 Agent A 读取 canonical evaluator、power-derived eligibility 和 data-boundary manifest；不复制 scoring。
- 向 Agent B 写原子 training requests；不直接改训练配置。
- noncanonical 人体输出可以保留为 `DIAGNOSTIC`，但不得进入主图。
- H2b 可以在 H1 不显著时运行，但不得充当模型选择器。

## 5. 写权限与资源

你独占：

```text
/data/hfosp_group_event_state_v0_3_3/agent_c/
results/group_event_state/v0_3_3/scientific_experiments/
shared/job_requests/science_*.json
```

CPU/GPU job 必须通过 supervisor/Agent B lease。所有线程为 1；长作业 nohup+setsid 或 tmux；原子状态、独立 log、幂等 resume。不得抢其它 agent 或 Topic 4 资源，不得 `pkill -f`。

## 6. 交付

核心图：

1. `H_rate→H_mark→H_mark+S` count/grammar gain；
2. within/cross-transfer + shared/private matrix；
3. H2a subset/later-continuation；
4. frozen H2b risk skill/calibration。

生成 PNG/PDF/metadata 和中文 figures README，逐图目视审查分母、方向和科学含义。

交付 plain/technical 报告、逐 patient/seed/anchor machine tables、checkpoint registry entries、完整复现命令和 `CURRENT_HANDOFF.md`。分开写工程完成、training adequacy、comparability 与科学结论。
