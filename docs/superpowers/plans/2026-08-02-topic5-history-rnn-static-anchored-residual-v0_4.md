# Topic 5 history-conditioned early-ictal field refinement v0.4 执行计划

对应 spec：

```text
docs/superpowers/specs/2026-08-02-topic5-history-rnn-static-anchored-residual-v0_4.md
```

本计划只实现四模型、三个关键解释对比和两个针对性历史对照。没有 architecture sweep、development 科学筛选或复合 hard gate。

---

## Milestone 1：冻结数据合同与编码坐标

### 1.1 建立输入 provenance manifest

只读核对并哈希：

- 15 位 primary patient inventory；
- `epilepsiae_1146` engineering-only 标记；
- clinical-onset `[0,10] s` **1–45 Hz** target cache 和字段来源；
- 1–150 Hz no-retrain sensitivity target；
- frozen static A/B template-field JSON；
- causal-prefix timeline、600 s guard、postictal exclusion；
- c30 target-blind LOSO checkpoints；
- exact contact order 和每 seizure contact denominator。

输出：

```text
results/topic5_history_conditioned_field_refinement_v0_4/INPUT_MANIFEST.json
```

如果现有 cache 不能证明某 target 字段是 1–45 Hz，先沿生产 artifact 追溯并重建 manifest，不能把 1–150 Hz 文件改名后冒充 primary。

### 1.2 复算 primary static anchor

在 1–45 Hz、完全相同 contact denominator 上复算 M0：

- per-seizure maxAB；
- per-patient median；
- 5000 次 matched channel null；
- 与论文现有 1–45 Hz 静态结果核对。

此前 v0.3 的 1–150 Hz 数字只登记为 sensitivity provenance，不作为这一项的逐位复现目标。

### 1.3 重建 outer-fold cache

新增或修改 cache builder，使其按 outer fold 运行：

```text
cache/outer_<heldout_patient>/<encoded_subject>/...
```

在 heldout patient \(p\) 的 fold 内：

1. 只加载 `E^(-p)` 对应的 target-blind c30 checkpoint；
2. 用它编码 14 位 target-training patients 和 heldout patient；
3. 每个 cache 写入同一 encoder checkpoint hash、code hash 和 input hash；
4. 拒绝加载由其他 subject-specific checkpoints 产生的 embeddings。

这是 P0 正确性门。未通过时不得开始正式训练。

### 1.4 静态基底的边界写入产物

manifest 必须显式记录：static A/B 来自全记录间期事件，可能包含目标发作之后的数据；它 target-blind，但不是 prospective。这个字段必须传入最终 summary 和 figure caption source table。

---

## Milestone 2：实现四个模型和精确数值合同

### 2.1 核心模块

更新或新增：

```text
src/topic5_static_anchored_history_residual.py
```

只包含：

1. `center()` / `unit_eps()`；
2. 带 residual-norm threshold 的 `compose_static_residual()`；
3. `soft_rank()` / `soft_maxab_loss()`；
4. `decay_state_to_cutoff()`；
5. M1 frozen-history head；
6. M2 fixed time-aware summary；
7. M3 joint-history model；
8. patient-balanced seizure loss；
9. exact evaluation maxAB。

数据读取、LOSO orchestration、统计和画图不放进核心模块。

### 2.2 四模型

正式模型名固定为：

```text
M0 STATIC_AB
M1 STATIC_AB_PLUS_FROZEN_HISTORY_HEAD
M2 STATIC_AB_PLUS_TIME_AWARE_NONRECURRENT
M3 STATIC_AB_PLUS_JOINT_RNN
```

实现要求：

- M1 完整训练 heads/gains，不能只做 5-epoch warm-up；
- M2 固定使用时间常数 \(\tau_0=2\) h 的 EWMA + mean/max + last event + count + span，不扫时间尺度；
- M2 的拼接 summary 只经一个线性层投影到 16 维，不加非线性隐藏层；
- M1 和 M3 共用前 30 epochs frozen-recurrent checkpoint，随后各训练 30 epochs；M1 继续冻结 recurrent/decay，M3 只额外解冻 HistoryGRU 和 decay；
- 分叉后的 M1/M3 使用相同 mini-batch 顺序并继承同一 head optimizer state，使更新预算匹配；
- EventRNN、contact embedding、normalization、static A/B 始终冻结；
- anchor penalty 只覆盖 recurrent weights 和 decay；
- gains 用近零不饱和的平方参数化约束在 `[0,1]`，初始化为 `1e-3`；
- 记录初始化 output 相对 static 的差和夹角。

### 2.3 删除正式 zero-state arm

从 runner、summary schema、统计和图中删除 `STATIC_AB_PLUS_RNN_ZERO_STATE`。只保留单元测试：zero state / `g=0` 应回到 M0。

### 2.4 两个 no-retrain history controls

在 M3 inference 中实现：

1. `ORDER_SHUFFLE_FULL_HISTORY`：整段 causal history 在原时间槽上置换，32 个固定 permutations；
2. `WITHIN_PATIENT_HISTORY_SWAP`：保持患者、static A/B、contact set 和 target 不变，替换为同患者其他发作的 history，遍历所有 donor 并取中位数。

两者都不得重训模型或读取 target 选择置换/donor。

### 2.5 单元与合成测试

在 `tests/test_topic5_history_rnn.py` 增加或更新：

1. primary endpoint 是 1–45 Hz；
2. 1–150 Hz 只能 no-retrain score；
3. 同一 outer fold 所有 patients 的 encoder hash 相同；
4. 不同 outer folds 可以使用不同 encoder；
5. heldout target 不进入训练；
6. `g=0` 逐位复现 M0；
7. tiny residual 不被单位化放大；
8. A/B 交换、整体翻号不改变 loss；
9. anchor penalty 不覆盖 heads/gains；
10. M1/M3 仅在 recurrent/decay trainability 上不同；
11. cutoff decay 随间隔增加不增大 state norm；
12. chunked 与 unchunked 短序列输出和梯度一致；
13. patient-balanced loss 不受复制某患者 seizures 影响；
14. order shuffle 覆盖完整历史且保留时间槽；
15. history swap 只在患者内且 contact name 对齐；
16. channel-null 每 draw 重做 A/B/sign selection；
17. synthetic history residual 能被 M1/M3 恢复，M0 不能。

运行：

```bash
python -m py_compile <new_or_changed_scripts>
bash -n <new_launchers>
pytest tests/test_topic5_history_rnn.py -q
git diff --check
```

---

## Milestone 3：固定训练器并完成 15-fold × 3-seed

### 3.1 Engineering smoke

在 `epilepsiae_1146` 上只运行：

- 1 epoch；
- 1 seed；
- 最长完整历史；
- forward/backward；
- cutoff decay；
- output schema；
- determinism、峰值 RAM/VRAM 和 chunk-gradient。

不汇报其 scientific maxAB，不依据它修改 LR、epoch、模型或 loss。

若 OOM，只允许依次：

1. gradient checkpointing；
2. 降低同时 seizure batch；
3. patient-balanced gradient accumulation；
4. 降低并发。

不得截短历史、detach chunk 或改变 hidden size。

### 3.2 冻结训练配置

不做 grid search。写入唯一 config：

```text
config/topic5_history_conditioned_field_refinement_v0_4.json
```

内容固定：

```text
optimizer: AdamW
weight_decay: 0
hidden_size: 16
chunk_events: 256
gradient_clip: 1.0
common_frozen_recurrent_head_epochs: 30
M1_frozen_recurrent_continuation_epochs: 30
M2_total_epochs: 60
M3_joint_epochs_after_common_stage: 30
head_gain_lr: 3e-4
recurrent_decay_lr: 1e-4
lambda_gain: 1e-3
lambda_anchor_recurrent_only: 1e-4
seeds: [11, 29, 47]
```

### 3.3 正式运行单位

每个 heldout primary patient × 3 seeds 依次训练：

1. M1/M3 共用的 frozen-recurrent 30-epoch stage；
2. M1 frozen-recurrent continuation 与 M3 joint continuation；
3. M2 matched-budget nonrecurrent model。

每个 unit 保存：

- fixed config 与全部 hashes；
- epoch log；
- best 不用于选择，正式使用 final checkpoint；
- heldout A/B candidate fields；
- raw residual、gain、decay、state diagnostics；
- peak RAM/VRAM、wall time；
- `DONE.json` 或 `FAILED.json`。

三个 seed 不按 heldout target 选择；A/B candidate fields 分别 contact-wise 平均后评分。

### 3.4 可恢复并发

- 先按 smoke 实测显存决定 GPU worker 数；
- CPU 并行构建 outer-fold cache、null 和汇总；
- GPU 仅用于训练；
- 使用可恢复 `tmux`/`nohup` launcher；
- watcher 每 60 秒记录 done/failed、当前 epoch、RAM/VRAM、ETA；
- 重连后根据 `DONE.json` 跳过完成单元；
- 任何并发增加都先做 10 分钟稳定性观察，避免 OOM。

科学效果大小不构成中途停止条件。只有 leakage、坐标混用、NaN/OOM 无法通过资源调整解决、或 loss/null 合同错误时停止。

---

## Milestone 4：按预定层级评分

### 4.1 评分顺序

```text
per-seizure exact maxAB |Spearman|
-> per-patient median
-> 15-patient cohort statistic
```

主 endpoint 只用 1–45 Hz。

### 4.2 Primary

```text
M3 - M0
```

输出：

- patient median delta；
- bootstrap 95% CI；
- 正/负/并列数；
- tie-tolerant exact Wilcoxon；
- 每位患者的原始 score 和 delta。

### 4.3 必报解释性 comparisons

```text
M3 - M1   # target supervision 是否改变 recurrent dynamics
M3 - M2   # recurrence 是否超过简单时间汇总
M1 - M0   # frozen target-blind state 是否可读
M2 - M0   # 活动负荷/最近事件汇总是否可读
M3_true_order - M3_order_shuffle
M3_correct_history - M3_within_patient_swapped_history
```

每项使用完全相同的 eligible seizures/contact masks；swap 只在有至少两次合格发作的患者中汇总并单独写分母。

### 4.4 Matched channel null

主 endpoint 每患者 5000 draws：

- 每次 seizure 独立打乱 target contact labels；
- 所有模型共享同一 permutations；
- 每 draw 重算 A/B candidate 与 absolute-sign selection；
- 先 seizure maxAB，再 patient median。

### 4.5 1–150 Hz sensitivity

在所有模型、seeds 和训练配置冻结后：

- 不重训；
- 用 1–45 Hz 训练所得 A/B candidate fields；
- 对 1–150 Hz target 重复 exact scoring；
- 只报告效应方向和与 primary 的一致性；
- 不用它改正文主结论。

### 4.6 不设复合 gate

所有 contrasts 独立报告。任何一个 secondary 阴性都不把其他结果改写为 `FAIL/NOT RUN`，也不触发架构、seed、loss 或 endpoint 搜索。

---

## Milestone 5：图、报告与停止边界

### 5.1 六联图

| Panel | 科学含义 |
|---|---|
| A | 集合值任务：static A/B + causal history residual，不宣称唯一方向 |
| B | M0/M1/M2/M3 四模型及它们分别回答的问题 |
| C | 15 人 primary：M3 与 M0 的 heldout maxAB 配对比较 |
| D | M3−M1 与 M3−M2：联合 dynamics 是否必要 |
| E | true-order vs full-history shuffle；correct vs within-patient swapped history |
| F | observed 相对 matched channel null，并标注 1–45 Hz primary 与 retrospective-static 边界 |

representative patient 只用于展示模型如何修正两个候选场，不替代 cohort 统计。图实际生成后写中文 `figures/README.md`，逐张做原分辨率目视 QA。

### 5.2 最终报告必须回答

1. 1–45 Hz static A/B anchor 是否复现；
2. M3 是否在 heldout patients 上改善 M0；
3. M1 是否已足够，即 target-blind recurrent state 是否本来就可读；
4. M2 是否已足够，即简单时间加权汇总是否解释增量；
5. 正确历史是否优于同患者其他发作历史；
6. 真实顺序是否优于完整历史置换；
7. 结果属于 patient-level static refinement、history-content effect，还是 seizure-matched history-conditioned refinement；
8. retrospective static A/B 对整体前瞻解释造成什么边界。

### 5.3 输出

```text
results/topic5_history_conditioned_field_refinement_v0_4/
├── INPUT_MANIFEST.json
├── cache/outer_<heldout_patient>/
├── development_smoke/
├── per_subject/
├── logs/
├── figures/
│   └── README.md
├── history_conditioned_field_patient_metrics.csv
├── history_conditioned_field_channel_null.csv
├── HISTORY_CONDITIONED_FIELD_SUMMARY.json
└── REPRODUCIBILITY_MANIFEST.json
```

### 5.4 完成后停止

完成 15 folds × 3 seeds、主统计、两类 history controls、1–150 Hz no-retrain sensitivity、图和报告后停止。本轮禁止：

- 根据结果增加 hidden size、GRU 层或 architecture；
- 搜索 learning rate、epoch、时间尺度或最好 seed；
- 恢复 signed single-field loss；
- 把 zero-state 重新包装成科学对照；
- 改 A/B、contact denominator、target window 或 null；
- 因某项 secondary 阴性删除 primary 结果。

收口时按 spec 的预定解释层级给出结论，不追加追阳性的实验。
