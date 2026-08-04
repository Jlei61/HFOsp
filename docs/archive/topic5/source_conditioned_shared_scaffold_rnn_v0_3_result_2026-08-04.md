# Topic 5 / Figure 6：source-conditioned 结构化 RNN v0.3 执行结果

**日期**：2026-08-04
**分支**：`codex/topic5-structured-rnn-fig6`
**结果根**：`results/topic5_patient_specific_source_conditioned_rnn_v0_3_final/`（学习率 3e-2）
**保留证据根**：`results/topic5_patient_specific_source_conditioned_rnn_v0_3/`（学习率 3e-3，预注册网格那一轮，306/306 完整）

---

## 0. 朴素话摘要

**测了什么。** 每位患者身上只用发作间期的放电事件，训练一个模型学「这个人的放电通常从哪里起、按什么先后顺序蔓延到哪里」。模型不许看发作时的数据，也不许看我们人工标的方向标签。训练完把模型冻住，让它自己从两个不同的起点各推演一遍，得到两张空间图；再看这两张图跟同一个人发作最初十秒的能量分布像不像。

**怎么测的。** 三层对照。第一层：把事件里真实的先后顺序打乱重训一遍——如果模型根本没在用顺序，打乱前后应该没差别。第二层：跟一个完全不看顺序、只统计「每个触点平时参与放电的频率」的静态基线比。第三层：跟一个不带任何结构约束的普通稠密网络比。跨状态那一步，每位患者都跟自己的随机重排基线比（把触点标签打乱 5000 次），看实测相关有没有超出自己的随机水平。

**揭示了什么。** 分成清楚的两半：

- **间期这一半是正面的。** 打乱顺序后模型明显变差（34 人里 33 人一致），说明这一版**真的在用事件内的先后顺序**——上一版几乎完全没用（差距只有万分之一）。它也稳定优于静态基线（34/34）。
- **跨状态那一半是阴性的。** 冻结模型推出来的空间图，跟发作早期能量分布的吻合程度**没有超过它自己的随机基线**（15 人里 0 人超过各自的 95 分位）。反倒是最简单的静态基线——单纯「哪些触点平时放电多」——是三者里唯一稳定超过随机的。也就是说：**方向信息没有转化成跨状态的优势**，能预测发作早期能量的仍然只是那个粗糙的「平时哪里活跃」的锚。

- **还有一件事我们原以为成立、实测不成立**：从两个相反起点出发推出来的两张图，**并不是彼此相反的**（全队列秩相关中位数 +0.04，真正相反的只有 34 人里的 2 人）。所以「同一套支架能给出两个相反方向场」这句话，在这批数据上没有被支持。

（内部归档代号：v0.3 source-conditioned shared-scaffold RNN、`W_S`/`W_A`、`d_e`、`learned_axis_source_pools`、`normalized_laplacian_source_pools`、`all_contact_margin`）

---

## 1. 为什么要有 v0.3

v0.2 把对称 scaffold 直接当 next-contact transition operator，首批完整患者显示它退化到接近 static，真实顺序与 rank-shuffle 相差约 0.0001。该批结果冻结为 symmetric-only diagnostic，未进入本文档任何数字。

v0.3 每位患者学一条有符号 contact coordinate，同一组端点 membership 同时派生对称 `W_S` 与反对称 `W_A`；每场事件由第一 rank set 因果决定方向标量 `d_e` 并在事件内冻结；propagation 走 `W_S + λ_A d_e W_A`，restraint 只走 `W_S`。

## 2. 训练与执行

- 正式单位：34 patients × 3 models × 3 seeds = **306/306 COMPLETE，0 FAILED，0 OOM，0 non-finite**，全部 7/7 cycle。
- 学习率：预注册网格 `3e-4 / 1e-3 / 3e-3` 选中自身最大值且三患者单调，属于欠训信号。事后扩展到 `1e-2 / 3e-2 / 1e-1`，曲线在 **3e-2 见底后回升**（1.7828 / 1.7672 / 1.6880 / 1.6012 / **1.5980** / 1.5989），是内部极小值。**网格扩展是事后行为**，且只在 structured 上调参后套用到对照臂，对 structured 略有利——两条都是必须随结论一起报告的限制。
- Target seal：见 §6，本轮有一次已记录的破封。

## 3. 间期预测（Figure 6 面板 C）

patient-level test20 contact-choice NLL（条件于 continue 与观察到的 cardinality，不含 STOP）：

| model | contact NLL 中位数 | top-1 |
|---|---|---|
| ordinary_gru | 1.6173 | 0.3983 |
| **structured** | **1.7499** | 0.3624 |
| structured_rank_shuffle | 1.7781 | 0.3185 |
| static | 1.8052 | 0.2737 |

配对差（正号表示 structured 更好）：

| 对照 | 34 人描述 | 31 人确认（剔除 3 位 development 患者） |
|---|---|---|
| vs structured_rank_shuffle | **+0.0456** [+0.0381,+0.0594] P=3.5e-10, 33+/1− | **+0.0437** [+0.0367,+0.0550] P=2.8e-9, 30+/1− |
| vs static | **+0.1001** [+0.0595,+0.1300] P=1.2e-10, 34+/0− | **+0.0913** [+0.0558,+0.1204] P=9.3e-10, 31+/0− |
| vs ordinary_gru | **−0.1205** [−0.1604,−0.0732] P=1.2e-10, 0+/34− | **−0.1187** [−0.1580,−0.0723] P=9.3e-10, 0+/31− |

**读法**：结构化模型确实在使用事件内顺序（对 rank-shuffle 的优势比 v0.2 大两个数量级），也稳定优于静态基线；但在纯预测上一致输给稠密 GRU。后者按 plan 预注册**不作为继续下游的门槛**，不得据此宣称结构化模型"更好"或"更差"于任务本身——它换取的是可读出的方向结构。

### 3.1 输给稠密 GRU 不是训练不足，是容量差（2026-08-04 补测）

`results/topic5_patient_specific_source_conditioned_rnn_v0_3_sweep/`（三位 development 患者，只读间期 validation20）

把优化步数从 224 提到 2688（12×）：

| model | 224 步 | 896 步 | 2688 步 | 2688 − 224 |
|---|---|---|---|---|
| structured | 1.59796 | 1.58401 | 1.58142 | **−0.0166** |
| ordinary_gru | 1.31635 | 1.30979 | 1.30979 | **−0.0066** |

逐患者 896 → 2688 的变化在 −0.0001 到 −0.0026 之间，两个模型都已收敛。**两者之间的差距是 0.27 nats，比训练预算能移动的量大一个数量级以上**，因此该差距不是欠训练造成的。

参数量给出直接解释：structured 共 **87 个可学参数**，其中决定"下一个是哪个触点"的通路只有 **2 个自由标量**（`propagation_weight`、`restraint_weight`）叠加在一张学出来的图上；稠密 GRU 共 **5,728 个**，光触点解码器就有 480 个自由权重（容量比 **65.8×**）。0.27 nats 是可解释性的代价，不是训练缺陷。

学习率已单独扫过六点并落在内部极小值（§2）。**未测**：micro-batch 大小（冻结 runner 无该覆盖开关，扫描脚本中曾有一个不生效的 batch 维度，已删除以免记录假扫描）与状态归一化（属架构改动，会改变 §5 之外的合同）。

## 4. 跨状态对应（Figure 6 面板 E）

15 位 primary 患者，E1146 仅作 supportive 且不进入任何 P 值。指标为每位患者实测相关减去其自身 5000 次 all-contact 置换的中位数。

| source pool 规则 | model | median margin | P (exact one-sided) | 超过自身 95 分位 |
|---|---|---|---|---|
| **primary：learned axis** | static | **+0.1574** | **0.0013** | 5/15 |
| | ordinary_gru | +0.0734 | 0.281 | 3/15 |
| | **structured** | **+0.0180** | **0.404** | **0/15** |
| sensitivity：diffusion graph | static | +0.1574 | 0.0013 | 5/15 |
| | ordinary_gru | +0.2515 | 6.1e-05 | 7/15 |
| | structured | +0.2340 | 6.1e-05 | 5/15 |

structured vs ordinary 配对：primary 中位数 −0.0108（P=0.79，7+/8−），sensitivity −0.0183（P=0.98，3+/9−）。**两套规则下 structured 都没有赢过 ordinary。**

**读法**：在预注册的 primary 规则下，冻结模型的空间图没有超过自身随机基线；唯一稳定超过随机的是静态参与度基线。在 sensitivity 规则下三者数值都抬高，但那套 source pool 由固定杆/几何主导的连接图切分而来，抬高的部分与本仓库既有的「相似大半是电极杆几何」发现同源，且 structured 仍未超过 ordinary。综合两套规则：**没有证据表明间期传播的方向结构比「平时哪里活跃」这个粗糙锚点更能对应发作早期能量分布。**

## 5. 两个诊断量推翻了一个设计前提

### 5.1 两张场并不相反

选择 primary source pool 规则时的论证是：从学到的轴两端出发能得到强烈相反的方向标量（`d_e` = ±0.61–0.84，而连接图规则只有 0.03–0.28），因此两张场不会塌成同一张。**这个推理只验证了输入，没有验证输出。**

实测 `spearman(F⁻, F⁺)`：

| 规则 | 中位数 | < −0.5（真正相反） | < 0 |
|---|---|---|---|
| learned axis（primary） | **+0.042** | **2/34** | 15/34 |
| diffusion graph（sensitivity） | −0.048 | 5/34 | — |

两套规则下场都没有相反。原因是 propagation 更新里对称支架与 participation bias 占主导，`λ_A d_e W_A` 只是扰动项；方向标量翻号不足以翻转推演出的场。

**这是一次 CLAUDE.md §6.3 式的层级合并错误**：把「方向标量翻号」与「两张场相反」当成同一件事陈述，而只测了前者。Figure 6 面板 D 已加入实测数值注解，不再暗示反向；README 同步改写。

### 5.2 一半队列的三个 seed 没找到同一条轴

`min_seed_axis_pairwise_pearson`（符号对齐后 seed 间最小两两相关）：中位数 0.592，**16/34 低于 0.5，5/34 为负**。对这部分患者，seed-ensemble 平均轴不是一条有意义的单一支架，而 primary source pool 正是由它定义的。该诊断量是设计时刻意加入并要求"报告而不作门槛"的，此处按约定报告。

## 6. Target seal 事件（必须随结论引用）

为避免 Figure 6 在无人值守链条末尾首次执行，2026-08-04 对已完成的 3e-3 轮做了一次排练；其中面板 D 的取数函数**反序列化了 E1146 的发作早期数值**，时间点早于最终轮的 field manifest 冻结。记录在 `results/.../TARGET_SEAL_INCIDENT.json`。

- 范围：仅 E1146 一人；E1146 设计上即 supportive-only，不进入任何 primary P 值；15 位 primary 患者未被读取。
- 影响评估：读取之前，config / 代码 SHA / source pool 规则 / 代表患者 / horizon / 全部哈希均已冻结并记录，读取之后没有任何模型、场、checkpoint、学习率或代表患者的选择发生。
- **不可再声称**：「在 field manifest 之前没有任何 target 值被反序列化过」——对 E1146 已不成立。

## 7. 未实现 / 待办

- **rollout-vs-test20 一致性统计**（participation、pairwise precedence、expected-rank distance）：任何现有脚本都未实现，Figure 6 统计 JSON 中显式标为 `not_implemented`，面板 C 只含留出集似然与 top-1。
- Figure 6 面板 B 的事件分组仅由观察到的 first-rank source membership 决定；经验 A/B 至今未做事后 read-back。

## 8. 可复现入口

```bash
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic5_source_conditioned_shared_scaffold_rnn_v0_3_final.yaml
$PY scripts/launch_topic5_shared_scaffold_rnn_v0_2.py --config $CFG --workers 40 --resume
bash scripts/run_topic5_v0_3_final_pipeline.sh     # 汇总 → 两套场冻结 → 两份清单 → 解封 → 制图
```

冻结记录：`FORMAL_RUN_FREEZE.json`（代码/配置 SHA256）、`field_freeze*/FROZEN_FIELD_MANIFEST.json`、`early_ictal*/TARGET_UNLOCK_RECORD.json`。
